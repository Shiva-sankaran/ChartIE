from itertools import combinations
from tkinter.messagebox import NO
import torch, cv2, os, sys, json
import argparse, easyocr, random
import numpy as np
import tqdm as tq
from PIL import Image, ImageDraw, ImageFont
from numpy import unravel_index
from models.my_model import Model
from models.networks import Branched_Conv_Relation_Net
from models_det import build_model
from configs.det import parser_det
from configs.kp import parser_kp
from utils.misc import *
from utils import box_ops
import utils.misc as utils
from sklearn.cluster import KMeans
from scipy import ndimage
sys.path.append('/home/vp.shivasan/ChartIE')
sys.path.append('/home/md.hassan/charts/ChartIE')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--show_keypoints', default=True, type=bool)

args = parser.parse_args()
num_gpus = torch.cuda.device_count()

CUDA_ = 'cuda:3'

# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# image_path = '/home/md.hassan/charts/metrics/det/chartie/Adobe_synth/test/images'
image_path = '/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/images'
# image_path = '/home/md.hassan/charts/data/data/FigureSeerDataset/Annotated images'

image_save_path = '/home/md.hassan/charts/ChartIE/results_synth/'

# txt_save_path = '/home/md.hassan/charts/metrics/det/chartie/Adobe_synth/test/detections_postprocess/'
# txt_save_path = '/home/md.hassan/charts/metrics/det/chartie/synth/old_val_data/detections_postprocess/'
# txt_save_path = '/home/md.hassan/charts/metrics/det/chartie/FigureSeer/detections_postprocess/'

for f in os.listdir(image_save_path):
    os.remove(image_save_path + f) 

# Loading kp model
args_kp = parser_kp.parse_args()
kp_model  = Model(args_kp)
kp_state = torch.load('/home/vp.shivasan/ChartIE/checkpoint/checkpoint_l1_bigger_80.t7', map_location = 'cpu')
kp_model.load_state_dict(kp_state['state_dict'])
kp_model = kp_model.to(CUDA_)
kp_model.eval()
print("Loaded kp model at Epoch:", kp_state["epoch"])

# Loading element detection model
args_det = parser_det.parse_args()
args_det.device = CUDA_
det_model, _, _ = build_model(args_det)
checkpoint = torch.load(args_det.weights, map_location='cpu')
det_model.load_state_dict(checkpoint['model'])
det_model.to(CUDA_)
det_model.eval()
print("Loaded element detection model at Epoch {}".format(checkpoint['epoch']))

CLASSES = ['Legend', 'ValueAxisTitle', 'ChartTitle', 'CategoryAxisTitle', 'PlotArea', 
                'InnerPlotArea', 'XY_Ticks', 'LegendMarker', 'LegendText', 'LegendElement']
colors = [(0, 122 , 122), (122, 0, 122), (0, 122 , 122), (255, 0 , 255), (0, 255, 255), (255, 255, 0),
          (122, 122, 0), (255, 0, 0), (0, 255, 0), (0, 0 , 255)]

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
reader = easyocr.Reader(['en'], gpu = 'cuda:0') # only english for now

def normalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]    
    image /= 255.0
    image -= mean
    image /= std
    return image

def unnormalize(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]    
    image *= std
    image += mean
    return image

def plot_box(image, cls, box):
	if cls >= len(CLASSES):
		return
		box = box.cpu()
	cv2.rectangle(image, (int(box[0] - box[2]//2), int(box[1] - box[3]//2)), (int(box[0] + box[2]//2), int(box[1]+box[3]//2)), colors[cls], 1)            
    
def prepare_txt(txt_str, boxes, cls):
	if len(boxes) == 0:
		return txt_str
	for box in boxes:
		box = (int(box[0] - box[2]//2), int(box[1] - box[3]//2), int(box[0] + box[2]//2), int(box[1]+box[3]//2))
		txt_str += str(cls)+" 1.0 "+str(box[0])+" "+str(box[1])+" "+str(box[2])+" "+str(box[3])
		txt_str += "\n"
	return txt_str

def prepare_txt_unique(txt_str, boxes):
	for box, i in enumerate(boxes):
		box = (int(box[0] - box[2]//2), int(box[1] - box[3]//2)), (int(box[0] + box[2]//2), int(box[1]+box[3]//2))
		txt_str += str(i)+" "+box[0]+" "+box[1]+" "+box[2]+" "+box[3]
		txt_str += "\n"
	return txt_str

def find_max(Arr):
	i, j = unravel_index(Arr.argmax(), Arr.shape)    
	max_value = Arr[i][j]        
	Arr[i] = -1000*np.ones((1, Arr.shape[1]))
	Arr[:, j] = -1000*np.ones((Arr.shape[0], ))
	return Arr, max_value, i, j

def find_iou(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.arange(0, len(x1))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def match_text_boxes(ocr_boxes, det_boxes):
	if len(ocr_boxes) == 0 or len(det_boxes) == 0:
		return []
	det_boxes = torch.as_tensor(np.array(det_boxes))
	det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes)
	giou = box_ops.generalized_box_iou(torch.as_tensor(ocr_boxes), det_boxes)
	return giou

def get_best_rotated_ocr(crop):
	max_conf = 0
	result = []
	angle_ = 0
	rotated_ = []
	for angle in [0, -90, -30, -60, -45]:
		rotated = ndimage.rotate(crop, angle)
		r = reader.readtext(rotated)
		if len(r) != 0:
			conf = r[0][2]
			if conf >= max_conf:
				result = r
				max_conf = conf
				angle_ = angle
				rotated_ = rotated
	return result, rotated_, angle_

def handle_ticks(image_, ocr_text, ocr_box_ids, tick_bboxes, det_box_ids):
	# for handling ticks ocr: need to resize
	tick_texts = [ocr_text[idx] for idx in ocr_box_ids]
	final_ticks = [tick_bboxes[idx] for idx in det_box_ids]
	for idx in det_box_ids:
		tick_bboxes = np.delete(tick_bboxes, idx, 0)

	# (w_temp, h_temp) = bbox[2:] * 0.75
	(w_offset, h_offset) = (15, 15)
	w_resize = 100
	h, w, _ = image_.shape
	for bbox in tick_bboxes:
		try:
			crop = image_[int(bbox[1]-h_offset-bbox[3]//2):int(bbox[1]+h_offset+bbox[3]//2), int(bbox[0]-w_offset-bbox[2]//2):int(bbox[0]+w_offset+bbox[2]//2)]
			w_orig = bbox[0]+w_offset+bbox[2]//2 - (bbox[0]-w_offset-bbox[2]//2)
			h_orig = bbox[1]+h_offset+bbox[3]//2 - (bbox[1]-h_offset-bbox[3]//2)
			crop = cv2.resize(crop, (w_resize, int(w_resize*bbox[3]/bbox[2])))
		except:
			continue
		r_ = reader.readtext(crop, width_ths=0.8)

		if len(r_) == 0:
			gray = cv2.cvtColor(crop.copy(), cv2.COLOR_BGR2GRAY)
			gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			r_ = reader.readtext(crop, width_ths=0.8)
		if len(r_) == 0:
			gray = cv2.cvtColor(crop.copy(), cv2.COLOR_BGR2GRAY)
			gray = cv2.medianBlur(gray, 3)
			r_ = reader.readtext(crop, width_ths=0.8)
		if len(r_) > 0:
			# cv2.putText(image_, r_[0][1], bbox[:2].astype(np.uint32), font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)    
			r_ = sorted(r_, key=lambda l:l[2])
			tick_texts.append(r_[-1][1])
			crop_poly = r_[-1][0]

			cx_crop = (crop_poly[0][0] + crop_poly[2][0])/2
			cy_crop = (crop_poly[1][1] + crop_poly[3][1])/2
			w_crop = (crop_poly[2][0] - crop_poly[0][0])
			h_crop = (crop_poly[3][1] - crop_poly[1][1])
			cx_final = np.clip(bbox[0] - w_offset-bbox[2]//2 + cx_crop/w_resize*w_orig, 0, w)
			cy_final = np.clip(bbox[1] - h_offset-bbox[3]//2 + cy_crop/(w_resize*bbox[3]/bbox[2])*h_orig, 0, h)
			w_final = np.clip(w_crop/w_resize*w_orig, 0, w)
			h_final = np.clip(h_crop/(w_resize*bbox[3]/bbox[2])*h_orig, 0, h)
			tick_box = np.array([cx_final, cy_final, w_final, h_final])
			final_ticks.append(tick_box)

	final_ticks = np.array(final_ticks)

	return image_, final_ticks, tick_texts

def get_matched_boxes(samples_, giou_matrix, ocr_boxes, det_boxes):
	if len(giou_matrix) == 0 or len(ocr_boxes) == 0 or len(det_boxes) == 0:
		return [], [], []
	count = 0
	ocr_boxes = ocr_boxes.astype(np.uint32)
	det_boxes = torch.as_tensor(det_boxes)
	det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes)
	det_boxes = np.array(det_boxes).astype(np.uint32)
	text_boxes = []
	det_box_ids = []
	ocr_box_ids = []
	while(count < min(giou_matrix.shape)):
		giou_matrix, score, ocr_idx, det_box_idx = find_max(np.array(giou_matrix))
		if score > 0:
			text_boxes.append(ocr_boxes[ocr_idx])
			# cv2.rectangle(samples_, ocr_boxes[ocr_idx][0:2], ocr_boxes[ocr_idx][2:],(0, 255, 0), 1)
			# cv2.rectangle(samples_, det_boxes[det_box_idx][0:2], det_boxes[det_box_idx][2:], (0, 0, 255), 1)
		count += 1
	return text_boxes, det_box_ids, ocr_box_ids

def filter_outliers(array, m = 2):
	d = np.abs(array - np.median(array))
	mdev = np.median(d)
	s = d / (mdev if mdev else 1.)
	return [s < m]

def get_ticks_text_coord(ticks_text, ticks_boxes, axis_idx):
	texts = []
	coords = []
	for text, coord in zip(ticks_text, ticks_boxes):
		try:
			text = text.replace('o', '0')
			text = text.replace('O', '0')
			text = text.replace('l', '1')
			text = text.replace('I', '1')
			text = text.replace(',', '')
			text = text.replace(' ', '')
			text = text.replace('%', '')
			texts.append(float(text))
			coords.append(coord)
		except:
			continue
	idx = []
	unique_texts, ids = np.unique(np.array(texts), return_index=True)
	ticks_boxes = np.array(coords)[ids]
	if len(ticks_boxes) < 2:
		return [], []
	texts = unique_texts[ticks_boxes[:, axis_idx].argsort()]
	coords = ticks_boxes[ticks_boxes[:, axis_idx].argsort()]
	if axis_idx == 1:
		texts = texts[::-1]
		coords = coords[::-1]

	filtered_ids = filter_outliers(texts)
	texts = np.array(texts)[tuple(filtered_ids)]
	coords = np.array(coords)[tuple(filtered_ids)]
	if len(texts) == 0:
		return [], []
	min_idx = np.argmin(texts)
	for i in range(len(texts)):
		if texts[i] >= np.min(texts) and i < min_idx:
			idx.append(i)
	if len(idx):
		texts = np.delete(texts, idx)
		coords = np.delete(coords, idx, axis = 0)
	return texts, coords

def get_ratio(text, coord):
	r_list = []
	for i in range(len(text)):
		for j in range(i + 1, len(text)):
			r_list.append(abs(text[j] - text[i])/abs(coord[j] - coord[i]))

	# find median for better approximation of ratio. also return indices of ticks corresponding to median for better scaling
	med_idx = np.argsort(r_list)[len(r_list)//2]
	med = r_list[med_idx]
	combination_list = list((i,j) for ((i,_),(j,_)) in combinations(enumerate(text), 2))
	best_tick_ids = combination_list[med_idx]
	return med, best_tick_ids

def run_element_det(model):
	image = cv2.imread(image_path + '/' + image_name)
	image_ = image.copy()
	image = image.astype(np.float32)
	image = normalize(image)
	image = torch.from_numpy(image).to(CUDA_)
	image = image.permute(2, 0, 1)
	image = nested_tensor_from_tensor_list([image])
	with torch.no_grad():
		outputs = model(image.to(CUDA_))

	pred_logits = outputs['pred_logits'][0][:, :len(CLASSES)]
	pred_boxes = outputs['pred_boxes'][0]

	max_output = pred_logits.softmax(-1).max(-1)
	topk = max_output.values.topk(100)

	pred_logits = pred_logits[topk.indices]
	pred_boxes = pred_boxes[topk.indices]
	pred_classes = pred_logits.argmax(axis=1)

	unique_idx = []
	for i in range(0, 6):
		if i in pred_classes:
			unique_idx.append(np.where(pred_classes.cpu() == i )[0][0])
	non_unique_classes = []
	for i in range(6, len(CLASSES)):
		if i in pred_classes:
			non_unique_classes.append(np.where(pred_classes.cpu() == i )[0])

	image = image.tensors[0].permute(1, 2, 0).cpu().numpy()
	h, w, _ = image.shape
	image_ = unnormalize(image.copy())*255.0
	image_ = image_.astype(np.uint8)
	pred_boxes = pred_boxes.cpu() * torch.Tensor([w, h, w, h])

	# get boxes and text from OCR
	ocr_results = reader.readtext(image_, width_ths=1.2, rotation_info = [270]) # assuming 270 deg rotation for vertical text and (possibly) rotated ticks text
	# # plot OCR result
	# for result in ocr_results:
	# 	r = np.array(result[0]).astype(int)
	# 	cv2.rectangle(image_, r[0], r[2], (0, 255, 0), 1)  
	# 	cv2.putText(image_, result[1], r[0], font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)          

	# plot detr predictions (unique boxes)
	if len(ocr_results) > 0: 
		ocr_boxes = np.array([np.hstack((r[0][0], r[0][2])) for r in ocr_results])
		ocr_boxes = ocr_boxes[(ocr_boxes[:, 2] >= ocr_boxes[:, 0])*(ocr_boxes[:, 3] >= ocr_boxes[:, 1])]
		ocr_text = np.array([r[1] for r in ocr_results])
	else:
		ocr_boxes = []
		ocr_text = []

	unique_boxes = {}
	for cls, box in zip(pred_classes[unique_idx], pred_boxes[unique_idx]):
		# plot_box(image_, cls, box)
		if cls == 1:
			giou_ytitle = match_text_boxes(ocr_boxes, box.unsqueeze(0))
			ocr_box = ocr_boxes[giou_ytitle.argmax()]
			box = box_ops.box_xyxy_to_cxcywh(torch.as_tensor(ocr_box))
			# if find_iou(ocr_box, box) > 0: 
			# 	ytitle_text = ocr_text[giou_title.argmax()]
			# else:
			# 	ytitle_text = None
		# elif cls == 2:
		# 	giou_title = match_text_boxes(ocr_boxes, box.unsqueeze(0))
		# 	ocr_box = ocr_boxes[giou_title.argmax()]
		# 	if find_iou(ocr_box, box) > 0: 
		# 		title_text = ocr_text[giou_title.argmax()]
		# 	else:
		# 		title_text = None
		# elif cls == 3:
		# 	giou_xtitle = match_text_boxes(ocr_boxes, box.unsqueeze(0))
		# 	ocr_box = ocr_boxes[giou_xtitle.argmax()]
		# 	if find_iou(ocr_box, box) > 0: 
		# 		xtitle_text = ocr_text[giou_xtitle.argmax()]
		# 	else:
		# 		xtitle_text = None
		unique_boxes[int(cls)] = box

	# plot detr predictions (non unique boxes)
	tick_bboxes = []
	legend_marker_bboxes = []
	legend_text_bboxes = []     
	legend_ele_bboxes = []   
	for class_idx, box_idx in enumerate(non_unique_classes):
		cls_boxes = pred_boxes[non_unique_classes[class_idx]]
		dets = box_ops.box_cxcywh_to_xyxy(cls_boxes) #tl, br
		# convert to bl, tr for nms
		dets[:, 1] = h - dets[:, 1]
		dets[:, 3] = h - dets[:, 3]
		dets = torch.index_select(dets, 1, torch.LongTensor([0, 3, 2, 1]))
		nms_idx = nms(dets, 0.0)
		cls = pred_classes[box_idx][0]
		for box in cls_boxes[nms_idx]:    
			# plot_box(image_, cls, box)
			if cls == 6: # tick bbox
				tick_bboxes.append(np.array(box))
			elif cls == 7: # legend marker bbox
				legend_marker_bboxes.append(np.array(box))
			elif cls == 8: # legend text bbox
				legend_text_bboxes.append(np.array(box))
			elif cls == 9: # legend element bbox
				legend_ele_bboxes.append(np.array(box))

	# match OCR predicted legend text boxes and DETR predicted legend text boxes
	ocr_offset_box = [-5, -5, 5, 5]  #xyxy
	new_ocr_boxes = ocr_boxes.copy() + ocr_offset_box
	giou_matrix_legend_text = match_text_boxes(np.clip(new_ocr_boxes,0,np.max(new_ocr_boxes)), legend_text_bboxes.copy())
	# match OCR predicted tick text boxes and DETR predicted tick text boxes
	tick_offset_box = [0, 0, 10, 10] # cxcywh

	new_tick_boxes = np.array(tick_bboxes).copy() + tick_offset_box
	giou_matrix_tick = match_text_boxes(np.clip(new_ocr_boxes,0,np.max(new_ocr_boxes)), np.clip(new_tick_boxes,0,np.max(new_tick_boxes)))

	# find matched boxes
	legend_text_bboxes = np.array(legend_text_bboxes).astype(np.int32)
	matched_boxes_leg_text, _, _ = get_matched_boxes(image_, giou_matrix_legend_text, ocr_boxes, legend_text_bboxes)

	tick_bboxes = np.array(tick_bboxes)
	_, det_box_ids, ocr_box_ids = get_matched_boxes(image_, giou_matrix_tick, ocr_boxes, tick_bboxes)
		
	giou_matrix_legend_marker = match_text_boxes(np.array(matched_boxes_leg_text, dtype=np.int32).copy(), 
								legend_marker_bboxes.copy())
	count = 0
	final_marker = []
	final_leg_text = []
	final_leg_text_box = []
	ocr_boxes = ocr_boxes.astype(np.int32)
	while(count < min(np.array(giou_matrix_legend_marker).shape)):
		giou_matrix_legend_marker, score, text_box_idx, marker_idx = find_max(np.array(giou_matrix_legend_marker))
		try:
			final_leg_text_box.append(ocr_boxes[np.where(np.all(ocr_boxes == matched_boxes_leg_text[text_box_idx], axis=1))][0])
		except:
			count += 1
			continue
		final_marker.append(legend_marker_bboxes[marker_idx])
		final_leg_text.append(ocr_text[np.where(np.all(ocr_boxes == matched_boxes_leg_text[text_box_idx], axis=1))])
		count += 1
	if len(final_leg_text_box) == 0:
		final_leg_text_box = []
	else:
		# if len(matched_boxes_leg_text) != len(final_leg_text_box):
		# 	unmatched_ids = []
		# 	for id, box in enumerate(matched_boxes_leg_text):
		# 		if box not in np.array(final_leg_text_box):
		# 			unmatched_ids.append(id)
		# 	x_sep = np.mean(np.array(final_leg_text_box)[:, 0] - np.array(final_marker)[:, 0] - np.array(final_marker)[:, 2]/2)
		# 	for id in unmatched_ids:
		# 		# if len(matched_boxes_leg_text) != len(final_leg_text_box):
		# 			final_leg_text_box.append(ocr_boxes[np.where(np.all(ocr_boxes == matched_boxes_leg_text[id], axis=1))][0])
		# 			final_leg_text.append(ocr_text[np.where(np.all(ocr_boxes == matched_boxes_leg_text[id], axis=1))])
		# 			final_marker.append([box[0]-np.mean(np.array(final_marker)[:, 2])-x_sep, (box[1] + box[3])/2 - np.mean(np.array(final_marker)[:, 3]), box[0]-x_sep, (box[1] + box[3])/2 + np.mean(np.array(final_marker)[:, 3])])
		final_leg_text_box = box_ops.box_xyxy_to_cxcywh(torch.tensor(np.array(final_leg_text_box)))

	image_, final_ticks, tick_texts = handle_ticks(image_, ocr_text, ocr_box_ids, tick_bboxes, det_box_ids)
	final_ticks = np.array(final_ticks)
	x_temp = final_ticks[:, 0]
	y_temp = h - final_ticks[:, 1]
	slopes = y_temp/x_temp
	xticks = slopes < h/w
	yticks = slopes > h/w

	xticks_boxes = final_ticks[xticks]
	xticks_text = np.array(tick_texts)[xticks][xticks_boxes[:, 0].argsort()]
	xticks_boxes = xticks_boxes[xticks_boxes[:, 0].argsort()]
	yticks_boxes = final_ticks[yticks]
	yticks_text = np.array(tick_texts)[yticks][yticks_boxes[:, 1].argsort()]
	yticks_boxes = yticks_boxes[yticks_boxes[:, 1].argsort()]

	# DEBUG: drawing detection boxes
	xticks_boxes_ = xticks_boxes.copy().astype(np.uint32)
	yticks_boxes_ = yticks_boxes.copy().astype(np.uint32)
	for i in range(len(xticks_boxes_)):
		plot_box(image_, 6, xticks_boxes_[i])
		# cv2.rectangle(image_, xticks_boxes_[i][0:2]-xticks_boxes_[i][2:]//2, xticks_boxes_[i][0:2]+xticks_boxes_[i][2:]//2,(0, 255, 0), 1)
		cv2.putText(image_, xticks_text[i], xticks_boxes_[i][0:2], font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)      
	for i in range(len(yticks_boxes_)):
		plot_box(image_, 6, yticks_boxes_[i])
		# cv2.rectangle(image_, yticks_boxes_[i][0:2]-yticks_boxes_[i][2:]//2, yticks_boxes_[i][0:2]+yticks_boxes_[i][2:]//2,(0, 255, 0), 1)
		cv2.putText(image_, yticks_text[i], yticks_boxes_[i][0:2], font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)      
	for marker_box, text, text_box in zip(final_marker, final_leg_text, final_leg_text_box):
		# cv2.rectangle(image_, box[i][0:2]-box[i][2:]//2, box[i][0:2]+box[i][2:]//2,(0, 255, 0), 1)
		plot_box(image_, 7, marker_box)
		plot_box(image_, 8, text_box)
		cv2.putText(image_, str(text)[2:-2], marker_box[:2].astype(np.uint32), font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)
	for i in unique_boxes.keys():
		plot_box(image_, i, unique_boxes[i])
	cv2.imwrite(image_save_path + 'det_' + image_name, image_)

	x_text, x_coords = get_ticks_text_coord(xticks_text, xticks_boxes, 0)
	y_text, y_coords = get_ticks_text_coord(yticks_text, yticks_boxes, 1)
	if len(x_text) < 2 and len(y_text) >= 2:
		print("Not enough x ticks extracted. Unable to scale x axis")
		y_ratio, y_med_ids = get_ratio(y_text, np.array(y_coords)[:, 1])
		yticks_info = [y_text, y_coords, y_ratio, y_med_ids]
		return final_marker, final_leg_text, final_leg_text_box, legend_ele_bboxes, [[],[],[],[]], yticks_info, unique_boxes
	elif len(y_text) < 2 and len(x_text) >= 2:
		print("Not enough y ticks extracted. Unable to scale y axis")
		x_ratio, x_med_ids = get_ratio(x_text, np.array(x_coords)[:, 0])
		xticks_info = [x_text, x_coords, x_ratio, x_med_ids]
		return final_marker, final_leg_text, final_leg_text_box, legend_ele_bboxes, xticks_info, [[],[],[],[]], unique_boxes
	elif len(y_text) < 2 and len(x_text) < 2:
		print("Not enough x ticks and y ticks extracted. Unable to scale both axes")
		return final_marker, final_leg_text, final_leg_text_box, legend_ele_bboxes, [[],[],[],[]], [[],[],[],[]], unique_boxes

	x_ratio, x_med_ids = get_ratio(x_text, np.array(x_coords)[:, 0])
	y_ratio, y_med_ids = get_ratio(y_text, np.array(y_coords)[:, 1])
	xticks_info = [x_text, x_coords, x_ratio, x_med_ids]
	yticks_info = [y_text, y_coords, y_ratio, y_med_ids]

	return final_marker, final_leg_text, final_leg_text_box, legend_ele_bboxes, xticks_info, yticks_info, unique_boxes

def keypoints(model,image_path,args):
	image= cv2.imread(image_path)
	
	h,w,_ = image.shape
	image = cv2.resize(image, (args.input_size[0], args.input_size[1]))
	image = np.asarray(image)
	image = image.astype(np.float32) / 255
	
	image = torch.from_numpy(image)
	image = image.permute((2, 0, 1))
	image = torch.unsqueeze(image,0)

	torch.tensor(image, dtype=torch.float32)
	image = image.to(CUDA_, non_blocking=True)
		
	output = model(image,return_attn =False)
	out_bbox = output["pred_boxes"][0]
	out_bbox = out_bbox.cpu().detach().numpy()
	x_cords = (out_bbox[:,0]*w)
	y_cords = (out_bbox[:,1]*h)
	return [tuple(x) for x in np.array((x_cords,y_cords)).T]

# for f in os.listdir(txt_save_path):
# 	os.remove(txt_save_path + f) 
for image_name in tq.tqdm(os.listdir(image_path)):
	im = cv2.imread(image_path + '/' + image_name)
	if im is None:
		continue
	legend_bboxes, legend_text, legend_text_boxes, legend_ele_boxes,  xticks_info, yticks_info, unique_boxes = run_element_det(det_model)

	# # FOR METRICS
	# txt_str = ""
	# for c, box in unique_boxes.items():
	# 	box = box_ops.box_cxcywh_to_xyxy(box)
	# 	txt_str += str(c)+" 1.0 "+str(int(box[0]))+" "+str(int(box[1]))+" "+str(int(box[2]))+" "+str(int(box[3]))+"\n"
	# txt_str = prepare_txt(txt_str, np.array(xticks_info[1]), 6)
	# txt_str = prepare_txt(txt_str, np.array(yticks_info[1]), 6)
	# txt_str = prepare_txt(txt_str, np.array(legend_bboxes), 7)
	# txt_str = prepare_txt(txt_str, np.array(legend_text_boxes), 8)
	# txt_str = prepare_txt(txt_str, np.array(legend_ele_boxes), 9)
	# txt_str = txt_str[:-1]
	# with open(txt_save_path +image_name[:-4]+'.txt', 'w') as f:
	# 	f.write(txt_str)

	x_text, x_coords, x_ratio, x_med_ids = xticks_info
	y_text, y_coords, y_ratio, y_med_ids = yticks_info

	all_kps = keypoints(model=kp_model,image_path=image_path +'/'+ image_name,args=args_kp)

	image_cls = Image.open(image_path +'/'+ image_name)
	image = cv2.imread(image_path +'/'+ image_name)
	h, w, _ = image.shape
	scaled_kps = np.array(all_kps).copy()

	# data1 = ratio * (pixel1 - pixel0) + data0
	try:
		scaled_kps[:, 0] = (np.array(all_kps)[:, 0] - x_coords[x_med_ids[0]][0]) * x_ratio + x_text[x_med_ids[0]]
		scaled_kps[:, 1] = -1 * (np.array(all_kps)[:, 1] - y_coords[y_med_ids[0]][1]) * y_ratio + y_text[y_med_ids[0]]		
		for i, kp in enumerate(all_kps):
			image = cv2.circle(image, (int(kp[0]), int(kp[1])), radius=3, color=(0,255,0), thickness=-1)
			cv2.putText(image, str(round(float(str(scaled_kps[i,0])),1))+', '+str(round(float(str(scaled_kps[i,1])),1)), (int(kp[0]), int(kp[1])), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
	except:
		temp = 0
	cv2.imwrite(image_save_path + 'kp_' + image_name, image)
