import torch, cv2, os, sys, json
import argparse, easyocr, random
import numpy as np
import tqdm as tq
from PIL import Image, ImageDraw, ImageFont
from numpy import unravel_index
from models.my_model import Model
from models.networks import Branched_Conv_Relation_Net
from models_det import build_model
from configs.detr import parser_det
from util.misc import *
from util import box_ops
import util.misc as utils
sys.path.append('/home/vp.shivasan/ChartIE')
sys.path.append('/home/md.hassan/charts/ChartIE')

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

parser.add_argument('--batch_size', default=42, type=int)
parser.add_argument('--patch_size', default=16, type=int)

parser.add_argument('--position_embedding', default='enc_xcit',
					type=str, choices=('enc_sine', 'enc_learned', 'enc_xcit',
										'learned_cls', 'learned_nocls', 'none'),
					help="Type of positional embedding to use on top of the image features")
parser.add_argument('--activation', default='gelu', type=str, choices=('relu', 'gelu', "glu"),
					help="Activation function used for the transformer decoder")

parser.add_argument('--input_size', nargs="+", default=[288, 384], type=int,
					help="Input image size. Default is %(default)s")
parser.add_argument('--hidden_dim', default=384, type=int,
					help="Size of the embeddings for the DETR transformer")

parser.add_argument('--vit_dim', default=384, type=int,
					help="Output token dimension of the VIT")
parser.add_argument('--vit_weights', type=str, default="https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth",
					help="Path to the weights for vit (must match the vit_arch, input_size and patch_size).")
parser.add_argument('--vit_dropout', default=0., type=float,
					help="Dropout applied in the vit backbone")
parser.add_argument('--dec_layers', default=6, type=int,
					help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=1536, type=int,
					 help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--dropout', default=0., type=float,
					help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
					help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=64, type=int,
					help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--data_path', default="/home/vp.shivasan/data/data/ChartOCR_lines/line/images", type=str)

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--scale_factor', default=0.3, type=float, help="Augmentation scaling parameter \
														(default from simple baselines is %(default)s)")

parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--show_keypoints', default=True, type=bool)

args = parser.parse_args()
num_gpus = torch.cuda.device_count()

model_arch = 'Branched_Conv_Relation_Net'
model = Branched_Conv_Relation_Net()

# cuda_id = 3
# CUDA_ = 'cuda:' + str(cuda_id)
CUDA_ = 'cuda:1'

# fix the seed for reproducibility
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

data_path = '/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/'
save_path = '/home/md.hassan/charts/ChartIE/results/'
for f in os.listdir(save_path):
    os.remove(save_path + f) 

state = torch.load('/home/md.hassan/charts/s_CornerNet/legend_mapping2/ckpt/' + model_arch +'/epoch146.t7', map_location = 'cpu')
model.load_state_dict(state['state_dict'])
model = model.to(CUDA_)
model.eval()

# Loading kp model
kp_model  = Model(args)
kp_state = torch.load('/home/vp.shivasan/ChartIE/checkpoint/checkpoint_l1_bigger_80.t7', map_location = 'cpu')
kp_model.load_state_dict(kp_state['state_dict'])
kp_model = kp_model.to(CUDA_)
kp_model.eval()
print("Loaded kp model at Epoch:", kp_state["epoch"])

# Loading element detection model
args_det = parser_det.parse_args()
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

# reader = easyocr.Reader(['en'], gpu = 'cuda:0') # only english for now
reader = easyocr.Reader(['en'], gpu = False) # only english for now

image_path = data_path + 'images'
anno_path = data_path + 'anno/'
with open(anno_path + 'cls_anno.json') as f:
  cls_annos = json.load(f)
with open(anno_path + 'line_anno.json') as f:
  line_annos = json.load(f)

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
    
def find_max(Arr):
	i, j = unravel_index(Arr.argmax(), Arr.shape)    
	max_value = Arr[i][j]        
	Arr[i] = -1000*np.ones((1, Arr.shape[1]))
	Arr[:, j] = -1000*np.ones((Arr.shape[0], ))
	return Arr, max_value, i, j

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order = scores.argsort()[::-1]
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
    det_boxes = torch.as_tensor(det_boxes)
    det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes)
    giou = box_ops.generalized_box_iou(torch.as_tensor(ocr_boxes), det_boxes)
    return giou

def get_matched_boxes(samples_, giou_matrix, ocr_boxes, det_boxes):
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

def get_ticks_text_coord(ticks_text, ticks_boxes):
	texts = []
	coords = []
	for text, coord in zip(ticks_text, ticks_boxes):
		if text.isnumeric():
			texts.append(float(text))
			coords.append(coord)
	return texts, coords

def get_ratio(text, coord):
	r1 = abs(text[1] - text[0])/abs(coord[1] - coord[0])
	r2 = abs(text[2] - text[0])/abs(coord[2] - coord[0])
	if r1/r2 > 0.9 and r1/r2 < 1.1:
		ratio = (r1 + r2)/2
		return ratio
	else:
		return None

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
	ocr_results = reader.readtext(image_, rotation_info = [270]) # assuming 270 deg rotation for vertical text
	# # plot OCR result
	# for result in ocr_results:
	# 	r = np.array(result[0]).astype(int)
	# 	cv2.rectangle(image_, r[0], r[2], (0, 255, 0), 1)  
	# 	cv2.putText(image_, result[1], r[0], font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)          

	# plot detr predictions (unique boxes)
	unique_boxes = {}
	for cls, box in zip(pred_classes[unique_idx], pred_boxes[unique_idx]):
		# plot_box(image_, cls, box)
		unique_boxes[int(cls)] = box
		box = box_ops.box_cxcywh_to_xyxy(box)
		box[1] = h - box[1]
		box[3] = h - box[3]
		if cls == 5:
			inner_plot_area = box

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
		nms_idx = nms(dets, 0.1)
		cls = pred_classes[box_idx][0]
		for box in cls_boxes[nms_idx]:    
			# plot_box(image_, cls, box)
			if cls == 6: # tick bbox
				tick_bboxes.append(np.array(box))
				# plot_box(image_, cls, box)
			elif cls == 7: # legend marker bbox
				legend_marker_bboxes.append(np.array(box))
				# plot_box(image_, cls, box)
			elif cls == 8: # legend text bbox
				legend_text_bboxes.append(np.array(box))
				# plot_box(image_, cls, box)
			elif cls == 9: # legend ele bbox
				legend_ele_bboxes.append(np.array(box))
				# plot_box(image_, cls, box)

	if len(ocr_results) > 0: 
		ocr_boxes = np.array([np.hstack((r[0][0], r[0][2])) for r in ocr_results])
		ocr_boxes = ocr_boxes[(ocr_boxes[:, 2] >= ocr_boxes[:, 0])*(ocr_boxes[:, 3] >= ocr_boxes[:, 1])]
		ocr_text = np.array([r[1] for r in ocr_results])
	else:
		ocr_boxes = []
		ocr_text = []

	# match OCR predicted legend text boxes and DETR predicted legend text boxes
	giou_matrix_legend_text = match_text_boxes(ocr_boxes.copy(), legend_text_bboxes.copy())
	# match OCR predicted tick text boxes and DETR predicted tick text boxes
	giou_matrix_tick = match_text_boxes(ocr_boxes.copy(), tick_bboxes.copy())

	# giou_matrix_legend_ele = match_text_boxes(ocr_boxes.copy(), legend_ele_bboxes.copy())

	# overlay matched boxes
	# ocr boxes in green. detr boxes in red
	legend_text_bboxes = np.array(legend_text_bboxes).astype(np.int32)
	matched_boxes_leg_text, _, _ = get_matched_boxes(image_, giou_matrix_legend_text, ocr_boxes, legend_text_bboxes)

	# legend_text_bboxes = np.array(legend_text_bboxes).astype(np.int32)
	# matched_boxes2 = get_matched_boxes(image_, giou_matrix_legend_ele, ocr_boxes, legend_text_bboxes)

	tick_bboxes = np.array(tick_bboxes)
	_, det_box_ids, ocr_box_ids = get_matched_boxes(image_, giou_matrix_tick, ocr_boxes, tick_bboxes)
		
	'''
	for each marker, find corresponding text. if no corresponding text, have some placeholder text
	2 ways:
	1. find among the already matched legend text boxes
	2. find among all text boxes
	'''
	# for marker in legend_marker_bboxes:
	giou_matrix_legend_marker = match_text_boxes(np.array(matched_boxes_leg_text, dtype=np.int32).copy(), 
								legend_marker_bboxes.copy())
	count = 0
	final_marker = []
	final_leg_text = []
	while(count < min(giou_matrix_legend_marker.shape)):
		giou_matrix_legend_marker, score, text_box_idx, marker_idx = find_max(np.array(giou_matrix_legend_marker))
		final_marker.append(legend_marker_bboxes[marker_idx])
		final_leg_text.append(ocr_text[np.where(np.all(ocr_boxes == matched_boxes_leg_text[text_box_idx], axis=1))])
		count += 1

	# for box, text in zip(final_marker, final_leg_text):
	# 	cv2.putText(image_, str(text)[2:-2], box[:2].astype(np.uint32), font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)    

	# for handling ticks ocr: need to resize
	tick_texts = [ocr_text[idx] for idx in ocr_box_ids]
	final_ticks = [tick_bboxes[idx] for idx in det_box_ids]
	for idx in det_box_ids:
			tick_bboxes = np.delete(tick_bboxes, idx, 0)
	for bbox in tick_bboxes:
		try:
			crop = image_[int(bbox[1]-10-bbox[3]//2):int(bbox[1]+10+bbox[3]//2), int(bbox[0]-10-bbox[2]//2):int(bbox[0]+10+bbox[2]//2)]
			crop = cv2.resize(crop, (60, int(60*bbox[3]/bbox[2])))
		except:
			continue
		r_ = reader.readtext(crop)
		if len(r_) == 0:
			gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
			gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
			r_ = reader.readtext(gray)
		if len(r_) == 0:
			gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
			gray = cv2.medianBlur(gray, 3)
			r_ = reader.readtext(gray)
		if len(r_) > 0:
			# cv2.putText(image_, r_[0][1], bbox[:2].astype(np.uint32), font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)    
			tick_texts.append(r_[0][1])
			final_ticks.append(bbox)

	final_ticks = np.array(final_ticks)
	xticks = final_ticks[:, 1] + final_ticks[:, 3] > np.array(inner_plot_area[1])
	yticks = final_ticks[:, 0] < np.array(inner_plot_area[0])

	xticks_boxes = final_ticks[xticks]
	xticks_text = np.array(tick_texts)[xticks][xticks_boxes[:, 0].argsort()]
	xticks_boxes = xticks_boxes[xticks_boxes[:, 0].argsort()]

	yticks_boxes = final_ticks[yticks]
	yticks_text = np.array(tick_texts)[yticks][yticks_boxes[:, 1].argsort()]
	yticks_boxes = yticks_boxes[yticks_boxes[:, 1].argsort()]

	## DEBUG: drawing detection boxes
	xticks_boxes_ = xticks_boxes.copy().astype(np.uint32)
	yticks_boxes_ = yticks_boxes.copy().astype(np.uint32)
	for i in range(len(xticks_boxes_)):
		plot_box(image_, 6, xticks_boxes_[i])
		# cv2.rectangle(image_, xticks_boxes_[i][0:2]-xticks_boxes_[i][2:]//2, xticks_boxes_[i][0:2]+xticks_boxes_[i][2:]//2,(0, 255, 0), 1)
		cv2.putText(image_, xticks_text[i], xticks_boxes_[i][0:2], font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)      
	for i in range(len(yticks_boxes)):
		plot_box(image_, 6, yticks_boxes_[i])
		# cv2.rectangle(image_, yticks_boxes_[i][0:2]-yticks_boxes_[i][2:]//2, yticks_boxes_[i][0:2]+yticks_boxes_[i][2:]//2,(0, 255, 0), 1)
		cv2.putText(image_, yticks_text[i], yticks_boxes_[i][0:2], font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)      
	for box, text in zip(final_marker, final_leg_text):
		# cv2.rectangle(image_, box[i][0:2]-box[i][2:]//2, box[i][0:2]+box[i][2:]//2,(0, 255, 0), 1)
		plot_box(image_, 7, box)
		cv2.putText(image_, str(text)[2:-2], box[:2].astype(np.uint32), font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)
	for i in unique_boxes.keys():
		plot_box(image_, i, unique_boxes[i])
	cv2.imwrite(save_path + 'det_' + image_name, image_)

	x_text, x_coords = get_ticks_text_coord(xticks_text, xticks_boxes)
	y_text, y_coords = get_ticks_text_coord(yticks_text, yticks_boxes)
	x_ratio = get_ratio(x_text, np.array(x_coords)[:, 0])
	y_ratio = get_ratio(y_text, np.array(y_coords)[:, 1])
	xticks_info = [x_text, x_coords, x_ratio]
	yticks_info = [y_text, y_coords, y_ratio]

	return final_marker, final_leg_text, xticks_info, yticks_info, unique_boxes

def keypoints(model,image_path,args,save_image = False):

	image= cv2.imread(image_path)

	if(save_image):
		save_name = "kp_"+image_path.split("/")[-1]
		image2 = image
	
	h,w,_ = image.shape
	image = cv2.resize(image, (args.input_size[0], args.input_size[1]))
	image = np.asarray(image)
	image = image.astype(np.float32) / 255
	
	image = torch.from_numpy(image)
	image = image.permute((2, 0, 1))
	image = torch.unsqueeze(image,0)
	print(image.shape)

	torch.tensor(image, dtype=torch.float32)
	image = image.to(CUDA_, non_blocking=True)
		
	output = model(image,return_attn =False)
	out_bbox = output["pred_boxes"][0]
	out_bbox = out_bbox.cpu().detach().numpy()
	x_cords = (out_bbox[:,0]*w)
	y_cords = (out_bbox[:,1]*h)
	# if(save_image):
	# 	for x,y in zip(x_cords,y_cords):
	# 		image2 = cv2.circle(image2,(int(x), int(y)),radius=3,color=(0,255,0),thickness=1)
	# 	cv2.imwrite(save_path+save_name,image2)
	return [tuple(x) for x in np.array((x_cords,y_cords)).T]

for image_name in tq.tqdm(os.listdir(image_path)[:100]):
	# for file in cls_annos["images"]:
	# 	if file["file_name"] == image_name:
	# 		id = file["id"]

	# legend_bboxes = []
	# for anno in cls_annos["annotations"]:
	# 	if anno["image_id"] == id and anno["category_id"] == 7:
	# 		legend_bboxes.append(anno["bbox"])

	legend_bboxes, legend_text, xticks_info, yticks_info, unique_boxes = run_element_det(det_model)
	x_text, x_coords, x_ratio = xticks_info
	y_text, y_coords, y_ratio = yticks_info

	all_kps = keypoints(model=kp_model,image_path=image_path +'/'+ image_name,args=args,save_image=True)

	image_cls = Image.open(image_path +'/'+ image_name)
	image = cv2.imread(image_path +'/'+ image_name)
	h, w, _ = image.shape
	scaled_kps = np.array(all_kps).copy()

	# data1 = ratio * (pixel1 - pixel0) + data0
	scaled_kps[:, 0] = (np.array(all_kps)[:, 0] - x_coords[0][0]) * x_ratio + x_text[0]
	scaled_kps[:, 1] = -1 * (np.array(all_kps)[:, 1] - y_coords[0][1]) * y_ratio + y_text[0]
	for i, kp in enumerate(all_kps):
		image = cv2.circle(image, (int(kp[0]), int(kp[1])), radius=3, color=(0,255,0), thickness=-1)
		cv2.putText(image, str(round(float(str(scaled_kps[i,0])),1))+', '+str(round(float(str(scaled_kps[i,1])),1)), (int(kp[0]), int(kp[1])), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
	cv2.imwrite(save_path + 'kp_' + image_name, image)

	# # ------ For Pred charts ------
	# Scores = np.zeros((len(legend_bboxes), len(pred_lines))) # Legends in Rows, Lines in Cols
	# draw = ImageDraw.Draw(image_cls)
	# fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

	# for bbox in legend_bboxes:
	#   draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
	#   crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]

	#   # cv2.imwrite('temp.png', crop)
	#   # image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), color=(0, 255, 0), thickness=1)

	#   crop = crop.astype(np.float32) / 255.
	#   crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
	#   crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

	#   legends_list.append(crop)

	# for legend_idx, legend in enumerate(legends_list):
	#     legend = torch.from_numpy(legend).cuda(CUDA_)
	#     legend = torch.unsqueeze(legend, 0)

	#     for line_idx, line_kp in enumerate(pred_lines):
	#         x_kps, y_kps = np.array(line_kp)[:, 0], np.array(line_kp)[:, 1]

	#         num_matches = 0
	#         # DEBUG:
	#         xy_list = []
	#         for x, y in zip(x_kps, y_kps):
	#             xy_list.append((x, y))
	#         draw.line(xy_list, fill=(0, 255, 0), width=2)
				
	#         for x, y in zip(x_kps, y_kps):
	#             bbox = [x - 20, y - 10, 40, 20]
	#             # DEBUG:
	#             # draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      

	#             try:
	#                 crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]
	#                 crop = crop.astype(np.float32) / 255.
	#                 crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
	#             except:
	#                 continue
	#                 # print("keypoint crop out of bound; skipping. image_name = ", + str(image_name))
	#             crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
	#             crop = torch.from_numpy(crop).cuda(CUDA_)
	#             crop = torch.unsqueeze(crop, 0)

	#             with torch.no_grad():
	#                 output = model(legend, crop)[0]
	#                 confidence = max(output)
	#                 output = np.argmax(output.cpu().detach().numpy())
	#                 num_matches += output

	#         Scores[legend_idx][line_idx] = num_matches

	# while(np.sum(Scores) > 0):
	#     Scores, score, legend_idx_, line_idx_ = find_max(Scores)
	#     legend_bbox = legend_bboxes[legend_idx_]

	#     line_coords = pred_lines[line_idx_]

	#     draw.text((line_coords[-1][0], line_coords[-1][1]), str(score), font = fnt, fill = (255, 0, 0))
	#     xy_list = [(line_coords[-1][0], line_coords[-1][1]), (legend_bbox[0], legend_bbox[1])]
	#     draw.line(xy_list, fill=(255, 0, 0), width=1)
	# image_cls.save(save_path + '_pred/' + image_name)

	# # ------ For GT charts ------
	# Scores = np.zeros((len(legend_bboxes), len(legend_bboxes))) # Legends in Rows, Lines in Cols

	# for bbox in legend_bboxes:
	#   draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
	# #   crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]

	# #   # cv2.imwrite('temp.png', crop)
	# #   # image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), color=(0, 255, 0), thickness=1)

	# #   crop = crop.astype(np.float32) / 255.
	# #   crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
	# #   crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

	# #   legends_list.append(crop)

	# image_cls = Image.open(image_path +'/'+ image_name)
	# draw = ImageDraw.Draw(image_cls)
	# fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
	# for legend_idx, legend in enumerate(legends_list):
	#     legend = torch.from_numpy(legend).cuda(CUDA_)
	#     legend = torch.unsqueeze(legend, 0)

	#     for line_idx, line_kp in enumerate(line_kps):
	#         x_kps, y_kps = line_kp

	#         num_matches = 0
	#         xy_list = []
	#         for x, y in zip(x_kps, y_kps):
	#             xy_list.append((x, y))
	#         draw.line(xy_list, fill=(0, 255, 0), width=2)
				
	#         for x, y in zip(x_kps, y_kps):
	#             bbox = [x - 20, y - 10, 40, 20]
	#             # DEBUG:
	#             # draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      

	#             crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]
	#             crop = crop.astype(np.float32) / 255.
	#             crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
	#             crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
	#             crop = torch.from_numpy(crop).cuda(CUDA_)
	#             crop = torch.unsqueeze(crop, 0)

	#             with torch.no_grad():
	#                 output = model(legend, crop)[0]
	#                 confidence = max(output)
	#                 output = np.argmax(output.cpu().detach().numpy())
	#                 num_matches += output

	#         Scores[legend_idx][line_idx] = num_matches

	# while(np.sum(Scores) > 0):
	#     Scores, score, legend_idx_, line_idx_ = find_max(Scores)
	#     legend_bbox = legend_bboxes[legend_idx_]

	#     line_coords = line_kps[line_idx_]

	#     draw.text((line_coords[0][-1], line_coords[1][-1]), str(score), font = fnt, fill = (255, 0, 0))
	#     xy_list = [(line_coords[0][-1], line_coords[1][-1]), (legend_bbox[0], legend_bbox[1])]
	#     draw.line(xy_list, fill=(255, 0, 0), width=1)
	# image_cls.save(save_path + '_GT/' + image_name)



# 	# ------ (Grouping and Legend mapping on GT keypoints) ------
# 	Scores = np.zeros((len(legend_bboxes), len(all_kps))) # Legends in Rows, Lines in Cols
# 	image_cls = Image.open(image_path +'/'+ image_name)
# 	draw = ImageDraw.Draw(image_cls)
# 	fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

# 	legends_list = []
# 	for bbox in legend_bboxes:
# 		draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
# 		crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]

# 		# cv2.imwrite('temp.png', crop)
# 		# image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), color=(0, 255, 0), thickness=1)

# 		crop = crop.astype(np.float32) / 255.
# 		crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
# 		crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

# 		legends_list.append(crop)

# 	kps_list = []
# 	kps_Scores = np.zeros((len(all_kps), len(all_kps)))
# 	for x,y in all_kps:
# 		bbox = [x - 20, y - 10, 40, 20]
# 		# draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
# 		crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]

# 		#   cv2.imwrite('temp.png', crop)
# 		# image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), color=(0, 255, 0), thickness=1)

# 		crop = crop.astype(np.float32) / 255.
# 		crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
# 		crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

# 		kps_list.append(crop)

# 	# ##  UNDER WORK ##
# 	# for kp1_idx, kpbox1 in enumerate(kps_list):
# 	# 	if True:
# 	# 		temp = kpbox1.transpose((2,1,0)) # 3,20,40 -> 40,20,3
# 	# 		temp = temp*255
# 	# 		temp = temp.astype(np.int32)
			
# 	# 		cv2.imwrite('temp1.png', temp)
# 	# 	kpbox1 = torch.from_numpy(kpbox1).to(CUDA_)
# 	# 	kpbox1 = torch.unsqueeze(kpbox1, 0)

# 	# 	for kp2_idx, kpbox2 in enumerate(kps_list):
# 	# 		if(kp1_idx == kp2_idx):
# 	# 			continue
# 	# 		if True:
# 	# 			temp2 = kpbox2.transpose((2,1,0))
# 	# 			temp2 = temp2*255
# 	# 			temp2 = temp2.astype(np.int32)
# 	# 			cv2.imwrite('temp2.png', temp2)
# 	# 		# DEBUG:
# 	# 		# draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')  
# 	# 		kpbox2 = torch.from_numpy(kpbox2).to(CUDA_)
# 	# 		kpbox2 = torch.unsqueeze(kpbox2, 0)

# 	# 		with torch.no_grad():
# 	# 			output = model(kpbox1, kpbox2)[0]
# 	# 			match_confidence = output[1]
# 	# 			print(match_confidence)
# 	# 		kps_Scores[kp1_idx][kp2_idx] = match_confidence
# 	# ## UNDER WORK ##
		
		

# 	for legend_idx, legend in enumerate(legends_list):
# 		legend = torch.from_numpy(legend).cuda(CUDA_)
# 		legend = torch.unsqueeze(legend, 0)

# 		for kp_idx, kp in enumerate(all_kps):
# 			x, y = kp                
# 			bbox = [x - 20, y - 10, 40, 20]
# 			# DEBUG:
# 			# draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
# 			try:
# 				crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]
# 				crop = crop.astype(np.float32) / 255.
# 				crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
# 			except:
# 				print("keypoint crop out of bound; skipping. image_name = ", + str(image_name))
# 				continue
# 			crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
# 			crop = torch.from_numpy(crop).cuda(CUDA_)
# 			crop = torch.unsqueeze(crop, 0)

# 			with torch.no_grad():
# 				output = model(legend, crop)[0]
# 				match_confidence = output[1]
# 			Scores[legend_idx][kp_idx] = match_confidence
 
# 	kp_mapping = Scores.argmax(axis=0)
# 	lines = {}
# 	colors = [(0, 122 , 122), (122, 0, 122), (0, 122 , 122), (255, 0 , 255), (0, 255, 255), (255, 255, 0),
# 			(122, 122, 0), (255, 0, 0), (0, 255, 0), (0, 0, 0)]
# 	for i in range(len(legend_bboxes)):
# 		kp_indices = np.where(kp_mapping == i)[0]
# 		line = np.array(all_kps)[kp_indices]
# 		sorted_line = sorted(line, key=lambda x: x[0])
# 		line = [tuple(l) for l in sorted_line]
# 		# color = colors[np.random.choice(np.arange(len(colors)))]
# 		# colors.pop(colors.index(color))
# 		color = np.random.choice(np.arange(len(colors)))
# 		draw.line(line, fill=colors[color], width=2)
# 		colors.pop(color)
# 		lines[i] = line

# 	for line_idx_, line in lines.items():
# 		if len(line) == 0 :
# 			continue
# 		legend_bbox = legend_bboxes[line_idx_]
# 		draw.text((line[-1][0], line[-1][1]), str(len(line)), font = fnt, fill = (255, 0, 0))
# 		xy_list = [(line[-1][0], line[-1][1]), (legend_bbox[0], legend_bbox[1])]
# 		draw.line(xy_list, fill=(255, 0, 0), width=1)
# 	image_cls.save(save_path + image_name)
# 	# image_cls.save(save_path + '_new_grouping/' + image_name)

# # Issues:
# # Problems at point of intersection. Allots both points to the same line. Also, transformer would
# # predict a single point in such intersections. Need to allot the same point to both lines.

# # Minor issue - Which line-legend-patch pair to choose, if both get same score? (say 1.0 and 1.0) 
# # Would taking first one be okay?
