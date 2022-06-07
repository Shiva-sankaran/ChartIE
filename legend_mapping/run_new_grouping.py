import torch, cv2, os, sys, json
import numpy as np
import tqdm as tq
from PIL import Image, ImageDraw, ImageFont
from numpy import unravel_index
sys.path.append('/home/md.hassan/charts/ChartIE')
from legend_mapping.networks import MLP_Relation_Net, Conv_Relation_Net, Branched_MLP_Relation_Net, Branched_Conv_Relation_Net

num_gpus = torch.cuda.device_count()

model_arch = 'Branched_Conv_Relation_Net'

if model_arch == 'MLP_Relation_Net':
	model = MLP_Relation_Net()
	arch_type = 'MLP'
elif model_arch == 'Branched_MLP_Relation_Net':
	model = Branched_MLP_Relation_Net()
	arch_type = 'Branched_MLP'
elif model_arch == 'Conv_Relation_Net':
	model = Conv_Relation_Net()
	arch_type = 'Conv'
elif model_arch == 'Branched_Conv_Relation_Net':
	model = Branched_Conv_Relation_Net()
	arch_type = 'Branched_Conv'

cuda_id = 1
CUDA_ = 'cuda:' + str(cuda_id)
model = model.to(CUDA_)

model_path = '/home/md.hassan/charts/s_CornerNet/legend_mapping2/ckpt/' + model_arch +'/epoch291.t7'
state = torch.load(model_path, map_location = 'cpu')
model.load_state_dict(state['state_dict'])
model = model.to(CUDA_)
model.eval()

data_path = '/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/'
save_path = '/home/md.hassan/charts/ChartIE/legend_mapping/results/'

for f in os.listdir(save_path):
    os.remove(save_path + f) 

image_path = data_path + 'images'
anno_path = data_path + 'anno/'
with open(anno_path + 'cls_anno.json') as f:
	cls_annos = json.load(f)
with open(anno_path + 'line_anno.json') as f:
	line_annos = json.load(f)

def find_max(Arr):
	i, j = unravel_index(Arr.argmax(), Arr.shape)    
	max_value = Arr[i][j]        
	Arr[i] = np.zeros((1, Arr.shape[1]))
	Arr[:, j] = np.zeros((Arr.shape[0], ))
	return Arr, max_value, i, j

for image_name in tq.tqdm(os.listdir(image_path)[:100]):
	for file in cls_annos["images"]:
		if file["file_name"] == image_name:
			id = file["id"]

	legend_bboxes = []
	for anno in cls_annos["annotations"]:
		if anno["image_id"] == id and anno["category_id"] == 7:
			legend_bboxes.append(anno["bbox"])

	line_kps = []
	all_kps = []
	for anno in line_annos["annotations"]:
		if anno["image_id"] == id:
			temp = []
			for i in range(0, len(anno["bbox"]), 2):
				if anno["bbox"][i] != 0.0 and anno["bbox"][i+1] != 0.0:
					temp.append(anno["bbox"][i])
					temp.append(anno["bbox"][i+1])
					all_kps.append((anno["bbox"][i], anno["bbox"][i+1]))
			if len(temp) > 1:
				x_temp, y_temp = (np.array(temp[0::2]).astype(int), np.array(temp[1::2]).astype(int))
				line_kps.append((x_temp, y_temp))

	image_cls = Image.open(image_path +'/'+ image_name)
	image = cv2.imread(image_path +'/'+ image_name)
	legends_list = []

	# ------ (Grouping and Legend mapping on GT keypoints) ------
	Scores = np.zeros((len(legend_bboxes), len(all_kps))) # Legends in Rows, Lines in Cols
	image_cls = Image.open(image_path +'/'+ image_name)
	draw = ImageDraw.Draw(image_cls)
	fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)

	for bbox in legend_bboxes:
		draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
		crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]

		# cv2.imwrite('temp.png', crop)
		# image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), color=(0, 255, 0), thickness=1)

		crop = crop.astype(np.float32) / 255.
		crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
		crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

		legends_list.append(crop)

	for legend_idx, legend in enumerate(legends_list):
		legend = torch.from_numpy(legend).cuda(CUDA_)
		legend = torch.unsqueeze(legend, 0)

		for kp_idx, kp in enumerate(all_kps):
			x, y = kp                
			bbox = [x - 20, y - 10, 40, 20]
			# DEBUG:
			# draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
			try:
				crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]
				crop = crop.astype(np.float32) / 255.
				crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
			except:
				print("keypoint crop out of bound; skipping. image_name = ", + str(image_name))
				continue
			crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]
			crop = torch.from_numpy(crop).cuda(CUDA_)
			crop = torch.unsqueeze(crop, 0)

			with torch.no_grad():
				output = model(legend, crop)[0]
				match_confidence = output[1]
			Scores[legend_idx][kp_idx] = match_confidence
 
	kp_mapping = Scores.argmax(axis=0)
	lines = {}
	for i in range(len(legend_bboxes)):
		kp_indices = np.where(kp_mapping == i)[0]
		line = np.array(all_kps)[kp_indices]
		sorted_line = sorted(line, key=lambda x: x[0])
		line = [tuple(l) for l in sorted_line]
		draw.line(line, fill=(0, 255, 0), width=2)
		lines[i] = line

	for line_idx_, line in lines.items():
		if len(line) == 0 :
			continue
		legend_bbox = legend_bboxes[line_idx_]
		draw.text((line[-1][0], line[-1][1]), str(len(line)), font = fnt, fill = (255, 0, 0))
		xy_list = [(line[-1][0], line[-1][1]), (legend_bbox[0], legend_bbox[1])]
		draw.line(xy_list, fill=(255, 0, 0), width=1)
	image_cls.save(save_path + image_name)

# Issues:
# Problems at point of intersection. Allots both points to the same line. Also, transformer would
# predict a single point in such intersections. Need to allot the same point to both lines.

# Minor issue - Which line-legend-patch pair to choose, if both get same score? (say 1.0 and 1.0) 
# Would taking first one be okay?
