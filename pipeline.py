import torch, cv2, os, sys, json
import numpy as np
import tqdm as tq
from PIL import Image, ImageDraw, ImageFont
from numpy import unravel_index
sys.path.append('/home/vp.shivasan/ChartIE')
from models.my_model import Model
from ChartIE.models.networks import MLP_Relation_Net, Conv_Relation_Net, Branched_MLP_Relation_Net, Branched_Conv_Relation_Net
import argparse

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

parser.add_argument('--batch_size', default=42, type=int)
parser.add_argument('--patch_size', default=16, type=int)


parser.add_argument('--position_embedding', default='enc_xcit',
                    type=str, choices=('enc_sine', 'enc_learned', 'enc_xcit',
                                        'learned_cls', 'learned_nocls', 'none'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--activation', default='gelu', type=str, choices=('relu', 'gelu', "glu"),
                    help="Activation function used for the transformer decoder")

parser.add_argument('--input_size', nargs="+", default=[224, 224], type=int,
                    help="Input image size. Default is %(default)s")
parser.add_argument('--hidden_dim', default=256, type=int,
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

# if model_arch == 'MLP_Relation_Net':
#     model = MLP_Relation_Net()
#     arch_type = 'MLP'
# elif model_arch == 'Branched_MLP_Relation_Net':
#     model = Branched_MLP_Relation_Net()
#     arch_type = 'Branched_MLP'
# elif model_arch == 'Conv_Relation_Net':
#     model = Conv_Relation_Net()
#     arch_type = 'Conv'
# elif model_arch == 'Branched_Conv_Relation_Net':
#     model = Branched_Conv_Relation_Net()
#     arch_type = 'Branched_Conv'

cuda_id = 1
CUDA_ = 'cuda:' + str(cuda_id)
CUDA_ = 'cpu'


model = model.to(CUDA_)
state = torch.load('/home/vp.shivasan/ChartIE/legend_mapping2/ckpt/'+ model_arch +'/epoch146.t7', map_location = 'cpu')
model.load_state_dict(state['state_dict'])
model = model.to(CUDA_)
model.eval()

kp_model  = Model(args)
kp_state = torch.load('/home/vp.shivasan/ChartIE/checkpoint/checkpoint_l1_bigger_80.t7', map_location = 'cpu')
kp_model.load_state_dict(kp_state['state_dict'])
kp_model = kp_model.to(CUDA_)
kp_model.eval()

print("Loaded model at Epoch:",kp_state["epoch"])




# data_path = 'legend_mapping2/data/1k_val/'
data_path = '/home/vp.shivasan/ChartIE/legend_mapping2/data/1k_val/'
save_path = '/home/vp.shivasan/ChartIE/legend_mapping2/results/val_entangled'
# data_path = 'synth_data/data/line/st_line_complex/val_max_5/'
# save_path = 'legend_mapping2/results/val_max_5'

image_path = data_path + 'images'
anno_path = data_path + 'anno/'
with open(anno_path + 'cls_anno.json') as f:
  cls_annos = json.load(f)
with open(anno_path + 'line_anno.json') as f:
  line_annos = json.load(f)
with open(data_path + 'predicted.json') as f:
    pred_charts = json.load(f)

def find_max(Arr):
    i, j = unravel_index(Arr.argmax(), Arr.shape)    
    max_value = Arr[i][j]        
    # Arr = np.delete(Arr, i, 0)
    # Arr = np.delete(Arr, j, 1)
    Arr[i] = np.zeros((1, Arr.shape[1]))
    Arr[:, j] = np.zeros((Arr.shape[0], ))
    return Arr, max_value, i, j

def keypoints(model,image_path,args,save_image = False):

    image= cv2.imread(image_path)

    if(save_image):
        save_name = "detected_"+image_path.split("/")[-1]
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
    x_cords = (out_bbox[:,0]*w).astype(np.uint32)
    y_cords = (out_bbox[:,1]*h).astype(np.uint32)
    if(save_image):
        for x,y in zip(x_cords,y_cords):
            image2 = cv2.circle(image2,(x,y),radius=3,color=(0,255,0),thickness=1)
        cv2.imwrite(save_path+save_name,image2)
    return [tuple(x) for x in np.array((x_cords,y_cords)).T]

for image_name in tq.tqdm(os.listdir(image_path)[:100]):
    # if image_name != '1114.png':
    #   continue
    for file in cls_annos["images"]:
      if file["file_name"] == image_name:
        id = file["id"]

    legend_bboxes = []
    for anno in cls_annos["annotations"]:
        if anno["image_id"] == id and anno["category_id"] == 6:
            legend_bboxes.append(anno["bbox"])

    pred_lines = pred_charts[image_name]

    # line_kps = []
    # all_kps = []
    # for anno in line_annos["annotations"]:
    #     if anno["image_id"] == id:
    #         temp = []
    #         for i in range(0, len(anno["bbox"]), 2):
    #             if anno["bbox"][i] != 0.0 and anno["bbox"][i+1] != 0.0:
    #                 temp.append(anno["bbox"][i])
    #                 temp.append(anno["bbox"][i+1])
    #                 all_kps.append((anno["bbox"][i], anno["bbox"][i+1]))
    #         if len(temp) > 1:
    #             x_temp, y_temp = (np.array(temp[0::2]).astype(int), np.array(temp[1::2]).astype(int))
    #             line_kps.append((x_temp, y_temp))

    all_kps = keypoints(model=kp_model,image_path=image_path +'/'+ image_name,args=args,save_image=False)

    image_cls = Image.open(image_path +'/'+ image_name)
    image = cv2.imread(image_path +'/'+ image_name)
    legends_list = []

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
    
    kps_list = []
    kps_Scores = np.zeros((len(all_kps), len(all_kps)))
    for x,y in all_kps:
      bbox = [x - 20, y - 10, 40, 20]
      draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')      
      crop = image[int(bbox[1]):int(bbox[3]+bbox[1]), int(bbox[0]):int(bbox[2]+bbox[0])]

    #   cv2.imwrite('temp.png', crop)
      # image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), color=(0, 255, 0), thickness=1)

      crop = crop.astype(np.float32) / 255.
      crop = cv2.resize(crop, (40, 20)) # resizing to average size(37, 18) of legend markers
      crop = crop.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

      kps_list.append(crop)

    ##  UNDER WORK ##
    for kp1_idx, kpbox1 in enumerate(kps_list):
        if True:
            temp = kpbox1.transpose((2,1,0)) # 3,20,40 -> 40,20,3
            temp = temp*255
            temp = temp.astype(np.int32)
            
            cv2.imwrite('temp1.png', temp)
        kpbox1 = torch.from_numpy(kpbox1).to(CUDA_)
        kpbox1 = torch.unsqueeze(kpbox1, 0)

        

        for kp2_idx, kpbox2 in enumerate(kps_list):
            if(kp1_idx == kp2_idx):
                continue
            if True:
                temp2 = kpbox2.transpose((2,1,0))
                temp2 = temp2*255
                temp2 = temp2.astype(np.int32)
                cv2.imwrite('temp2.png', temp2)
            # DEBUG:
            # draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])], outline='green')  
            kpbox2 = torch.from_numpy(kpbox2).to(CUDA_)
            kpbox2 = torch.unsqueeze(kpbox2, 0)

            with torch.no_grad():
                output = model(kpbox1, kpbox2)[0]
                match_confidence = output[1]
                print(match_confidence)
            kps_Scores[kp1_idx][kp2_idx] = match_confidence
    ## UNDER WORK ##
        
        

    


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
    image_cls.save(save_path + '_new_grouping/' + image_name)

# Issues:
# Problems at point of intersection. Allots both points to the same line. Also, transformer would
# predict a single point in such intersections. Need to allot the same point to both lines.

# Minor issue - Which line-legend-patch pair to choose, if both get same score? (say 1.0 and 1.0) 
# Would taking first one be okay?

