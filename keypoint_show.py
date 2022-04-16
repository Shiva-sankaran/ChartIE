import torch
import sys
sys.path.append("/home/vp.shivasan/ChartIE")
from models.my_model import Model
import os
import argparse
from datasets.coco_line import build as build_line
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
LOCAL_RANK = 1
DATA_DIR = "/home/vp.shivasan/data/data/ChartOCR_lines"
LOG_NAME  = "ChartIE"
DIST = False
DEVICE = "cpu"
SPLIT_RATIO = 1.0
ROOT_DIR = "/home/vp.shivasan/ChartIE"
SAVE_PATH_BASE = "/home/vp.shivasan/ChartIE/training"
GPUS = 1
save_path = "/home/vp.shivasan/ChartIE/results/"

OUT_DIR = "/home/vp.shivasan/ChartIE/attention_map"
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
# parser.add_argument('--show_keypoints', default=True, type=bool)



args = parser.parse_args()
os.chdir(ROOT_DIR)
model  = Model(args)
cuda_id = 1
CUDA_ = 'cuda:' + str(cuda_id)
CUDA_ = "cpu"
model = model.to(CUDA_)
state = torch.load('/home/vp.shivasan/ChartIE/checkpoint/checkpoint_l1_bigger_80.t7', map_location = 'cpu')
model.load_state_dict(state['state_dict'])
model = model.to(CUDA_)
model.eval()
print("Loaded model at Epoch:",state["epoch"])
torch.manual_seed(317)
torch.backends.cudnn.benchmark = True  
num_gpus = torch.cuda.device_count()

def keypoints(image_path,save_image = False):
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
    image = image.to(DEVICE, non_blocking=True)
    
    
    output = model(image,return_attn = True)
    out_bbox = output["pred_boxes"][0]
    out_bbox = out_bbox.cpu().detach().numpy()
    x_cords = (out_bbox[:,0]*w).astype(np.uint32)
    y_cords = (out_bbox[:,1]*h).astype(np.uint32)
    if(save_image):
        for x,y in zip(x_cords,y_cords):
            image2 = cv2.circle(image2,(x,y),radius=3,color=(0,255,0),thickness=1)
        cv2.imwrite(save_path+save_name,image2)
    return [tuple(x) for x in np.array((x_cords,y_cords)).T]
    

IMAGEPATH = "/home/vp.shivasan/data/data/ChartOCR_lines/line/images/test2019/f4a2a26ef3d6b6da14e661c013590327_d3d3LnNpZS5nb2IuZG8JMzUuMTg1LjgzLjIwNw==-0-0.png"
IMAGE_DIR =  "/home/vp.shivasan/data/data/ChartOCR_lines/line/images/val2019/"
i = 1
for file_name in os.listdir(IMAGE_DIR)[:20]:
    print(i)
    file_path = IMAGE_DIR+file_name
    temp = keypoints(file_path,save_image=True)
    i+=1
# print(keypoints(IMAGEPATH,save_image=True))
