import torch
import sys
sys.path.append("/home/vp.shivasan/ChartIE")
from models.my_model import Model
import os
import argparse
from datasets.coco_line import build as build_coco_line
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
LOCAL_RANK = 1
DATA_DIR = "/home/vp.shivasan/data/data/ChartOCR_lines"
LOG_NAME  = "ChartIE"
DIST = False
DEVICE = "cuda:1"
SPLIT_RATIO = 1.0
ROOT_DIR = "/home/vp.shivasan/ChartIE"
SAVE_PATH_BASE = "/home/vp.shivasan/ChartIE/training"
GPUS = 1
save_path = "/home/vp.shivasan/ChartIE/results/"

parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--batch_size', default=42, type=int)
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--vit_arch', default="dino_deit_small", type=str)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--lr_drop', default=50, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')
parser.add_argument('--debug', action='store_true', help="for faster debugging")

# Model parameters
parser.add_argument('--init_weights', type=str, default=None,
                    help="Path to the pretrained model.")

parser.add_argument('--position_embedding', default='enc_xcit',
                    type=str, choices=('enc_sine', 'enc_learned', 'enc_xcit',
                                        'learned_cls', 'learned_nocls', 'none'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--activation', default='gelu', type=str, choices=('relu', 'gelu', "glu"),
                    help="Activation function used for the transformer decoder")

parser.add_argument('--vit_as_backbone', action='store_true', help="Use VIT as the backbone of DETR, instead of the encoder part in vitdetr")
parser.add_argument('--input_size', nargs="+", default=[224, 224], type=int,
                    help="Input image size. Default is %(default)s")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings for the DETR transformer")
# PPP When VIT is used as a backbone this argument only affects the backbone.
# The DETR transformer still has the same hidden_dims 
# (controlled by the transformer.d_model value)
# When using vitdetr (no backbone) vit_dim must be equal to hidden_dim
parser.add_argument('--vit_dim', default=384, type=int,
                    help="Output token dimension of the VIT")
parser.add_argument('--vit_weights', type=str, default="https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth",
                    help="Path to the weights for vit (must match the vit_arch, input_size and patch_size).")
parser.add_argument('--vit_dropout', default=0., type=float,
                    help="Dropout applied in the vit backbone")

# * Transformer
parser.add_argument('--dec_arch', default="detr", type=str, choices=('xcit', 'detr'))
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
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
parser.add_argument('--with_lpi', action='store_true',
                    help="For the xcit decoder. Use lpi in decoder blocks")

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")

# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)

parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

# dataset parameters
parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', default="/home/vp.shivasan/data/data/ChartOCR_lines/line/images", type=str)

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--eval', action='store_true')
parser.add_argument('--use_det_bbox', action='store_true', help='For keypoints detecti8on, use person detected \
                    bboxes (from json file) for evaluation')
parser.add_argument('--scale_factor', default=0.3, type=float, help="Augmentation scaling parameter \
                                                        (default from simple baselines is %(default)s)")

parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=1000)
parser.add_argument('--val_interval', type=int, default=2500)
parser.add_argument('--local_rank', type=int, default=1)
parser.add_argument('--lr_decay', type=int, default=10)


args = parser.parse_args()
os.chdir(ROOT_DIR)


model  = Model(args)
cuda_id = 1
CUDA_ = 'cuda:' + str(cuda_id)
model = model.to(CUDA_)

state = torch.load('/home/vp.shivasan/ChartIE/training/line_latest_ckpt.t7', map_location = 'cpu')


model.load_state_dict(state['state_dict'])
model = model.to(CUDA_)
model.eval()
print("Loaded model at Epoch:",state["epoch"])

args.batch_size =1
torch.manual_seed(317)
torch.backends.cudnn.benchmark = True  
num_gpus = torch.cuda.device_count()
dataset_val = build_coco_line(image_set='val', args=args)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)

pin_memory = False
if GPUS is not None:
    pin_memory = True

val_loader = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, persistent_workers=True,
                                
                                num_workers=args.num_workers,
                                pin_memory=pin_memory)

def train():
        for batch_idx, batch in enumerate(val_loader):
            for k in batch:
                batch[k] = batch[k].to(device=DEVICE, non_blocking=True)
            res1 = get_attention_map(batch['image'])
            # plot_attention_map()
            print(res1)
            outputs = model(batch['image'])
            img_name_arr = batch['image_file'].cpu().detach().numpy()[0]
            path = ""
            for val in img_name_arr:
                path+=chr(val)
            # h,w= batch["size"][0]
            plotted_image = show_keypoints(outputs,path)
            cv2.imwrite(save_path+str(batch["image_id"].cpu().detach().numpy()[0][0])+".png",plotted_image)
            

def show_keypoints(outputs,image_path):
    image= cv2.imread(image_path)
    h,w,_ = image.shape
    print(h,w)
    out_bbox = outputs["pred_boxes"][0]
    out_bbox = out_bbox.cpu().detach().numpy()
    x_cords = (out_bbox[:,0]*w).astype(np.uint32)
    y_cords = (out_bbox[:,1]*h).astype(np.uint32)
    for x,y in zip(x_cords,y_cords):
        image = cv2.circle(image,(x,y),radius=3,color=(0,255,0),thickness=1)
    return image

def get_attention_map(img, get_mask=False):
    # x = transform(img)
    # x.size()

    att_mat = model(img)

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:        
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")
    
    return result

def plot_attention_map(original_img, att_map):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
    ax1.set_title('Original')
    ax2.set_title('Attention Map Last Layer')
    _ = ax1.imshow(original_img)
    _ = ax2.imshow(att_map)


train()
        