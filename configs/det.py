import argparse
parser_det = argparse.ArgumentParser()

parser_det.add_argument('--lr', default=1e-4, type=float)
parser_det.add_argument('--lr_backbone', default=1e-5, type=float)
parser_det.add_argument('--batch_size', default=1, type=int)
parser_det.add_argument('--weight_decay', default=1e-4, type=float)
parser_det.add_argument('--epochs', default=300, type=int)
parser_det.add_argument('--lr_drop', default=200, type=int)
parser_det.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# Model parameters
parser_det.add_argument('--weights', type=str, default='/home/md.hassan/charts/detr/charts/ckpt/figqa_dataset/checkpoint110.pth')
# parser_det.add_argument('--weights', type=str, default='/home/md.hassan/charts/detr/charts/ckpt/final_dataset/checkpoint_latest.pth')

# * Backbone
parser_det.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser_det.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser_det.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

# * Transformer
parser_det.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser_det.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser_det.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser_det.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser_det.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser_det.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser_det.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser_det.add_argument('--pre_norm', action='store_true')

# * Segmentation
parser_det.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# Loss
parser_det.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser_det.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser_det.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser_det.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser_det.add_argument('--mask_loss_coef', default=1, type=float)
parser_det.add_argument('--dice_loss_coef', default=1, type=float)
parser_det.add_argument('--bbox_loss_coef', default=5, type=float)
parser_det.add_argument('--giou_loss_coef', default=2, type=float)
parser_det.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

# dataset parameters
parser_det.add_argument('--coco_path', type=str)
parser_det.add_argument('--dataset_file', default='charts')
parser_det.add_argument('--output_dir', default='/home/md.hassan/charts/detr/charts',
                    help='path where to save, empty for no saving')
parser_det.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser_det.add_argument('--num_workers', default=2, type=int)

parser_det.add_argument('--distributed', default=False, type=bool)
