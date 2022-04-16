from re import L
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import os
from models import my_model
import torch.nn as nn
from datetime import datetime
from datasets.coco_line import build as build_coco_line
from torch.utils.data import DataLoader
import util.misc as utils
import time
from matcher import build_matcher
from losses import SetCriterion
import numpy as np
import math

torch.cuda.set_device(1)
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '29500'

# LOCAL_RANK = 1
DATA_DIR = "/home/vp.shivasan/data/data/ChartOCR_lines"
LOG_NAME  = "ChartIE"
DIST = True
DEVICE = "cuda:1"
SPLIT_RATIO = 1.0
ROOT_DIR = "/home/vp.shivasan/ChartIE"
SAVE_PATH_BASE = "/home/vp.shivasan/ChartIE/training"
GPUS = 4


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
parser.add_argument('--data_path', default="/home/vp.shivasan/data/data/new_train/images", type=str)

parser.add_argument('--seed', default=42, type=int)

parser.add_argument('--eval', action='store_true')
parser.add_argument('--use_det_bbox', action='store_true', help='For keypoints detecti8on, use person detected \
                    bboxes (from json file) for evaluation')
parser.add_argument('--scale_factor', default=0.3, type=float, help="Augmentation scaling parameter \
                                                        (default from simple baselines is %(default)s)")

parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--lr_step', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--log_interval', type=int, default=1000)
parser.add_argument('--val_interval', type=int, default=2500)
parser.add_argument('--local_rank', type=int, default=1)
parser.add_argument('--lr_decay', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.5)


parser.add_argument('--distributed', default=True, type=bool)
# parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--resume', default=False)


args = parser.parse_args()
os.chdir(ROOT_DIR)
def remove_items(test_list, item):
      
    # using list comprehension to perform the task
    res = [i for i in test_list if i != item]
  
    return res
def main(args):
    LOCAL_RANK = args.local_rank
    if(LOCAL_RANK == 1):
        writer = SummaryWriter(DATA_DIR + '/tensorboard/' + LOG_NAME)
    iter = 0
    epoch = None
    
    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True 
    # DIST = False 
    if DIST:
        DEVICE = torch.device('cuda:%d' % LOCAL_RANK)
        torch.cuda.set_device(LOCAL_RANK)
        dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.num_gpus, rank=LOCAL_RANK)
        print("DIST IS RUNNING")
    
    else:
        CUDA_ = 'cuda:1'
        DEVICE = torch.device(CUDA_)
    
    print('Setting up data...')
    dataset_val = build_coco_line(image_set='val', args=args)
    if not args.debug and not args.eval:
        dataset_train = build_coco_line(image_set='train', args=args)
    else:
        dataset_train = dataset_val
    # dataset_val = build_coco_line(image_set='val', args=args)
    if(DIST):
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val,shuffle = False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val,shuffle = False)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    pin_memory = False
    if GPUS is not None:
        pin_memory = True
    
    train_loader = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    persistent_workers=True,
                                   num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, persistent_workers=True,
                                 
                                 num_workers=args.num_workers,
                                 pin_memory=pin_memory)
    

    print('Creating model...')
    # args.device = "cuda"
    # DEVICE = "cuda"
    enc_dec_model = my_model.Model(args)
    enc_dec_model.to(DEVICE)
    
    bb = "encoder"
    param_dicts = [
            {"params": [p for n, p in enc_dec_model.named_parameters() if bb not in n and p.requires_grad]},
            {
                "params": [p for n, p in enc_dec_model.named_parameters() if bb in n and p.requires_grad],
                "lr": enc_dec_model.args.lr_backbone,
            },
        ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', verbose=True)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    if(args.resume):
        state = torch.load('/home/vp.shivasan/ChartIE/training/line_50_70000_ckpt.t7', map_location = 'cpu')
        enc_dec_model.load_state_dict(state['state_dict'])
        print("Loaded model at Epoch:",state["epoch"])
        epoch = state["epoch"]
        if('optimizer' in state):
            optimizer.load_state_dict(state['optimizer'])
            print(optimizer)
        if('scheduler' in state):
            scheduler.load_state_dict(state['scheduler'])
        if('scheduler2' in state):
            scheduler2.load_state_dict(state['scheduler2'])
    
    if DIST:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = model.to(cfg.device)
        enc_dec_model = nn.parallel.DistributedDataParallel(enc_dec_model,
                                                device_ids=[LOCAL_RANK])
                                                
        print("MODEL IN DIST")
    else:
    # comment this for single GPU
    # model = nn.DataParallel(model).to(cfg.device)
        enc_dec_model = nn.DataParallel(enc_dec_model,device_ids=[1,]).to('cuda:1')
    

    
    def train(epoch,iter,optimizer):
        if(LOCAL_RANK == 1):
            print('\n%s Epoch: %d' % (datetime.now(), epoch))
        enc_dec_model.train()
        tic = time.perf_counter()
        epoch_loss = 0.0
        epoch_l1_loss =0.0
        epoch_ang_loss = 0.0
        alpha = args.alpha
        # if(epoch <18):
        #         alpha = args.alpha
        # elif(epoch>23):
        #         alpha = 0.1
        # else:
        #         alpha = args.alpha*(math.exp(-0.322*(epoch-18)))

        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device=DEVICE, non_blocking=True)

            if LOCAL_RANK == 1 and epoch >= args.epochs:
                state = {
                'iter': iter,
                'epoch': epoch,
                'state_dict': enc_dec_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
                torch.save(state, SAVE_PATH_BASE + '/line_' + str(epoch) + "_" + str(iter) + "_ckpt.t7")
                torch.save(state, SAVE_PATH_BASE + '/line_latest_ckpt.t7')
                exit(0)

            outputs = enc_dec_model(batch['image'])  ### !!!!!!!!!!!!!!!!!!! Y CORDS SIMILAR BTW IMAGE KEYPOINTS AND ANCHRS 
            # l = batch["bboxes"].tolist()
            # targets = []
            # for elem in l:
            #     targets.append({'bboxes':elem})
            num_classes = 2
            args.set_cost_giou = 0.0
            matcher = build_matcher(args)
            weight_dict = {'loss_bbox': args.bbox_loss_coef}
            losses = ['keypoints']
            
            criterion= SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses,alp = alpha)
            targets = []
            i = 0
         
            for idx in range(len(batch["bboxes"])):
                elem = batch["bboxes"][idx]
                mask = batch["masks"][idx]
                mask_len = batch["mask_len"][idx]
                anchors = batch["anchors"][idx]
                targets.append({"bboxes":elem,"mask":mask,"labels":np.arange(1),"mask_len":mask_len,"anchors":anchors})
                i+=1
            loss_dict = criterion(outputs,targets)

            del outputs
            loss = loss_dict["loss_bbox"]
            l1_loss = loss_dict["l1_loss"]
            ang_loss = loss_dict["ang_loss"]
            # print(loss)

            if(LOCAL_RANK == 1):
                writer.add_scalar('loss', loss.item(), iter)
                writer.add_scalar('l1_loss', l1_loss.item(), iter)
                writer.add_scalar('ang_loss', ang_loss.item(), iter)

                
            epoch_loss+=loss
            epoch_l1_loss +=l1_loss
            epoch_ang_loss+=ang_loss
            loss = loss.unsqueeze(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if epoch > 0 and epoch % args.lr_step == 0:
            #     if optimizer.param_groups[0]["lr"] !=args.lr / (args.lr_decay) ** (epoch // args.lr_step):
            #         optimizer = decay_lr(optimizer, args.lr_decay, epoch, iter)
            #         print(optimizer.param_groups[0]["lr"])
            if LOCAL_RANK == 1 and iter % args.save_interval == 0:
                state = {
                'iter': iter,
                'epoch': epoch,
                'state_dict': enc_dec_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scheduler2': scheduler2.state_dict()

                }
                torch.save(state, SAVE_PATH_BASE+ '/line_' + str(epoch) + "_" + str(iter) + "_ckpt.t7")

            if iter % args.log_interval == 0:
                state = {
                'iter': iter,
                'epoch': epoch,
                'state_dict': enc_dec_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scheduler2': scheduler2.state_dict()
                }
                if LOCAL_RANK == 1: # log and save only one model from all GPUs
                    torch.save(state, SAVE_PATH_BASE + '/line_latest_ckpt.t7')
                    
                    # writer.add_scalar('loss', loss.item(), epoch)
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                # print('[%d/%d-%d/%d] iteration: %d  ' % (epoch, args.epochs, batch_idx, len(train_loader), iter) +
                # 'batch_loss=%.5f' %
                # (loss.item()) +
                # ' (%d samples/sec)' % (args.batch_size * args.log_interval / duration))
        
            if epoch > 0 and epoch % args.val_interval == 0:
                val(epoch, iter)
                enc_dec_model.train()
            iter += 1
        epoch_loss/=len(train_loader)
        epoch_l1_loss/=len(train_loader)
        epoch_ang_loss/=len(train_loader)
        scheduler.step(epoch_loss)
        
        if(LOCAL_RANK == 1):
            print("Epoch:%d"%(epoch)+"Epoch loss=%.5f"%(epoch_loss)+"Epoch l1 loss=%.5f"%(epoch_l1_loss)+"Epoch angular loss=%.5f"%(epoch_ang_loss))
            writer.add_scalar('Epoch loss', epoch_loss, epoch)
            writer.add_scalar('Epoch L1 loss', epoch_l1_loss, epoch)
            writer.add_scalar('Epoch Angular loss', epoch_ang_loss, epoch)
            writer.add_scalar('Alpha', alpha, epoch)

        return iter, optimizer
    def val(epoch, iter):
        print('\n%s Val@Epoch: %d, Iteration: %d' % (datetime.now(), epoch, iter))
        enc_dec_model.eval()
        # torch.cuda.empty_cache()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                for k in batch:
                    batch[k] = batch[k].to(device=DEVICE, non_blocking=True)

                outputs = enc_dec_model(batch['image'])
                num_classes = 2
                args.set_cost_giou = 0.0
                matcher = build_matcher(args)
                weight_dict = {'loss_bbox': args.bbox_loss_coef}
                losses = ['keypoints']
                criterion= SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                eos_coef=args.eos_coef, losses=losses)
                targets = []
                for elem in batch["bboxes"]:
                    targets.append({"bboxes":elem,"labels":np.arange(1)})
                loss_dict = criterion(outputs,targets )
                del outputs
                loss = loss_dict["loss_bbox"]
                val_loss+=loss
            
            val_loss/=(batch_idx+1)
            print('epoch [%d/%d] ' % (epoch, args.epochs) +
            ' val_loss_total = %.5f ' %
            (val_loss.item()))

            if LOCAL_RANK == 0: # log for only one GPU
                writer.add_scalar('val_loss', val_loss.item(), epoch)

    print('Starting training...')

    if epoch is None:
        epoch = 1
    for epoch in range(epoch, args.epochs + 1):
        if(DIST):
            sampler_train.set_epoch(epoch)
        iter, optimizer = train(epoch, iter, optimizer)

    


    

# def decay_lr(optimizer, lr_decay, epoch, iter):
#   print("At Epoch " + str(epoch) + ", Iteration " + str(iter) + ", dividing learning rate by: " + str(lr_decay))
#   for param_group in optimizer.param_groups:
#     param_group["lr"] /= lr_decay # CHANGED from param_group["lr"] = lr
#     # param_group["lr"] = lr/lr_decay
#   return optimizer


if __name__ == '__main__':
  main(args)



#   tensorboard --logdir=/home/vp.shivasan/data/data/ChartOCR_lines/tensorboard/ChartIE --host "10.0.62.205" --port 6009 --reload_multifile=true

# python -m torch.distributed.launch --nproc_per_node=4 --node_rank=0 main.py --vit_arch xcit_small_12_p16 --batch_size 42 --input_size 288 384 --hidden_dim 384 --vit_dim 384 --num_workers 24 --vit_weights https://dl.fbaipublicfiles.com/xcit/xcit_small_12_p16_384_dist.pth --alpha 0.5 
