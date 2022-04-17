import argparse
import random
import time
import os
from pathlib import Path
import tqdm as tq
import numpy as np
import torch
import cv2
import easyocr
from util.misc import *
from util import box_ops

import util.misc as utils
from models_det import build_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--weights', type=str, default=None)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--dataset_file', default='charts')
    parser.add_argument('--output_dir', default='/home/md.hassan/charts/detr/charts',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--distributed', default=False, type=bool)

    return parser

CLASSES = ['Legend', 'ValueAxisTitle', 'ChartTitle', 'CategoryAxisTitle', 'PlotArea', 
                'InnerPlotArea', 'XY_Ticks', 'LegendMarker', 'LegendText', 'LegendElement']
colors = [(0, 122 , 122), (122, 0, 122), (0, 122 , 122), (255, 0 , 255), (0, 255, 255), (255, 255, 0),
          (122, 122, 0), (255, 0, 0), (0, 255, 0), (0, 0 , 255)]
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
reader = easyocr.Reader(['en'], gpu = True) # only english for now

def main(args):
    device = args.device

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    args.distributed = False
    model, _, _ = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    checkpoint = torch.load(args.weights, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    print("Loaded model at Epoch {}".format(epoch))

    output_path = '/home/md.hassan/charts/ChartIE/results_det/'
    image_path = '/home/md.hassan/charts/s_CornerNet/synth_data/data/line/figqa/val/images/'

    for f in os.listdir(output_path):
        os.remove(output_path+'/'+f) 

    ctr = 0
    for image_name in tq.tqdm(os.listdir(image_path)[:50]):
        samples = cv2.imread(image_path + image_name)

        samples = samples.astype(np.float32)
        samples = normalize(samples)
        samples = torch.from_numpy(samples).to(device)
        samples = samples.permute(2, 0, 1)

        samples = nested_tensor_from_tensor_list([samples])
        with torch.no_grad():
            outputs = model(samples.to(device))

        pred_logits = outputs['pred_logits'][0][:, :len(CLASSES)]
        pred_boxes = outputs['pred_boxes'][0]

        max_output = pred_logits.softmax(-1).max(-1)
        topk = max_output.values.topk(50)

        pred_logits = pred_logits[topk.indices]
        pred_boxes = pred_boxes[topk.indices]
        pred_classes = pred_logits.argmax(axis=1)

        unique_idx = []
        for i in range(0, 6):
            if i in pred_classes:
                unique_idx.append(np.where(pred_classes.cpu() == i )[0][0])
        non_unique_classes = []
        for i in range(6, 10):
            if i in pred_classes:
                non_unique_classes.append(np.where(pred_classes.cpu() == i )[0])

        samples = samples.tensors[0].permute(1, 2, 0).cpu().numpy()
        h, w, _ = samples.shape
        samples_ = unnormalize(samples.copy())*255.0
        samples_ = samples_.astype(np.int32)
        pred_boxes = pred_boxes.cpu() * torch.Tensor([w, h, w, h])

        # plot detr predictions (unique boxes)
        for cls, box in zip(pred_classes[unique_idx], pred_boxes[unique_idx]):
            plot_box(samples_, cls, box, w, h)

        # plot detr predictions (non unique boxes)
        legend_text_bboxes = []
        tick_bboxes = []
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
                # plot_box(samples_, cls, box, w, h)
                if cls == 7: # if tick bbox, then save it
                    tick_bboxes.append(np.array(box))
                elif cls == 8: # if legend text bbox, then save it
                    legend_text_bboxes.append(np.array(box))

        # get boxes and text from OCR
        results = reader.readtext(samples_.astype(np.uint8), rotation_info = [270]) # assuming 270 deg rotation for vertical text
        # plot OCR result
        for result in results:
            r = np.array(result[0]).astype(int)
            cv2.rectangle(samples_, r[0][0], r[0][2], (0, 255, 0), 1)  
            cv2.putText(samples_, result[1], r[0][0], font, fontScale, (0, 0, 0), 1, cv2.LINE_AA)          

        # match OCR predicted legend text boxes and DETR predicted legend text boxes
        ocr_boxes = np.array([np.hstack((r[0][0], r[0][2])) for r in results])
        giou_matrix = match_text_boxes(ocr_boxes, legend_text_bboxes)

        # match OCR predicted tick text boxes and DETR predicted tick text boxes
        ocr_boxes = np.array([np.hstack((r[0][0], r[0][2])) for r in results])
        giou_matrix = match_text_boxes(ocr_boxes, tick_bboxes)

        # overlay matched boxes
        # ocr boxes in green. detr boxes in red
        legend_text_bboxes = np.array(legend_text_bboxes).astype(np.int32)
        plot_matched_boxes(samples_, giou_matrix, ocr_boxes, legend_text_bboxes)

        tick_bboxes = np.array(tick_bboxes).astype(np.int32)
        plot_matched_boxes(samples_, giou_matrix, ocr_boxes, tick_bboxes)


        '''
        - To get text box, GIoU must be > some thresh[0?] b/w:
            - ocr text and detr text boxes, or
            - ocr text and detr legend element boxes, or
            - ocr text and big legend element box?
            - or horizontally adjacent to some detected legend marker

        - Problems: 
            - what if ocr does not detect text box, but detr detects some legend element?
                   do we do legend mapping for that detected marker then?
            - if detr misses a legend marker, is it possible to get it from its detected text boxes?
                - can do if thats the only legend text left and others have been matched

        - Other todo:
            - after finalizing detected legend text and markers, relate each marker to its text
            - if a marker does not have associated detected text, have some placeholder text?
            - deal with xticks and yticks detection and unit scaling
        '''

        cv2.imwrite(output_path + str(ctr) + '.png', samples_)
        ctr += 1

def match_text_boxes(ocr_boxes, det_boxes):
    det_boxes = torch.as_tensor(det_boxes)
    det_boxes = box_ops.box_cxcywh_to_xyxy(det_boxes)
    giou = box_ops.generalized_box_iou(torch.as_tensor(ocr_boxes), det_boxes)
    return giou

def plot_matched_boxes(samples_, giou_matrix, ocr_boxes, det_boxes):
    count = 0
    while(count < min(giou_matrix.shape)):
        giou_matrix, score, ocr_idx, det_box_idx = find_max(np.array(giou_matrix))
        if score > 0:
            cv2.rectangle(samples_, ocr_boxes[ocr_idx][0:2], ocr_boxes[ocr_idx][2:],(0, 255, 0), 1)
            cv2.rectangle(samples_, det_boxes[det_box_idx][0:2], det_boxes[det_box_idx][2:], (0, 0, 255), 1)
        count += 1

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

from numpy import unravel_index
def find_max(Arr):
    i, j = unravel_index(Arr.argmax(), Arr.shape)    
    max_value = Arr[i][j]        
    Arr[i] = np.zeros((1, Arr.shape[1]))
    Arr[:, j] = np.zeros((Arr.shape[0], ))
    return Arr, max_value, i, j

def plot_box(image, cls, box, w, h):
    if cls >= len(CLASSES):
        return
    box = box.cpu()
    cv2.rectangle(image, (int(box[0] - box[2]//2), int(box[1] - box[3]//2)), (int(box[0] + box[2]//2), int(box[1]+box[3]//2)), colors[cls], 1)            
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR running script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
