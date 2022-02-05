from typing import Sequence
from numpy.core.fromnumeric import squeeze
import torch
import torch.nn as nn
import numpy as np

def _ae_loss(tag0, tag1, mask): # ORIGINAL embedding push/pull loss
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _ae_line_loss(tag_full, mask_full):  # embedding push/pull loss when multiple lines?
    # mask_full [batch, Max_group, Max_len]
    # tag_full  [batch, Max_group, Max_len]
    pull = 0
    push = 0
    tag_full = torch.squeeze(tag_full)
    tag_full[1-mask_full] = 0
    # tag_full[~mask_full] = 0
    num = mask_full.sum(dim=2, keepdim=True).float()
    tag_avg = tag_full.sum(dim=2, keepdim=True) / num
    pull = torch.pow(tag_full - tag_avg, 2) / (num + 1e-4)
    pull = pull[mask_full].sum()

    tag_avg = torch.squeeze(tag_avg)
    mask = mask_full.sum(dim=2)
    mask = mask.gt(1)
    num = mask.sum(dim=1, keepdim=True).float()
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)

    dist = tag_avg.unsqueeze(1) - tag_avg.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _regr_loss(regr, gt_regr, mask): # ORIGINAL offset loss?
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _neg_loss(preds, gt, lamda, lamdb):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    neg_weights = torch.pow(1 - gt[neg_inds], lamda)
    loss = 0
    for pred in preds:
        # print(pred.shape)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        # print(pos_pred)
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, lamdb)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, lamdb) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _offset_loss(offset, gt_offset):
    offset_loss = nn.functional.smooth_l1_loss(offset, gt_offset, size_average=True)
    return offset_loss

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x

def AELossPureCls(outs, batch):
    regr_weight = 1
    lamda = 4
    lamdb = 2
    _cls_loss = nn.CrossEntropyLoss(size_average=True)
    
    stride = 4
    if len(outs) == 10:
        tl_heats = outs[0:-2:stride]
        br_heats = outs[1:-2:stride]
        tl_regrs = outs[2:-2:stride]
        br_regrs = outs[3:-2:stride]
        cls = outs[-2]
        offset = outs[-1]
    elif len(outs) == 6:
        tl_heats = [outs[0]]
        br_heats = [outs[1]]
        tl_regrs = [outs[2]]
        br_regrs = [outs[3]]
        cls = outs[-2]
        offset = outs[-1]

    gt_tl_heat = batch['hmap_tl']
    gt_br_heat = batch['hmap_br']
    gt_mask    = batch['ind_masks']
    gt_tl_regr = batch['regs_tl']
    gt_br_regr = batch['regs_br']
    gt_cls     = batch['cls']
    gt_offset  = batch['offset']

    # focal loss
    focal_loss = 0

    tl_heats = [_sigmoid(t) for t in tl_heats]
    br_heats = [_sigmoid(b) for b in br_heats]

    focal_loss += _neg_loss(tl_heats, gt_tl_heat, lamda, lamdb)
    focal_loss += _neg_loss(br_heats, gt_br_heat, lamda, lamdb)

    regr_loss = 0
    for tl_regr, br_regr in zip(tl_regrs, br_regrs):
        regr_loss += _regr_loss(tl_regr, gt_tl_regr, gt_mask)
        regr_loss += _regr_loss(br_regr, gt_br_regr, gt_mask)
    regr_loss = regr_weight * regr_loss

    cls_loss = _cls_loss(cls, gt_cls)
    cls_loss = regr_weight * cls_loss

    gt_offset = gt_offset.unsqueeze(1)
    gt_offset = torch.tensor(gt_offset, dtype=torch.float, device='cuda') #added for type compatibility issue
    offset_loss = _offset_loss(offset, gt_offset) 
    offset_loss = regr_weight * offset_loss

    # loss = (focal_loss + regr_loss) / len(tl_heats) + cls_loss + offset_loss
    return focal_loss/len(tl_heats), regr_loss/len(tl_heats), cls_loss, offset_loss 
    # return loss.unsqueeze(0)

def AELossLine(outs, batch):
    pull_weight = 1e-1
    push_weight = 1e-1
    regr_weight = 1
    lamda = 4
    lamdb = 2
    
    stride = 5
    if len(outs) == 10:
        key_heats = outs[0::stride]
        hybrid_heats = outs[1::stride]
        key_tags  = outs[2::stride]
        key_tags_grouped  = outs[3::stride]
        key_regrs = outs[4::stride]
    elif len(outs) == 5:
        key_heats = [outs[0]]
        hybrid_heats = [outs[1]]
        key_tags = [outs[2]]
        key_tags_grouped = [outs[3]]
        key_regrs = [outs[4]]

    gt_key_heat = batch['key_heatmaps']
    gt_hybrid_heat = batch['hybrid_heatmaps']
    gt_mask = batch['tag_masks']
    gt_mask_grouped = batch['tag_masks_grouped']
    gt_key_regr = batch['key_regrs']

    # gt_mask = gt_mask.type(torch.bool) # changed for pytorch 1.9
    # gt_mask_grouped = gt_mask_grouped.type(torch.bool) # changed for pytorch 1.9

    # focal loss
    focal_loss = 0

    key_heats = [_sigmoid(t) for t in key_heats]
    hybrid_heats = [_sigmoid(b) for b in hybrid_heats]

    focal_loss += _neg_loss(key_heats, gt_key_heat, lamda, lamdb)
    focal_loss += _neg_loss(hybrid_heats, gt_hybrid_heat, lamda, lamdb)

    # tag loss
    pull_loss = 0
    push_loss = 0

    for key_tag_grouped in key_tags_grouped:
        pull, push = _ae_line_loss(key_tag_grouped, gt_mask_grouped)
        pull_loss += pull
        push_loss += push
    pull_loss = pull_weight * pull_loss
    push_loss = push_weight * push_loss

    regr_loss = 0
    for key_regr in key_regrs:
        regr_loss += _regr_loss(key_regr, gt_key_regr, gt_mask)
    regr_loss = regr_weight * regr_loss

    # loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(key_heats)
    return focal_loss/len(key_heats), pull_loss/len(key_heats), push_loss/len(key_heats), regr_loss/len(key_heats)
    # return loss.unsqueeze(0)

