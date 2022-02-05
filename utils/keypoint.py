import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _nms(heat, kernel=1):
  hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()
  return heat * keep


def _gather_feat(feat, ind, mask=None):
  dim = feat.size(2)
  ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
  feat = feat.gather(1, ind)
  if mask is not None:
    mask = mask.unsqueeze(2).expand_as(feat)
    feat = feat[mask]
    feat = feat.view(-1, dim)
  return feat


def _tranpose_and_gather_feat(feat, ind): #from cornernet
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat
    
def _tranpose_and_gather_feature(feature, ind):
  feature = feature.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] => [B, H, W, C]
  feature = feature.view(feature.size(0), -1, feature.size(3))  # [B, H, W, C] => [B, H x W, C]
  ind = ind[:, :, None].expand(ind.shape[0], ind.shape[1], feature.shape[-1])  # [B, num_obj] => [B, num_obj, C]
  feature = feature.gather(1, ind)  # [B, H x W, C] => [B, num_obj, C]
  return feature


def _topk(score_map, K=20):
  batch, cat, height, width = score_map.size()

  topk_scores, topk_inds = torch.topk(score_map.view(batch, -1), K)

  topk_classes = (topk_inds / (height * width)).int()
  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()
  return topk_scores, topk_inds, topk_classes, topk_ys, topk_xs


def _decode(hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br,
            K, kernel, ae_threshold, num_dets=1000):
  batch, cat, height, width = hmap_tl.shape

  hmap_tl = torch.sigmoid(hmap_tl)
  hmap_br = torch.sigmoid(hmap_br)

  # perform nms on heatmaps
  hmap_tl = _nms(hmap_tl, kernel=kernel)
  hmap_br = _nms(hmap_br, kernel=kernel)

  scores_tl, inds_tl, clses_tl, ys_tl, xs_tl = _topk(hmap_tl, K=K)
  scores_br, inds_br, clses_br, ys_br, xs_br = _topk(hmap_br, K=K)

  xs_tl = xs_tl.view(batch, K, 1).expand(batch, K, K)
  ys_tl = ys_tl.view(batch, K, 1).expand(batch, K, K)
  xs_br = xs_br.view(batch, 1, K).expand(batch, K, K)
  ys_br = ys_br.view(batch, 1, K).expand(batch, K, K)

  if regs_tl is not None and regs_br is not None:
    regs_tl = _tranpose_and_gather_feature(regs_tl, inds_tl)
    regs_br = _tranpose_and_gather_feature(regs_br, inds_br)
    regs_tl = regs_tl.view(batch, K, 1, 2)
    regs_br = regs_br.view(batch, 1, K, 2)

    xs_tl = xs_tl + regs_tl[..., 0]
    ys_tl = ys_tl + regs_tl[..., 1]
    xs_br = xs_br + regs_br[..., 0]
    ys_br = ys_br + regs_br[..., 1]

  # all possible boxes based on top k corners (ignoring class)
  bboxes = torch.stack((xs_tl, ys_tl, xs_br, ys_br), dim=3)

  embd_tl = _tranpose_and_gather_feature(embd_tl, inds_tl)
  embd_br = _tranpose_and_gather_feature(embd_br, inds_br)
  embd_tl = embd_tl.view(batch, K, 1)
  embd_br = embd_br.view(batch, 1, K)
  dists = torch.abs(embd_tl - embd_br)

  scores_tl = scores_tl.view(batch, K, 1).expand(batch, K, K)
  scores_br = scores_br.view(batch, 1, K).expand(batch, K, K)
  scores = (scores_tl + scores_br) / 2

  # reject boxes based on classes
  clses_tl = clses_tl.view(batch, K, 1).expand(batch, K, K)
  clses_br = clses_br.view(batch, 1, K).expand(batch, K, K)
  cls_inds = (clses_tl != clses_br)

  # reject boxes based on distances
  dist_inds = (dists > ae_threshold)

  # reject boxes based on widths and heights
  width_inds = (xs_br < xs_tl)
  height_inds = (ys_br < ys_tl)

  scores[cls_inds] = -1
  scores[dist_inds] = -1
  scores[width_inds] = -1
  scores[height_inds] = -1

  scores = scores.view(batch, -1)
  scores, inds = torch.topk(scores, num_dets)
  scores = scores.unsqueeze(2)

  bboxes = bboxes.view(batch, -1, 4)
  bboxes = _gather_feat(bboxes, inds)

  classes = clses_tl.contiguous().view(batch, -1, 1)
  classes = _gather_feat(classes, inds).float()

  scores_tl = scores_tl.contiguous().view(batch, -1, 1)
  scores_br = scores_br.contiguous().view(batch, -1, 1)
  scores_tl = _gather_feat(scores_tl, inds).float()
  scores_br = _gather_feat(scores_br, inds).float()

  detections = torch.cat([bboxes, scores, scores_tl, scores_br, classes], dim=2)
  return detections

def _decode_pure_cls(
        tl_heat, br_heat, tl_regr, br_regr, cls, offset,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    # print(tl_scores)
    tl_regr_ = _tranpose_and_gather_feature(tl_regr, tl_inds) # change  to  function?
    br_regr_ = _tranpose_and_gather_feature(br_regr, br_inds)


    tl_scores_ = tl_scores.view(1, batch, K)
    tl_clses_ = tl_clses.view(1, batch, K)
    tl_xs_ = tl_xs.view(1, batch, K)
    # print('_________________')
    # print(tl_xs_[0, 0])
    tl_ys_ = tl_ys.view(1, batch, K)
    tl_regr_ = tl_regr_.view(1, batch, K, 2)
    tl_xs_ += tl_regr_[:, :, :, 0]
    # print(tl_xs_[0, 0])
    tl_ys_ += tl_regr_[:, :, :, 1]
    br_scores_ = br_scores.view(1, batch, K)
    br_clses_ = br_clses.view(1, batch, K)
    br_xs_ = br_xs.view(1, batch, K)
    br_ys_ = br_ys.view(1, batch, K)
    br_regr_ = br_regr_.view(1, batch, K, 2)
    br_xs_ += br_regr_[:, :, :, 0]
    br_ys_ += br_regr_[:, :, :, 1]
    detections_tl = torch.cat([tl_scores_, tl_clses_.float(), tl_xs_, tl_ys_], dim=0)
    detections_br = torch.cat([br_scores_, br_clses_.float(), br_xs_, br_ys_], dim=0)
    print(cls)
    cls = torch.squeeze(torch.argmax(cls, 1))
    offset = torch.squeeze(offset)
    return detections_tl, detections_br, cls, offset

def _decode_pure_line(    # for decoding test o/p for Line keypoint network
        key_heat, hybrid_heat, key_tag, _, key_regr,
        K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    batch, cat, height, width = key_heat.size()

    key_heat = torch.sigmoid(key_heat)
    hybrid_heat = torch.sigmoid(hybrid_heat)

    # perform nms on heatmaps
    key_heat = _nms(key_heat, kernel=kernel)
    hybrid_heat = _nms(hybrid_heat, kernel=kernel)

    key_scores, key_inds, key_clses, key_ys, key_xs = _topk(key_heat, K=K)
    hybrid_scores, hybrid_inds, hybrid_clses, hybrid_ys, hybrid_xs = _topk(hybrid_heat, K=K)
    # print(key_scores)
    key_regr_ = _tranpose_and_gather_feat(key_regr, key_inds)
    hybrid_regr_ = _tranpose_and_gather_feat(key_regr, hybrid_inds)
    key_tag_ = _tranpose_and_gather_feat(key_tag, key_inds)
    hybrid_tag_ = _tranpose_and_gather_feat(key_tag, hybrid_inds)

    key_tag_ = key_tag_.view(1, batch, K)
    hybrid_tag_ = hybrid_tag_.view(1, batch, K)
    key_scores_ = key_scores.view(1, batch, K)
    key_clses_ = key_clses.view(1, batch, K)
    key_xs_ = key_xs.view(1, batch, K)
    # print('_________________')
    # print(key_xs_[0, 0])
    key_ys_ = key_ys.view(1, batch, K)
    key_regr_ = key_regr_.view(1, batch, K, 2)
    key_xs_ += key_regr_[:, :, :, 0]
    # print(key_xs_[0, 0])
    key_ys_ += key_regr_[:, :, :, 1]
    hybrid_scores_ = hybrid_scores.view(1, batch, K)
    hybrid_clses_ = hybrid_clses.view(1, batch, K)
    hybrid_xs_ = hybrid_xs.view(1, batch, K)
    hybrid_ys_ = hybrid_ys.view(1, batch, K)
    hybrid_regr_ = hybrid_regr_.view(1, batch, K, 2)
    hybrid_xs_ += hybrid_regr_[:, :, :, 0]
    hybrid_ys_ += hybrid_regr_[:, :, :, 1]
    detections_key = torch.cat([key_scores_, key_tag_, key_clses_.float(), key_xs_, key_ys_], dim=0)
    detections_hybrid = torch.cat([hybrid_scores_, hybrid_tag_, hybrid_clses_.float(), hybrid_xs_, hybrid_ys_], dim=0)

    return detections_key, detections_hybrid

def _rescale_points(dets, ratios, borders, sizes):
    xs, ys = dets[:, :, 2], dets[:, :, 3]
    xs    /= ratios[0, 1]
    ys    /= ratios[0, 0]
    xs    -= borders[0, 2]
    ys    -= borders[0, 0]
    np.clip(xs, 0, sizes[0, 1], out=xs)
    np.clip(ys, 0, sizes[0, 0], out=ys)
