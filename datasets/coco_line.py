"""
COCO Person dataset.
Persons (Cropped) with keypoints.

Code adapted from the simplebaselines repo:
https://github.com/microsoft/human-pose-estimation.pytorch/tree/master/lib/dataset

"""

import torch
import torchvision
from pathlib import Path
import copy
import cv2
import random

from util.sb_transforms import fliplr_joints, affine_transform, get_affine_transform

import datasets.transforms as T
from torchvision import transforms

from PIL import Image
from typing import Any, Tuple, List
import os
import numpy as np
from pycocotools.coco import COCO
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img, target):
        do_it = random.random() <= self.prob
        if not do_it:
            return img, target

        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max))), target


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return ImageOps.solarize(img), target
        else:
            return img, target


class ColorJitter(object):

    def __init__(self, jitter_p=0.8, gray_p=0.2):
        color_jitter = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(brightness=0.4,
                                                                                          contrast=0.4,
                                                                                          saturation=0.2,
                                                                                          hue=0.1)],
                                                                  p=jitter_p),
                                          transforms.RandomGrayscale(p=gray_p)])
        self.tr = color_jitter

    def __call__(self, img, target):
        return self.tr(img), target


def make_coco_person_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.NormalizePerson([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # TODO move resize/augment operations here
    # instead of the dataset
    if image_set == 'train':
        # tr = T.Compose([ColorJitter(0.8, 0.2),
                        # GaussianBlur(0.1),
                        # Solarization(0.2),
                        # normalize])
        return normalize  # tr

    if image_set == 'val':
        return normalize

    raise ValueError(f'unknown {image_set}')

# from ChartOCR
def _get_border(border, size):
  i = 1
  while size - border // i <= border // i:
    i *= 2
  return border // i

# from ChartOCR
def random_crop_line(image, detections, random_scales, view_size, border=64):
    view_height, view_width = (view_size['h'], view_size['w'])
    image_height, image_width = image.shape[0:2]

    scale  = np.random.choice(random_scales)
    height = int(view_height * scale)
    width  = int(view_width  * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    cropped_detections = detections.copy()
    cropped_detections[:, 0:-1:2] -= x0
    cropped_detections[:, 1:-1:2] -= y0
    cropped_detections[:, 0:-1:2] += cropped_ctx - left_w
    cropped_detections[:, 1:-1:2] += cropped_cty - top_h

    return cropped_image, cropped_detections, scale

def crop_image(image, center, new_size):
  cty, ctx = center
  height, width = new_size
  im_height, im_width = image.shape[0:2]
  cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

  x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
  y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

  left, right = ctx - x0, x1 - ctx
  top, bottom = cty - y0, y1 - cty

  cropped_cty, cropped_ctx = height // 2, width // 2
  y_slice = slice(cropped_cty - top, cropped_cty + bottom)
  x_slice = slice(cropped_ctx - left, cropped_ctx + right)
  cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

  border = np.array([
    cropped_cty - top,
    cropped_cty + bottom,
    cropped_ctx - left,
    cropped_ctx + right
  ], dtype=np.float32)

  offset = np.array([
    cty - height // 2,
    ctx - width // 2
  ])

  return cropped_image, border, offset

def color_jittering_(data_rng, image):
  functions = [brightness_, contrast_, saturation_]
  random.shuffle(functions)

  gs = grayscale(image)
  gs_mean = gs.mean()
  for f in functions:
    f(data_rng, image, gs, gs_mean, 0.4)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
  alpha = data_rng.normal(scale=alphastd, size=(3,))
  image += np.dot(eigvec, eigval * alpha)

def saturation_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
  alpha = 1. + data_rng.uniform(low=-var, high=var)
  blend_(alpha, image, gs_mean)

def blend_(alpha, image1, image2):
  image1 *= alpha
  image2 *= (1 - alpha)
  image1 += image2

def grayscale(image):
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


class CocoLine(torchvision.datasets.VisionDataset):
    '''
    "keypoints": {
        0: "line"
    },
    "skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, root, ann_file, image_set, transforms=None, is_train=False,
                 input_size=(224, 224), scale_factor=0.3):
        super().__init__(root)
        self.image_set = image_set
        self.is_train = is_train
        self.image_size = input_size
        self.split = image_set
        self.coco = COCO(ann_file)
        self.image_set_index = self.coco.getImgIds()

        self.db = self._get_db()

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        image_file = str(self.root) + '/' + self.image_set + '2019/' + str(db_rec['file_name'])

        image  = cv2.imread(image_file)
        height, width, _ = image.shape
        image = cv2.resize(image, (self.image_size[0], self.image_size[1]))
        image = np.asarray(image)

        bboxes = db_rec['objs']

        # print(db_rec)
        # bboxes = np.array([anno['bbox'] for anno in annotations])

        # for ind_bbox in range(len(bboxes)):   #if in the same chart, one line has more points... 
        # if len(bboxes[ind_bbox]) < max_len: #then pad others to match length.CAUSES PREDICTIONS @ORIGIN?
        #     bboxes[ind_bbox] = np.pad(bboxes[ind_bbox], (0, max_len - len(bboxes[ind_bbox])), 'constant',
        #                             constant_values=(0, 0))
        # bboxes = np.array(bboxes, dtype=float)

        # SKIPPING CROPS FOR NOW!!

        # random crop (for training) or center crop (for validation)
        # if self.split == 'train':
        #     image, bboxes, scale = random_crop_line(data_numpy, # function from ChartOCR, some extra functionality
        #                                 bboxes,
        #                                 random_scales=self.rand_scales,
        #                                 view_size=self.img_size,
        #                                 border=self.padding)
        # else:
        #     image, border, _ = crop_image(data_numpy,
        #                                         center=[data_numpy.shape[0] // 2, data_numpy.shape[1] // 2],
        #                                         new_size=[max(data_numpy.shape[0:2]), max(data_numpy.shape[0:2])])
        # bboxes[:, 0::2] += border[2]
        # bboxes[:, 1::2] += border[0]

        # # resize image and bbox
        # height, width = image.shape[:2]
        # image = cv2.resize(image, (self.img_size['w'], self.img_size['h']))

        # for i in range(len(bboxes)):   #if in the same chart, one line has more points... 
        #     if len(bboxes[i]) < 128: #then pad others to match length.CAUSES PREDICTIONS @ORIGIN?
        #         bboxes[i] = np.pad(bboxes[i], (0, 128 - len(bboxes[i])), 'constant',
        #                           constant_values=(0, 0))
        #     if(len(bboxes[i])>128):
        #         bboxes[i]=bboxes[i][:128] 
            
        

        bboxes = [bbox.tolist() for bbox in bboxes]

        m = 0
        for i in bboxes:
            m = max(m, len(i))                
        max_len = m
        for ind_bbox in range(len(bboxes)):   #if in the same chart, one line has more points... 
            if len(bboxes[ind_bbox]) < max_len: #then pad others to match length.CAUSES PREDICTIONS @ORIGIN?
                bboxes[ind_bbox] = np.pad(bboxes[ind_bbox], (0, max_len - len(bboxes[ind_bbox])), 'constant',
                                        constant_values=(0, 0))
        # print(np.array(bboxes).shape)
        # exit()
        bboxes = np.array(bboxes)
        # try:
        X_cords = bboxes[:, 0::2] *( 1 / width)
        # except:
            # print(image_file)
        Y_cords = bboxes[:, 1::2] *( 1 / height) # xy coords become 0 to 1
        X_cords = X_cords.flatten()
        Y_cords = Y_cords.flatten()
        bboxes = np.array((X_cords,Y_cords)).T

        if(bboxes.shape[0]<64):
            bboxes = np.vstack((bboxes,np.zeros((64-bboxes.shape[0],2))))
        if(bboxes.shape[0]>64):
            bboxes = bboxes[:64]
        
        # print(bboxes.shape)
        # exit()

        # # discard non-valid bboxes
        # bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.img_size['w'] - 1)
        # bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.img_size['h'] - 1)
        # keep_inds = np.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 0,
        #                         (bboxes[:, 3] - bboxes[:, 1]) > 0)
        # bboxes = bboxes[keep_inds]

        # RANDOM FLIPS REMOVED !!!
        image = image.astype(np.float32) / 255.

        # randomly change color and lighting
        if self.split == 'train':
            color_jittering_(np.random.RandomState(), image)
            # lighting_(np.random.RandomState(), image, 0.1, self.eig_val, self.eig_vec)

        # NO NORMALIZE AND TRANSPOSE FOR NOW

        # image -= self.mean
        # image /= self.std
        image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]


        
        target = {
            'image':torch.tensor(image, dtype=torch.float32),
            'size': torch.tensor(list(self.image_size)),
            'orig_size': torch.tensor([width, height]),
            'image_id': torch.tensor([db_rec['image_id']], dtype=torch.int64),
            'bboxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': np.arange(1)
        }

        # img = Image.fromarray(image)
        return target

    def __len__(self) -> int:
        return len(self.db)

    def _get_db(self):
        # use ground truth bbox
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db
    
    
    def _load_coco_person_detection_results(self):
        import json
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            print('=> Load %s fail!' % self.bbox_file)
            return None

        print('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            index = det_res['image_id']
            img_name = self.image_path_from_index(index)
            box = det_res['bbox']
            score = det_res['score']
            area = box[2] * box[3]

            if score < self.image_thre or area < 32**2:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image_id': index,
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,  # Try this score for evaluation (with COCOEval)
                'area': area,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        print('=> Total boxes after filter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']
        file_name = im_ann['file_name']

        annIds = self.coco.getAnnIds(imgIds=index)
        objs = self.coco.loadAnns(annIds)

        rec = []
        temp = []
        for obj in objs:
            temp.append(np.array(obj['bbox']))

        rec.append({
            'image_id': index,
            'image': self.image_path_from_index(index),
            'objs': temp,
            'file_name': file_name,
        })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        # PPP Tight bbox
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        root = Path(self.root)
        file_name = '%012d.jpg' % index
        image_path = root / f"{self.image_set}2017" / file_name

        return image_path


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    PATHS = {
        "train": ('/home/md.hassan/charts/ChartIE/PE-former/data/ChartOCR_lines/line/annotations_cleaned/clean_instancesLine(1023)_train2019.json'),
        "val": ('/home/md.hassan/charts/ChartIE/PE-former/data/ChartOCR_lines/line/annotations_cleaned/clean_instancesLine(1023)_val2019.json'),
    }

    ann_file = PATHS[image_set]

    dataset = CocoLine(root, ann_file, image_set,
                         is_train=(image_set == 'train'),
                         input_size=args.input_size, scale_factor=args.scale_factor)
    return dataset
