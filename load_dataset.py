from audioop import tostereo
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import pycocotool.coco as coco
from utils.image import random_crop, crop_image, random_crop_cls, random_crop_line
from utils.image import color_jittering_, lighting_
from utils.image import draw_gaussian, gaussian_radius



class Line(Dataset):
  def __init__(self, data_dir, split, split_ratio=1.0, gaussian=True, img_size=511):
    super(Line, self).__init__()
    self.split = split
    self.gaussian = gaussian

    self.img_size = {'h': img_size, 'w': img_size}
    self.padding = 128

    self.data_rng = np.random.RandomState(123) #for jittering and lighting
    self.rand_scales = np.arange(0.6, 1.4, 0.1)
    self.gaussian_iou = 0.3

    # self.data_dir = os.path.join(data_dir, 'linedata/line/')
    self.data_dir = os.path.join(data_dir, 'line/')

    # self.img_dir = self.data_dir + 'images/' + split + '2019'
    self.img_dir = self.data_dir + '/images/'+split+"2019/"
    self.anno_dir = self.data_dir + "annotations_cleaned/"

    if split == 'test':
      # self.annot_path = os.path.join(self.data_dir, 'annotations', 'instancesLine(1023)_test2019.json')
      self.annot_path = os.path.join(self.anno_dir, 'clean_instancesLine(1023)_test2019.json')
    else:
      # self.annot_path = os.path.join(self.data_dir, 'annotations', 'instancesLine(1023)_%s2019.json' % split)
      self.annot_path = os.path.join(self.anno_dir, 'clean_instancesLine(1023)_%s2019.json'%split)




    print('==> initializing Line %s data.' % split)
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()

    if 0 < split_ratio < 1:
      split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
      self.images = self.images[:split_size]

    self.num_samples = len(self.images)

    print('Loaded %d %s samples' % (self.num_samples, split))

  def __getitem__(self, index):
    img_id = self.images[index]
    image = cv2.imread(os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name']))
    annotations = self.coco.loadAnns(ids=self.coco.getAnnIds(imgIds=[img_id])) #matches _extract_data fn from ChartOCR

    bboxes= []
    max_len = 0
    ori_size = image.shape
    scale = 1
    for annotation in annotations:
      bbox = np.array(annotation["bbox"])
      bboxes.append(bbox)
      max_len = max(max_len, len(bbox))

    for ind_bbox in range(len(bboxes)):   #if in the same chart, one line has more points... 
      if len(bboxes[ind_bbox]) < max_len: #then pad others to match length.CAUSES PREDICTIONS @ORIGIN?
        bboxes[ind_bbox] = np.pad(bboxes[ind_bbox], (0, max_len - len(bboxes[ind_bbox])), 'constant',
                                  constant_values=(0, 0))
    bboxes = np.array(bboxes, dtype=float)

    # bboxes = np.array([anno['bbox'] for anno in annotations])
    if len(bboxes) == 0:
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)

    # bboxes = bboxes[0: self.max_group_len]  # EXTRA!!, set detections max length !!!!!removed for now
    # 'categories' variable from ChartOCR, is 0 for line annotations, so skipping that here
    # 'labels' here is the same

    # random crop (for training) or center crop (for validation)
    if self.split == 'train':
      image, bboxes, scale = random_crop_line(image, # function from ChartOCR, some extra functionality
                                  bboxes,
                                  random_scales=self.rand_scales,
                                  view_size=self.img_size,
                                  border=self.padding)
    else:
      image, border, offset = crop_image(image,
                                         center=[image.shape[0] // 2, image.shape[1] // 2],
                                         new_size=[max(image.shape[0:2]), max(image.shape[0:2])])
      bboxes[:, 0::2] += border[2]
      bboxes[:, 1::2] += border[0]

    # resize image and bbox
    height, width = image.shape[:2]
    image = cv2.resize(image, (self.img_size['w'], self.img_size['h']))
    bboxes[:, 0:bboxes.shape[1]:2] *= self.img_size['w'] / width  # modified to adjust for len(bbox)
    bboxes[:, 1:bboxes.shape[1]:2] *= self.img_size['h'] / height # modified to adjust for len(bbox)

    # discard non-valid bboxes -> ChartOCR code not really discarding anything here...
    # clipping values above 510 and below 0 to 510 and 0 respectively
    # possible problem - will cause wrong annotations? especially near edges?
    bboxes[:, 0:bboxes.shape[1]:2] = np.clip(bboxes[:, 0:bboxes.shape[1]:2], 0, self.img_size['w'] - 1)
    bboxes[:, 1:bboxes.shape[1]:2] = np.clip(bboxes[:, 1:bboxes.shape[1]:2], 0, self.img_size['h'] - 1)

    # RANDOM FLIPS REMOVED !!!

    image = image.astype(np.float32) / 255.
    # randomly change color and lighting
    if self.split == 'train':
      color_jittering_(self.data_rng, image)
      lighting_(self.data_rng, image, 0.1, self.eig_val, self.eig_vec)

    # NORMALIZATION REMOVED IN CHARTOCR????? [REVIEW]
    # image -= self.mean
    # image /= self.std
    image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

    


    return {'image_id':img_id,
            'image': image, 
            'size':list(self.image_size),
            'ori_size':ori_size,
            'bboxes':bboxes

            }

  def __len__(self):
    return self.num_samples