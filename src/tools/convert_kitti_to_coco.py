from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2
import os
from tqdm import tqdm

import _tools_init

from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

DEBUG = True

DATA_PATH = '../../data/kitti/'
# VAL_PATH = DATA_PATH + 'training/label_val/'
SPLITS = ['3dop', 'subcnn']
'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''


def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]


def read_clib(calib_path):
  f = open(calib_path, 'r')
  lines = f.readlines()
  f.close()
  for i, line in enumerate(lines):
    if i == 2:
      calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
      calib = calib.reshape(3, 4)
      return calib


cats = [
    'Pedestrian', 'Car', 'Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram',
    'Misc', 'DontCare'
]
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})

color_map = {
    'Pedestrian': (255, 0, 0),
    'Car': (0, 255, 0),
    'Cyclist': (0, 0, 255),
    'Van': (0, 255, 255),
    'Truck': (255, 0, 255),
    'Person_sitting': (255, 255, 0),
    'Tram': (0, 125, 125),
    'Misc': (125, 0, 125),
    'DontCare': (255, 255, 255)
}

for SPLIT in SPLITS:
  image_set_dir = os.path.join(DATA_PATH, f'ImageSets_{SPLIT}')
  if not os.path.isdir(image_set_dir):
    print(f'Dir not exist, skipped: {image_set_dir}')
    continue
  splits = ['train', 'val']
  # splits = ['trainval', 'test']
  calib_type = {
      'train': 'training',
      'val': 'training',
      'trainval': 'training',
      'test': 'testing'
  }

  for split in splits:
    vis_dir = os.path.join(DATA_PATH, calib_type[split], 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    f = open(os.path.join(image_set_dir, f'{split}.txt'), 'r')
    lines = f.readlines()
    f.close()
    lines = lines[:256] if split == 'train' else lines[:64]  # debug
    image_to_id = {}
    for line in tqdm(lines):
      line = line.strip()
      if not line:
        continue
      image_id = int(line)
      calib_path = os.path.join(DATA_PATH, calib_type[split], 'calib',
                                f'{line}.txt')
      calib = read_clib(calib_path)
      image_info = {
          'file_name': f'{line}.png',
          'id': int(image_id),
          'calib': calib.tolist()
      }
      ret['images'].append(image_info)
      if split == 'test':
        continue
      ann_path = os.path.join(DATA_PATH, calib_type[split], 'label_2',
                              f'{line}.txt')
      # if split == 'val':
      #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
      f = open(ann_path, 'r')
      anns = f.readlines()
      f.close()

      if DEBUG:
        img_path = os.path.join(DATA_PATH, calib_type[split], 'image_2',
                                image_info['file_name'])
        image = cv2.imread(img_path)

      for ann_ind, txt in enumerate(anns):
        tmp = txt[:-1].split(' ')
        cat_id = cat_ids[tmp[0]]
        truncated = int(float(tmp[1]))
        occluded = int(tmp[2])
        alpha = float(tmp[3])
        bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]
        dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])]
        location = [float(tmp[11]), float(tmp[12]), float(tmp[13])]
        rotation_y = float(tmp[14])

        ann = {
            'image_id': image_id,
            'id': int(len(ret['annotations']) + 1),
            'category_id': cat_id,
            'dim': dim,
            'bbox': _bbox_to_coco_bbox(bbox),
            'depth': location[2],
            'alpha': alpha,
            'truncated': truncated,
            'occluded': occluded,
            'location': location,
            'rotation_y': rotation_y
        }
        ret['annotations'].append(ann)
        if DEBUG and tmp[0] != 'DontCare':
          box_3d = compute_box_3d(dim, location, rotation_y)
          box_2d = project_to_image(box_3d, calib)
          # print('box_2d', box_2d)
          image = draw_box_3d(image, box_2d, color_map[tmp[0]])
          x = (bbox[0] + bbox[2]) / 2
          '''
          print('rot_y, alpha2rot_y, dlt', tmp[0], 
                rotation_y, alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0]),
                np.cos(
                  rotation_y - alpha2rot_y(alpha, x, calib[0, 2], calib[0, 0])))
          '''
          depth = np.array([location[2]], dtype=np.float32)
          pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                           dtype=np.float32)
          pt_3d = unproject_2d_to_3d(pt_2d, depth, calib)
          pt_3d[1] += dim[0] / 2
          # print('pt_3d', pt_3d)
          # print('location', location)
      if DEBUG:
        # cv2.imshow('image', image)
        # cv2.waitKey()
        vis_path = os.path.join(vis_dir, image_info['file_name'])
        cv2.imwrite(vis_path, image)

    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))

    # import pdb; pdb.set_trace()
    out_dir = os.path.join(DATA_PATH, 'annotations')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'kitti_{SPLIT}_{split}.json')
    f = open(out_path, 'w')
    json.dump(ret, f, indent=2)
    f.close()
    print(f'Dump to {out_path}')
