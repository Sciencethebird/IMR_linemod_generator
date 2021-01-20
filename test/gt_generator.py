import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import utils as ut
import yaml

# object id
obj_ids = {'chimp': 1, 'waterer': 5, 'drill': 8, 'egg': 10, 'iron': 13}

# input directory setting
input_path = 'input'
source_folder = "2020_11_16_linemod"

# output directory setting
output_path = 'output'
target_folder = source_folder+'_out'


# working files and directory
work_dir = os.path.join(input_path, source_folder)
output_dir = os.path.join(output_path, target_folder)
gt_output_dir = os.path.join(output_dir, 'data')

cam_pos_file = os.path.join(work_dir, 'info', 'camera_pose.yml')
obj_pos_file = os.path.join(work_dir, 'info', 'object_pose.yml')
mask_img_dir = os.path.join(work_dir, 'mask')

if not os.path.exists(output_dir):
        os.makedirs(output_dir)
if not os.path.exists(gt_output_dir):
        os.makedirs(gt_output_dir)


# parameter setting
target_object = 'iron'
img_idx = 0
img_names = os.listdir(mask_img_dir)


# load camera poses and object poses and object color
with open(cam_pos_file, 'r') as f:
    cam_pos = yaml.load(f)
with open(obj_pos_file, 'r') as f:
    obj_pos = yaml.load(f)
with open('object_color.yml', 'r') as f:
    obj_color = yaml.load(f)


# calculate extrinsic from camera pose and object pose
rot_to_cam = np.array([[ 0,-1, 0, 0],
                       [ 0, 0,-1, 0],
                       [ 1, 0, 0, 0],
                       [ 0, 0, 0, 1]])


# iterate through all object 
for obj_name, obj_id in obj_ids.items():
    obj_folder = os.path.join(gt_output_dir, '{:02d}'.format(obj_id))
    if not os.path.exists(obj_folder):
        os.makedirs(obj_folder)
        print('created ', obj_folder)
    print('generating ', obj_name, ' gt...')
    gt = {}
    # iterate trough all image
    for idx, img_name in enumerate(img_names):
        print('working on ', img_name, '...')
        if idx != int(img_name[:4]):
            print('[ERROR] index and image name not match!!')

        world_to_cam_extrinsic = ut.extrinsic_from_coord(cam_pos[idx], inv = 1)
        object_to_world_extrinsic = ut.extrinsic_from_coord(obj_pos[obj_name])

        combined_extrinsic = np.matmul(world_to_cam_extrinsic, object_to_world_extrinsic)
        combined_extrinsic = np.matmul(rot_to_cam, combined_extrinsic)

        #print(combined_extrinsic)
        #print(combined_extrinsic[0:3, 0:3].reshape(9).tolist())
        #print(combined_extrinsic[0:3, 3].tolist())

        gt_element = {}
        mask_img_path = os.path.join(mask_img_dir, img_name)
        gt_element['cam_R_m2c'] = [round(num, 8) for num in combined_extrinsic[0:3, 0:3].reshape(9).tolist()]
        gt_element['cam_t_m2c'] = [round(1000*num, 8) for num in combined_extrinsic[0:3, 3].tolist()]
        gt_element['obj_bb'] = ut.mask_to_bbox( ut.color_block_to_binary_mask(mask_img_path, obj_color[obj_name]) )
        gt_element['obj_id'] = obj_id

        gt[idx] = [gt_element]

    with open(os.path.join(obj_folder, 'gt.yml'), 'w') as f:
        yaml.dump_all([gt], f, width=150)

