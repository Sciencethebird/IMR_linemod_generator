import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import utils as ut
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_folder_name", type=str,
                    help="specify input folder")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("--intrinsics", nargs="+", type=float, default=[ 609.513427734375, 0.0, 322.82989501953125, 0.0,  609.4949951171875, 242.43870544433594, 0.0, 0.0, 1.0] )
args = parser.parse_args()


# object id
obj_ids = {'chimp': 1, 'waterer': 5, 'drill': 8, 'egg': 10, 'iron': 13}

# output directory setting
output_path = 'output'
target_folder = args.input_folder_name

# working files and directory
output_dir = os.path.join(output_path, target_folder)
validation_output_dir = os.path.join(output_dir, 'data')

# open model.ply
points = ut.load_model_point_cloud("models/chimp.ply", 200)

# show model on plt
ut.show_point_cloud(points)

# load object extrinsic
with open(os.path.join('output',target_folder, 'data','01','gt.yml'), 'r') as f:
    extrinsics = yaml.load(f)
#print(extrinsics)
synth_intrinsic_list = [609.513427734375, 0.0, 322.82989501953125, 0.0, 609.4949951171875, 242.43870544433594, 0.0, 0.0, 1.0] #from hFOV
real_intrinsic_list = args.intrinsics
 # from linemod dataset
synth_intrinsic = np.array(synth_intrinsic_list).reshape((3, 3))
real_intrinsic  = np.array( real_intrinsic_list).reshape((3, 3))
intrinsic = real_intrinsic 

validation_input_folder  = os.path.join(validation_output_dir, '01', 'rgb')
validation_output_folder = os.path.join(validation_output_dir, '01', 'val_rgb')
print( validation_input_folder )

sample_rate = 1
for idx, image_name in enumerate(sorted(os.listdir(validation_input_folder))):
    if (idx % sample_rate == 0):
        #print(extrinsics[idx][0]['cam_R_m2c'])
        print(image_name)
        extrinsic = ut.combine_RT_list(extrinsics[idx][0]['cam_R_m2c'], extrinsics[idx][0]['cam_t_m2c'])
        points_cam_frame = np.matmul(extrinsic , points.transpose())
        pixel_coord = np.matmul(intrinsic, points_cam_frame[0:3, :])
    
        img = Image.open(os.path.join(validation_input_folder, image_name))
        ut.show_point_cloud_on_img(pixel_coord, img, save_fig = True, path =os.path.join(validation_output_folder , image_name) )