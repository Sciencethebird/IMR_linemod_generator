import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import utils as ut
import yaml
import numpy.ma as ma

def draw_bb(img, obj_bb):
    x, y, w, h = obj_bb
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    return img

def color_block_to_binary_mask(img, block_color):
    lower = np.array(block_color)
    upper = np.array(block_color)
    mask = cv2.inRange(img, lower, upper)
    return mask
    

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]

with open('object_color.yml', 'r') as f:
    obj_color = yaml.load(f)

label = np.array(Image.open("input/2020_11_16_fov1.019/mask/0000.png"))

synth_intrinsic = [572.411400033838, 0.0, 319.0, 0.0, 572.4114000338382, 239, 0.0, 0.0, 1.0] #from hFOV
real_intrinsic = [572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0] # from linemod dataset
synth_intrinsic = np.array(synth_intrinsic).reshape((3, 3))
real_intrinsic = np.array(real_intrinsic).reshape((3, 3))
label = ut.simulate_intrinsic(label, synth_intrinsic, real_intrinsic)
plt.imshow(label)
plt.show()

target_obj = 'iron'
mask = color_block_to_binary_mask(label, obj_color[target_obj])

mask_out = np.zeros_like(label)
mask_out[:, :, 0] = mask
mask_out[:, :, 1] = mask
mask_out[:, :, 2] = mask
plt.imshow(mask_out)
plt.show()


obj_bb = mask_to_bbox(mask)
print(obj_bb)

img = draw_bb(label,obj_bb)
plt.imshow(img)
plt.show()
