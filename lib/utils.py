import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

# get rx, ry, rz from camera x, y, z and pointing coord
def get_camera6dcoord(cam_xyz, point_at):
    dx = point_at[0] - cam_xyz[0]
    dy = point_at[1] - cam_xyz[1]
    dz = point_at[2] - cam_xyz[2]
    base = np.sqrt(np.sum(np.square([dx, dy])))
    yaw = np.arctan2(dy,dx)
    pitch = -1.0*np.arctan2(dz,base)
    rotation = [0.0, float(pitch), float(yaw)]
    #print(rotation)
    return cam_xyz + rotation


# inverse homogeneous matrix fuction
def inv_homogenous_matrix(rvec, tvec):
    #https://mathematica.stackexchange.com/questions/106257/how-do-i-get-the-inverse-of-a-homogeneous-transformation-matrix
    #print(rvec)
    rvec_out = inv(rvec)
    #print(np.matmul(rvec_out, rvec))
    #print(-1*rvec_out)
    #print(tvec)
    tvec_out = np.matmul(-1*rvec_out, tvec.transpose())
    return rvec_out, tvec_out

# calculate rvec, tvec function
def extrinsic_from_coord(coord, inv = 0):
    
    roll, pitch, yaw = coord[3:6]

    Rx = [[1,            0,             0],
          [0, np.cos(roll), -np.sin(roll)],
          [0, np.sin(roll),  np.cos(roll)]]

    Ry = [[ np.cos(pitch), 0, np.sin(pitch)],
          [0             , 1,             0],
          [-np.sin(pitch), 0, np.cos(pitch)]]

    Rz = [[np.cos(yaw), -np.sin(yaw), 0],
          [np.sin(yaw),  np.cos(yaw), 0],
          [          0,            0, 1]]


    rvec = np.matmul( np.array(Rz), np.matmul(np.array(Ry), np.array(Rx)) ) #gazebo rotation order x->y->z
    tvec = np.array(coord[0:3])
    #print(rvec, tvec)

    if inv:
        rvec, tvec = inv_homogenous_matrix(rvec, tvec)
        #print(rvec, tvec)

    trivial_row = np.array([0, 0, 0, 1]).reshape((1, 4))
    extrinsic = np.append(rvec, tvec.reshape((3, 1)), axis=1)
    extrinsic = np.append(extrinsic, trivial_row, axis=0)

    return extrinsic

# combine RT
def combine_RT_list(rvec_list, tvec_list):
    rvec = np.array(rvec_list).reshape((3, 3))
    tvec = np.array(tvec_list).transpose()
    extrinsic = np.zeros((4, 4))
    extrinsic[0:3, 0:3] = rvec
    extrinsic[0:3, 3] = tvec
    extrinsic[3, 3] = 1
    return extrinsic

# image shifting function
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted


#simulate image from different intrinsic
def simulate_intrinsic(img, original_intrinsic, target_intrinsic):
    w, h = img.shape[1], img.shape[0]
    new_w = np.ceil((target_intrinsic[0,0] / original_intrinsic[0,0]) * w)
    new_h = np.ceil((target_intrinsic[1,1] / original_intrinsic[1,1]) * h)
    
    # image streching
    img = cv2.resize(img, (int(new_w), int(new_h)), interpolation=cv2.INTER_CUBIC )
    img = img[int(new_h/2-h/2):int(new_h/2+h/2), int(new_w/2-w/2):int(new_w/2+w/2)]
    #print(img.shape)
    
    # image shifting
    u_shift = int(target_intrinsic[0,2] - original_intrinsic[0,2])
    v_shift = int(target_intrinsic[1,2] - original_intrinsic[1,2])
    #print()
    img = translate(img, u_shift, v_shift)
    return img
    

def draw_bb(img, obj_bb):
    x, y, w, h = obj_bb
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    return img

# returns a [0, 1] mask from image color block
def color_block_to_binary_mask(img, block_color, use_dir = True):
    if use_dir:
        img = np.array(Image.open(img))
    lower = np.array(block_color)
    upper = np.array(block_color)
    mask = cv2.inRange(img, lower, upper)
    return mask
    
# calculate bounding box, code from densefusion dataset.py
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

# load model point cloud
def load_model_point_cloud(model_path, num_of_points=1000):
    f = open(model_path)
    lines = f.readlines()
    points = []
    point_num = int(lines[3].split()[-1])
    print("point count: ", point_num)
    sample_rate = point_num//num_of_points
    for idx, line in enumerate(lines[17:17+point_num]):
        if idx % sample_rate == 0:
            x, y, z = line.split(' ')[:3]
            x = float(x)
            y = float(y)
            z = float(z)
            # you need to shift all model points cuz zero is not at the bottom of your model
            # gazebo uses model bottom as zero point
            points.append([x, y, z, 1.0])
        
    return np.array(points)

# show point cloud
def show_point_cloud(points):

    fig = plt.figure(figsize=(10,10)) # specify figsize or your image will be distorted
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(0, 0, 0, c = 'red', s = 50)
    plt.show()


# show point cloud onto img
def show_point_cloud_on_img(pixel_coord, img, save_fig = False, path = None):
    img = np.array(img)
    plt.clf()
    plt.imshow(img)
    for row in pixel_coord.transpose():
        row[0] /= row[2]
        row[1] /= row[2]
        row[2] /= row[2]
        if abs(row[0]) < 1000:
            plt.scatter([row[0]], [row[1]], s = 0.01, c = 'red',  marker = '*')
    
    if save_fig:
        plt.savefig(path)
    else:
        plt.show()
