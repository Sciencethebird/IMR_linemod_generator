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
import yaml
import lib.utils as ut
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("input_folder_name", type=str,
                    help="specify input folder")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("--intrinsics", nargs="+", type=float, default=[ 609.513427734375, 0.0, 322.82989501953125, 0.0,  609.4949951171875, 242.43870544433594, 0.0, 0.0, 1.0] )
args = parser.parse_args()


class linemod_dataset_generator:

    def __init__(self, input_folder_name, output_folder_name):
        # init input and output directory
        self.obj_ids = {'chimp': 1, 'waterer': 5, 'drill': 8, 'egg': 10, 'iron': 13}
        self.input_dir  = os.path.join('input', input_folder_name)

        self.output_dir = os.path.join('output', output_folder_name)
        self.output_data_dir = os.path.join(self.output_dir, 'data')
        self.output_obj_folders = {}
        
        # build directories
        self.build_dir()

    def build_dir(self):
        # output folder
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # output/data
        if not os.path.exists(self.output_data_dir):
            os.makedirs(self.output_data_dir)
        # output/data/01/sub_dirs...
        sub_dirs = ['depth', 'mask', 'rgb', 'val_depth', 'val_mask', 'val_rgb']
        for obj_name, obj_id in self.obj_ids.items():
            obj_folder = os.path.join(self.output_data_dir, '{:02d}'.format(obj_id))
            self.output_obj_folders[obj_name] = obj_folder
            if not os.path.exists(obj_folder):
                os.makedirs(obj_folder)

            for dir in sub_dirs:
                dir = os.path.join(obj_folder, dir)
                if not os.path.exists(dir):
                    os.makedirs(dir)
    
    def gt_info_generator(self, intrinsic_list):
        # load camera poses and object poses and object color
        with open(os.path.join(self.input_dir, 'info', 'camera_poses.yml'), 'r') as f:
            cam_pos = yaml.load(f)
        with open(os.path.join(self.input_dir, 'info', 'object_poses.yml'), 'r') as f:
            obj_pos = yaml.load(f)
        with open(os.path.join(self.input_dir, 'info', 'object_color.yml'), 'r') as f:
            obj_color = yaml.load(f)

        # calculate extrinsic from camera pose and object pose

        rot_to_cam = np.array([[ 0,-1, 0, 0],
                               [ 0, 0,-1, 0],
                               [ 1, 0, 0, 0],
                               [ 0, 0, 0, 1]])

        mask_img_dir = os.path.join(self.input_dir, 'mask')
        img_names = sorted(os.listdir(mask_img_dir))

        # iterate through all object 
        for obj_name, obj_id in self.obj_ids.items():
            obj_folder = self.output_obj_folders[obj_name] 
            print('generating ', obj_name, ' gt...')
            gt = {}
            info = {}
            # iterate trough all image
            for idx, img_name in enumerate(img_names):
                print('generating ', img_name, obj_name, "/", 'gt.yml...')
                if idx != int(img_name[:4]):
                    print('[ERROR] index and image name not match!!')
        
                world_to_cam_extrinsic = ut.extrinsic_from_coord(cam_pos[idx], inv = 1)
                #print(world_to_cam_extrinsic)
                object_to_world_extrinsic = ut.extrinsic_from_coord(obj_pos[obj_name])
                #print(object_to_world_extrinsic)

                combined_extrinsic = np.matmul(world_to_cam_extrinsic, object_to_world_extrinsic)
                combined_extrinsic = np.matmul(rot_to_cam, combined_extrinsic)
        
                gt_element = {}
                mask_img_path = os.path.join(mask_img_dir, img_name)
                gt_element['cam_R_m2c'] = [round(num, 8) for num in combined_extrinsic[0:3, 0:3].reshape(9).tolist()]
                gt_element['cam_t_m2c'] = [round(1000*num, 8) for num in combined_extrinsic[0:3, 3].tolist()]
                gt_element['obj_bb'] = ut.mask_to_bbox( ut.color_block_to_binary_mask(mask_img_path, obj_color[obj_name]) )
                gt_element['obj_id'] = obj_id
  
                gt[idx] = [gt_element]

                info_element = {}
                info_element['cam_K'] = list(intrinsic_list) #avoid yaml anchoring
                info_element['depth_scale'] = 1.0

                info[idx] = info_element
        
            with open(os.path.join(obj_folder, 'gt.yml'), 'w') as f:
                yaml.dump_all([gt]  , f, width=150, default_flow_style=None)
            with open(os.path.join(obj_folder, 'info.yml'), 'w') as f:
                yaml.dump_all([info], f, width=150, default_flow_style=None)

    def image_generator(self, source_intrinisc=None, target_intrinsic=None):

        with open(os.path.join(self.input_dir, 'info', 'object_color.yml'), 'r') as f:
            obj_color = yaml.load(f)

        input_mask_dir = os.path.join(self.input_dir, 'mask')
        input_rgb_dir = os.path.join(self.input_dir, 'rgb')
        input_depth_dir = os.path.join(self.input_dir, 'depth')
        
        # output binary mask
        for obj_name, obj_id in self.obj_ids.items():
            for img_name in sorted(os.listdir(input_mask_dir)):

                img_path = os.path.join(input_mask_dir, img_name)
                img = np.array(Image.open(img_path))
                if source_intrinisc != None:
                    img = ut.simulate_intrinsic(img, source_intrinisc, target_intrinsic)
                mask = ut.color_block_to_binary_mask(img, obj_color[obj_name], use_dir = False)
                mask =np.array(mask)
                mask_out = np.zeros_like(img)

                mask_out[:, :, 0] = mask
                mask_out[:, :, 1] = mask
                mask_out[:, :, 2] = mask
               
                im = Image.fromarray(mask_out)
                print('saveing {}-{} mask...'.format(obj_name, img_name))
                
                obj_folder = self.output_obj_folders[obj_name] 
                im.save(os.path.join(obj_folder ,'mask',  img_name))

        # output rgb (only to chimp folder for time saving purpose)
        for img_name in sorted(os.listdir(input_rgb_dir)):
            img_path = os.path.join(input_rgb_dir, img_name)
            img = Image.open(img_path)
            img = np.array(img) 
            if source_intrinisc != None:
                img = ut.simulate_intrinsic(img, source_intrinisc, target_intrinsic)
            im = Image.fromarray(img)
            print('saveing ', img_name, ' rgb...')
            #for obj_name, obj_id in self.obj_ids.items():
            obj_folder = self.output_obj_folders['chimp'] 
            im.save(os.path.join(obj_folder ,'rgb',  img_name))
        
        # output depth (only to chimp folder cuz it takes forever)
        for img_name in sorted(os.listdir(input_depth_dir)):
            img_path = os.path.join(input_depth_dir, img_name)
            img = Image.open(img_path)
            img = np.array(img) 
            img_cat = np.zeros((480, 640, 3))
            img_cat[:, :, 0] = img
            img_cat[:, :, 1] = img
            img_cat[:, :, 2] = img
            if source_intrinisc != None:
                img = ut.simulate_intrinsic(img_cat, source_intrinisc, target_intrinsic)
                img = img[:, :, 0]
            im = Image.fromarray(img)
            print('saveing ', img_name, ' depth...')
            #for obj_name, obj_id in self.obj_ids.items():
            obj_folder = self.output_obj_folders['chimp'] 
            cv2.imwrite(os.path.join(obj_folder ,'depth',  img_name), img.astype(np.uint16))

    def test_train_script_generator(self, threshold = 2200):
        # this function generate train.txt and test.txt.
        # if the white pixel count is below threshold, the image won't be selected.
        # cuz if a image mask is heavily occluded or has more than one region it's ganna cause problem 
        # in the desefusion traing process.
        print(os.path.join(self.output_obj_folders['chimp'] , 'mask'))
        mask_dir = os.path.join(self.output_obj_folders['chimp'] , 'mask')
        good_dir = os.path.join(self.output_obj_folders['chimp'] , 'good_image')
        valid_images = []
        for file_name in sorted( os.listdir( mask_dir ) ):
            print(file_name)
            img = cv2.imread(os.path.join(mask_dir, file_name) )
            print(img.shape)
            target_pixel_count = 0
            for row in img:
                for pixel in row:
                    if pixel[0] == 255:
                        target_pixel_count += 1
            print(target_pixel_count)
            if target_pixel_count > threshold:
                valid_images.append(file_name)
                cv2.imwrite(os.path.join(good_dir, file_name), img)

        
        y_train, y_test = train_test_split(valid_images, test_size=0.2, random_state=42)
        print(sorted(y_train))
        print(sorted(y_test))
        with open(os.path.join(self.output_obj_folders['chimp'] , 'train.txt'), 'w') as f:
            for num in sorted(y_train):
                f.write(num.split('.')[0]+"\n")
        with open(os.path.join(self.output_obj_folders['chimp'] , 'test.txt'), 'w') as f:
            for num in sorted(y_test):
                f.write(num.split('.')[0]+"\n")
            
         
if __name__ == "__main__":
	print(args.input_folder_name)
	print(args.intrinsics)
	test = linemod_dataset_generator( input_folder_name  = args.input_folder_name,
                                     output_folder_name = args.input_folder_name+"_output")

	# generates ground truth yml
	test.gt_info_generator(args.intrinsics)
	# generates images
	test.image_generator()
	# generates test-train script, also filters out bad images with mask pixel under ceratin threshold
	test.test_train_script_generator(2200)
'''


# fx 914.2700805664062
# fy 914.2424926757812
# px 644.244873046875 , 5.244873046875
# py 363.6580505371094, 4.658050537109375
                
synth_intrinsic_list = [ 609.513427734375, 0.0, 319.0, 0.0,  609.513427734375, 239.0, 0.0, 0.0, 1.0] #from hFOV
real_intrinsic_list = [ 609.513427734375, 0.0, 322.82989501953125, 0.0,  609.4949951171875, 242.43870544433594, 0.0, 0.0, 1.0] # from linemod dataset

synth_intrinsic = np.array(synth_intrinsic_list).reshape((3, 3))
real_intrinsic = np.array(real_intrinsic_list).reshape((3, 3))

# calculate RT from camera and object 6D coordinates and save as yaml
#test.gt_info_generator(real_intrinsic_list)

# this will generate rgb, depth, mask without any intrinsics calibration
#test.image_generator()

# simulate real intrinsic image and output rgb, binary, depth mask
# test.image_generator(synth_intrinsic, real_intrinsic)

test.test_train_script_generator(2200)

'''