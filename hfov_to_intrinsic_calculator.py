import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

	print(sys.argv[1])
	hfov = float(sys.argv[1])
	img_width  = int(sys.argv[2])
	img_height = int(sys.argv[3])
	# calculate vertical fov from hfov, height and width
	vfov = np.arctan(np.tan(hfov/2)/img_width*img_height)*2 # derive from hfov 
	fx = (img_width * 0.5)/np.tan(hfov*0.5)
	fy = (img_height * 0.5)/np.tan(vfov*0.5)
	print("fx, fy: ")
	print(fx, fy)
	
	camera_intrinsic = np.array([[fx, 0 , img_width/2 ],
	                             [0 , fy, img_height/2],
	                             [0 , 0 , 1           ]])
	print("intrinsic: ")
	print(camera_intrinsic)
	print()
	print(camera_intrinsic.reshape(9).tolist())