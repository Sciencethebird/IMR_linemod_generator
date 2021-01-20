import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


img_width = 640
img_height = 480

fx = 609.513427734375
hfov = 2*np.arctan2(img_width,(2*fx))
print(hfov)
# calculate vertical fov from hfov, height and width
vfov = np.arctan(np.tan(hfov/2)/img_width*img_height)*2 # derive from hfov 
fx = (img_width * 0.5)/np.tan(hfov*0.5)
fy = (img_height * 0.5)/np.tan(vfov*0.5)
print(fx, fy)

# realsense hfov@1280*720 = 0.6733603022466694
# realsense hfov@640*480 =  0.9669080233152941
