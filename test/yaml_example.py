import yaml

obj_pos = {
    "chimp":[0.082722, -0.071363, 1.06088, 0, 0, -0.667257],
    "drill":[0, 0, 1.119, 0, -0, 0],
    "egg":[-0.090844, -0.127384, 1.05142, 0, 0, -0.607437],
    "iron":[0.070261, 0.132254, 1.08557, 0, 0, -1.09497],
    "waterer":[-0.065568, 0.115477, 1.11187, 0, -0, 2.19359]
}
with open('input/2020_11_16_fov1.019/info/object_pose.yml', 'w') as f:
    yaml.dump_all([obj_pos], f) # you need [] to store it properly
    #y = yaml.load(f)
    #print (y['chimp'][1])


with open('gt.yml', 'r') as f:
    #yaml.dump_all([obj_pos, obj_pos], f) # you need [] to store it properly
    y = yaml.load(f)
    print (y)
'''

cam_pose = {}
with open('input/2020_11_16_fov1.019/info/camera_script.txt', 'r') as f:
    lines = f.readlines()

    for idx, line in enumerate(lines):
        nums = line.split(',')
        x, y, z = float(nums[0]), float(nums[1]), float(nums[2])
        cam_pose[idx] = [x, y, z]

with open('input/2020_11_16_fov1.019/info/camera_script.yml', 'w') as f:
    yaml.dump_all([cam_pose], f)
'''


object_color = {"chimp":[6, 6, 6],
                "egg":[102,102, 92], 
                "waterer":[102, 102, 82], 
                "iron":[0, 41, 102], 
                "drill":[0, 31, 0]}

with open('object_color.yml', 'w') as f:
    yaml.dump_all([object_color], f)