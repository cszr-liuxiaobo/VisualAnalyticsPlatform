import os
import cv2
from PIL import Image

obj_dir="6"
path="./frames_patch_32249/NAVER1784_ch3_20190831080000_20190831090000/{}/".format(obj_dir)
images_name = os.listdir(path)
images_name.sort(key=lambda x:int(x.split("_")[0]))
# images_name.sort(key=lambda x:int(x.split("_")[1].split(".")[0]))
print(images_name)

for im_name in images_name:
    in_num=im_name.split("_")[0]
    coor=im_name.split("c")[1]
    with open("obj_{}_coors.txt".format(obj_dir), "a")as f:
        f.write(in_num)
        f.write("_")
        f.write(coor)
        f.write("\n")
    print(im_name)

    # if int(images_name[int(in_num)-1].split("_")[0]) != int(in_num):
    #     with open("obj_{}_coors.txt".format(obj_dir), "a")as f:
    #         f.write("\n")
