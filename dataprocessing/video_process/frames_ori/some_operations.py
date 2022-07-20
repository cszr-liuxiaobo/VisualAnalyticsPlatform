"""
用来隔250帧选择一张图片
"""

import os
import shutil

import cv2

curr_dir=os.getcwd()
dir_name="NAVER1784_ch3_20190830080000_20190830090000"
images_path = curr_dir+'/{}/allframes/'.format(dir_name)
images_list = os.listdir(images_path)
images_list.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

i=0
for img in images_list:
    print(img)
    img_index=int(img.split("-")[-1].split(".")[0])
    savedpath = curr_dir+'/{}/VOCdevkit/VOC2007/JPEGImages/'.format(dir_name)
    if not os.path.exists(savedpath):
        os.makedirs(savedpath)
    # 生成VOC文件夹系统
    VOC_dir1=curr_dir + '/{}/VOCdevkit/VOC2007/Annotations/'.format(dir_name)
    VOC_dir2=curr_dir + '/{}/VOCdevkit/VOC2007/ImageSets/Main/'.format(dir_name)
    VOC_dir3=curr_dir + '/{}/VOCdevkit/VOC2007/labels/'.format(dir_name)
    if not os.path.exists(VOC_dir1):
        os.makedirs(VOC_dir1)
    if not os.path.exists(VOC_dir2):
        os.makedirs(VOC_dir2)
    if not os.path.exists(VOC_dir3):
        os.makedirs(VOC_dir3)
    shutil.copy("./bak/voc_label.py", "./{}/voc_label.py".format(dir_name))

    if img_index % 250 == 0:
        print(img_index)
        frame=cv2.imread(images_path+img)
        cv2.imwrite(savedpath+img, frame)
