import math
import json
import cv2
# import torchvision
from PIL import Image
import os
import random

from utils import listdir, mean_average

def normal_ratio_rate(ratio_list,obj_dir):
    # 同时也做了average
    if not os.path.exists("./{}/ratiorate".format(video_name)):
        os.makedirs("./{}/ratiorate".format(video_name))
    if os.path.exists("./{}/ratiorate/normalratiorate_obj_{}.txt".format(video_name,obj_dir)):
        os.remove("./{}/ratiorate/normalratiorate_obj_{}.txt".format(video_name,obj_dir))

    ratiochangerate_list=[]
    for i in range(len(ratio_list)):
        if i%interval_a==0 and i>0:
            ratiochangerate=abs(float(ratio_list[i])-float(ratio_list[i-between_b]))
            ratiochangerate_list.append(ratiochangerate)
        elif i >= len(ratio_list):
            break
    avg_ratiochangerate_list=mean_average(ratiochangerate_list,block=5)
    for rate in avg_ratiochangerate_list:
        normalratiorate=rate/max(avg_ratiochangerate_list)
        with open("./{}/ratiorate/normalratiorate_obj_{}.txt".format(video_name,obj_dir), "a") as f:
            f.write("{:.5f}".format(normalratiorate))
            f.write("\n")

def ratio_rate(ratio_list,obj_dir):
    if not os.path.exists("./{}/ratiorate".format(video_name)):
        os.makedirs("./{}/ratiorate".format(video_name))
    if os.path.exists("./{}/ratiorate/ratiochangerate_obj_{}.txt".format(video_name,obj_dir)):
        os.remove("./{}/ratiorate/ratiochangerate_obj_{}.txt".format(video_name,obj_dir))
    for i in range(len(ratio_list)):
        if i%interval_a==0 and i>0:
            ratiochangerate=abs(float(ratio_list[i])-float(ratio_list[i-between_b]))
            with open("./{}/ratiorate/ratiochangerate_obj_{}.txt".format(video_name,obj_dir), "a") as f:
                f.write("{:.5f}".format(ratiochangerate))
                f.write("\n")
        elif i>=len(ratio_list):
            break

def ratio(image_paths,obj_dir):
    ratio_list=[]
    if not os.path.exists("./{}/ratio".format(video_name)):
        os.makedirs("./{}/ratio".format(video_name))
    if os.path.exists("./{}/ratio/ratio_obj_{}.txt".format(video_name,obj_dir)):
        os.remove("./{}/ratio/ratio_obj_{}.txt".format(video_name,obj_dir))
    print("image_paths",image_paths)
    image_paths.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
    for im_num in range(len(image_paths)):
        print(image_paths[im_num])
        image = cv2.imread(image_paths[im_num])
        size=image.shape
        ratio = float(size[1]/size[0])
        ratio_list.append(ratio)
        with open("./{}/ratio/ratio_obj_{}.txt".format(video_name,obj_dir),"a")as f:
            f.write("{:.5f}".format(ratio))
            f.write("\n")
    ratio_rate(ratio_list,obj_dir)
    normal_ratio_rate(ratio_list,obj_dir)

if __name__ == '__main__':
    # 设置路径
    curr_path = os.getcwd()
    base_path = os.path.dirname(curr_path)
    video_name="NAVER1784_ch3_20190831080000_20190831090000"
    patch_pathes=base_path+"/dataprocessing/patches_process/frames_patch_arrange/{}/".format(video_name)

    # 设置步长和计算ratio的帧与帧的间距
    interval_a=30
    between_b=1

    dirs = os.listdir(patch_pathes)
    for dir in dirs:
        img_paths = []
        predict_obj_path=os.path.join(patch_pathes, dir)
        if os.listdir(predict_obj_path) is None:
            print("in dir has no img".format(dir))
            continue
        if os.path.isdir(predict_obj_path):
            print(predict_obj_path)
            obj_img_paths=listdir(predict_obj_path, img_paths)
            ratio(obj_img_paths,dir)

