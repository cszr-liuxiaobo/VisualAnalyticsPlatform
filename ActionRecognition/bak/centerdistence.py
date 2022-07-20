import math
import json

import cv2
# import torchvision
from PIL import Image
import os
import random
path="../video_process/frames_patch/naverformworkb/all_patches/0/"
# G:\Projects\Tracker\multi-object-tracker\logprocess\videoprocess\frames_patch\naverformworka\all_patches\0\1_0c803-623-70-165c.jpg
images_name = os.listdir(path)
# video_5138.jpg
images_name.sort(key=lambda x:int(x.split("_")[0]))
# print(images_name)

def ratio():
    for im_num in range(len(images_name)):
        image = cv2.imread(path + images_name[im_num])
        size=image.shape
        ratio = (size[1]/size[0])

        with open("bak/ratio/ratio.txt", "a")as f:
            f.write("{:.4f}".format(ratio))
            f.write("\n")

def centcoor():
    for im_num in range(len(images_name)):
        image = cv2.imread(path + images_name[im_num])
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if im_num%1==0:
            with open("../video_process/all_coors.txt","r")as f:
                lines=f.readlines()
                line=lines[int(im_num/1)]
                coor=line.split("_")[1]
                x=int(coor.split("-")[0])
                y=int(coor.split("-")[1])
                xx=int(coor.split("-")[2])
                yy=int(coor.split("-")[3])

            center_coor=(x+xx/2,y+yy/2)
            with open("../dataprocessing/centercoor.json", "a")as f:
                json_str=json.dumps(center_coor)
                f.write(json_str)
                f.write("\n")


all_coors=[]
def cal_centercoor_distance(interval):
    with open("../dataprocessing/centercoor.json", "r", encoding='UTF-8') as f:
        lines=f.readlines()
    for line in lines:
        line=json.loads(line)
        all_coors.append(line)
    print(all_coors)

    for i in range(0,len(all_coors)-interval):
        if i>0 and i% interval==0:
            print(i)
            j=(float(list(all_coors[i])[0])-float(list(all_coors[i-interval])[0]))**2 + \
              (float(list(all_coors[i])[1])-float(list(all_coors[i-interval])[1]))**2
            distence=math.sqrt(j)

            with open("distence/cal_centercoor_distance_{}.json".format(interval),"a") as f:
                json_str=json.dumps(distence)
                f.write(json_str)
                f.write("\n")

def deal_ratio(interval):
    with open("bak/ratio/ratio.txt", "r") as f:
        lines=f.readlines()

    for line in range(len(lines)):
        if line %interval==0:
            line = json.loads(lines[line])
            with open("./ratio/ratio_{}.json".format(interval),"a")as f:
                json_str=json.dumps(line)
                f.write(json_str)
                f.write("\n")

if __name__ == '__main__':
    interval=30
    # ratio()
    # centcoor()
    # ----------------------
    # cal_centercoor_distance(interval)
    deal_ratio(interval)
