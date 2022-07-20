import math
import json
import math
import cv2
# import torchvision
from PIL import Image
import os
import random
from utils import listdir, mean_average
#
# path="../dataprocessing/patches_process/frames_patch_arrange/all_patches/0/"
# # G:\Projects\Tracker\multi-object-tracker\logprocess\videoprocess\frames_patch\naverformworka\all_patches\0\1_0c803-623-70-165c.jpg
# images_name = os.listdir(path)
# # video_5138.jpg
# images_name.sort(key=lambda x:int(x.split("_")[0]))
# # print(images_name)

# def centcoor():
#     for im_num in range(len(images_name)):
#         image = cv2.imread(path + images_name[im_num])
#         img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if im_num%1==0:
#             with open("../video_process/all_coors.txt","r")as f:
#                 lines=f.readlines()
#                 line=lines[int(im_num/1)]
#                 coor=line.split("_")[1]
#                 x=int(coor.split("-")[0])
#                 y=int(coor.split("-")[1])
#                 xx=int(coor.split("-")[2])
#                 yy=int(coor.split("-")[3])
#
#             center_coor=(x+xx/2,y+yy/2)
#             with open("../dataprocessing/centercoor.json", "a")as f:
#                 json_str=json.dumps(center_coor)
#                 f.write(json_str)
#                 f.write("\n")


all_coors=[]
def cal_centercoor_distance(interval,image_paths,obj_dir):
    center_coor_list=[]
    if not os.path.exists("./{}/distance".format(video_name)):
        os.makedirs("./{}/distance".format(video_name))
    if os.path.exists("./{}/distance/distance_obj_{}.txt".format(video_name,obj_dir)):
        os.remove("./{}/distance/distance_obj_{}.txt".format(video_name,obj_dir))
    print("image_paths",image_paths)
    image_paths.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
    for im_num in range(len(image_paths)):
        if im_num==0 or (im_num+1)%30==0:
            print(image_paths[im_num])
            # image = cv2.imread(image_paths[im_num])
            # size=image.shape
            coor=image_paths[im_num].split("\\")[-1].split("c")[1]
            leftx=int(coor.split("-")[0])
            lefty=int(coor.split("-")[1])
            xx=int(coor.split("-")[2])
            yy=int(coor.split("-")[3])
            center_coor=(leftx+xx/2,lefty+yy/2)

            center_coor_list.append(center_coor)
            with open("./{}/distance/center_coor_obj_{}.txt".format(video_name,obj_dir),"a")as f:
                f.write("{}".format(center_coor))
                f.write("\n")
    distancelist=[]
    for j in range(1,len(center_coor_list)):
        distancex=float(center_coor_list[j][0]-center_coor_list[j-1][0])
        distancey=float(center_coor_list[j][1]-center_coor_list[j-1][1])
        distance= math.sqrt(math.pow(distancex, 2) + math.pow(distancey, 2))
        distancelist.append(distance)
        with open("./{}/distance/distance_obj_{}.txt".format(video_name, obj_dir), "a") as f:
            f.write("{:.5f}".format(distance))
            f.write("\n")
    normal_distance(distancelist,obj_dir)

def normal_distance(distancelist,obj_dir):
    avg_distance_list = mean_average(distancelist, block=5)
    for rate in avg_distance_list:
        normaldistance = rate / max(avg_distance_list)
        with open("./{}/distance/normaldistance_obj_{}.txt".format(video_name, obj_dir), "a") as f:
            f.write("{:.5f}".format(normaldistance))
            f.write("\n")

if __name__ == '__main__':
    # 设置路径
    curr_path = os.getcwd()
    base_path = os.path.dirname(curr_path)
    video_name="NAVER1784_ch3_32249"
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
            cal_centercoor_distance(interval_a,obj_img_paths,dir)

