"""
进行optical flow value的计算
整体的思想是每隔interval帧就对图像进行一次截图，这次截图必须和前interval帧大小坐标完全一致。
目前设定的interval为1，但是要隔30帧计算一次。即第30帧和第29帧进行光流计算对比。
"""
import time

import opyf
import cv2
import numpy as np
import os
import json
# import torchvision
from PIL import Image
from my_utils.utils_dataprocess import MyEncoder
from utils import listdir,images_onlyname
# 每隔开calcul_interval帧计算一次
calcul_interval=1
# 给开几倍计算一次
change_coor_interval=1
# 以前一帧为对比计算，每隔30帧计算一次
geduoshaozhen=30


def interval_opyval(mean,mean_list,obj_dir):
    with open("./{}/opticalvalue/opticalvalue_obj_{}.txt".format(video_name,obj_dir), "a")as f:
        mean_json=json.dumps(mean,cls=MyEncoder)
        f.write(mean_json)
        f.write("\n")
    mean_list.append(mean)

def optical_flow(one, two,imgsize,im_num,mean_list,obj_dir):
    # one= fromarray(cv2.cvtColor(one, cv2.COLOR_BGR2RGB))
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((imgsize[1],imgsize[0], 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:,:,1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=30,
                                        iterations=1,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    # print(flow)
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    max=np.max(mag)
    mean=np.mean(mag)
    middle=np.median(mag)
    # print(max,mean,middle)
    interval_opyval(mean,mean_list,obj_dir)
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow

def save_oldimage(flage,objs_pathes,image=None):
    if flage==0:
        # print("1111111111111111111111")
        old_image = cv2.imread(objs_pathes + patches_ordered[0])
        return old_image
    # print("22222222")
    old_image=image
    return old_image

def walk_dir(dir, fileinfo, topdown=True):
    for root, dirs, files in os.walk(dir, topdown):
        for name in files:
            fileinfo.append(os.path.join(root, name))


if __name__ == '__main__':
    curr_dir=os.getcwd()
    base_path = os.path.dirname(curr_dir)
    # 注意修改video_name和具体的obj_dir
    video_name="video202192_Trim"
    # obj_dir = "0"
    objs_dirs_path=base_path+"/dataprocessing/patches_process/frames_patch_arrange/{}" \
                            "/".format(video_name)

    obj_dirs=os.listdir(objs_dirs_path)
    print(obj_dirs)
    for obj_dir in obj_dirs:
        predict_obj_path=os.path.join(objs_dirs_path, obj_dir)
        if os.listdir(predict_obj_path) is None:
            print("in dir has no img".format(dir))
            continue

        i = 0
        j = 0
        dir_ori=base_path+"/dataprocessing/video_process/frames_ori/" \
                          "{}/allframes/".format(video_name)
        # obj_dir_coors=base_path+"/dataprocessing/patches_process/obj_{}_coors.txt".format(obj_dir)
        objs_pathes = base_path+"/dataprocessing/patches_process/frames_patch_arrange/{}" \
                                "/{}/".format(video_name,obj_dir)
        # dirs = os.listdir(patch_pathes)
        # for dir in dirs:
        #     predict_obj_path = os.path.join(patch_pathes, dir)
        #     if os.path.isdir(predict_obj_path):
        #         print(predict_obj_path)
        # 这两者crop和ori不需要一一对应，最终的计算以其frame_num为准
        if not os.path.exists("./{}/opticalvalue".format(video_name)):
            os.makedirs("./{}/opticalvalue".format(video_name))
        if os.path.exists("./{}/opticalvalue/opticalvalue_obj_{}.txt".format(video_name,obj_dir)):
            os.remove("./opticalvalue/opticalvalue_obj_{}.txt".format(video_name,obj_dir))
        if os.path.exists("./{}/opticalvalue/normalopticalvalue_obj_{}.txt".format(video_name,obj_dir)):
            os.remove("./{}/opticalvalue/normalopticalvalue_obj_{}.txt".format(video_name,obj_dir))

        mean_list = []
        images_fullpath=[]
        images_onlynames=[]
        images_fullpath = listdir(objs_pathes,images_fullpath)
        patches_ordered,images_num = images_onlyname(images_fullpath,images_onlynames)
        images_name_ori = os.listdir(dir_ori)
        images_name_ori.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
        # 将在obj中出现的帧对应的original image保留
        images_name_ori_for_obj=[]
        images_name_ori_numlist=[]
        for name in images_name_ori:
            num=int(name.split("-")[-1].split(".")[0])
            if num in images_num:
                images_name_ori_for_obj.append(name)
                images_name_ori_numlist.append(num)

        if i==0:
            oldimage = save_oldimage(0,objs_pathes)
        for im_num in range(len(patches_ordered)):
            print("im_num",im_num)
            if (im_num % int(calcul_interval) == 0):
                # print(objs_pathes + patches_ordered[im_num])
                frame = cv2.imread(objs_pathes + patches_ordered[im_num])
                # cv2.imshow("img", frame)
                # cv2.waitKey()

                imgInfo_now = frame.shape
                size_now = (imgInfo_now[1], imgInfo_now[0])  # get the width and hight of img
                imgInfo_pre = oldimage.shape
                size_pre=(imgInfo_pre[1],imgInfo_pre[0])

                min_width=min(imgInfo_now[1],imgInfo_pre[1])
                min_hight=min(imgInfo_now[0],imgInfo_pre[0])
                size=(min_width,min_hight)

                # cv2.imshow('person_detection', frame)

                if i > 0:
                    # save_oldimage(flage=2,objs_pathes,image=frame)
                    # oldimage = cv2.resize(oldimage, size)
                    if im_num % change_coor_interval == 0:
                        # print("im_num:",im_num)
                        # print(patches_ordered[im_num])
                        # 获取patches_ordered中对应顺序下的图片分割出frame_num，
                        # 根据这个frame_num在images_name_ori_numlist中的下标找到dir_ori中对应下标的原图，这就做到了一一对应。
                        crop_dir_num=int(patches_ordered[im_num].split("_")[0])
                        ori_correspond_index=images_name_ori_numlist.index(crop_dir_num)
                        # print(crop_dir_num,images_name_ori_numlist,"\n",ori_correspond_index)
                        # print(dir_ori + images_name_ori_for_obj[ori_correspond_index])
                        newcrop_nowimg=cv2.imread(dir_ori + images_name_ori_for_obj[ori_correspond_index])
                        # # 附加一个鲁棒性操作，即如果patches_ordered里有不连续的情况就加一个空格，
                        # if int(crop_dir_num)-1 != int(patches_ordered[im_num-1].split("_")[0]):
                        #     with open("./dataprocess/makeimages_{}.txt".format(calcul_interval), "a")as f:
                        #         f.write("\n")
                        # 用patches图的坐标剪切now图的patch
                        # with open(obj_dir_coors,"r")as f:
                        #     lines=f.readlines()
                        #     line=lines[im_num-calcul_interval]
                        #     # print("line",im_num-calcul_interval,line)
                        #     coor = line.split("_")[1]
                        #     x = int(coor.split("-")[0])
                        #     y = int(coor.split("-")[1])
                        #     xx = int(coor.split("-")[2])
                        #     yy = int(coor.split("-")[3])
                        # crop_area = (x, y)
                        # crop_area = (crop_area[0], crop_area[1], crop_area[0] + xx, crop_area[1] + yy)

                        now_patch_name=os.path.basename(objs_pathes + patches_ordered[im_num])
                        x=now_patch_name.split("c")[1].split("-")[0]
                        y=now_patch_name.split("c")[1].split("-")[1]
                        crop_area = (float(x), float(y))
                        img = Image.fromarray(cv2.cvtColor(newcrop_nowimg, cv2.COLOR_BGR2RGB))
                        oldimageinfo=oldimage.shape
                        crop_area = (crop_area[0], crop_area[1], crop_area[0] + oldimageinfo[1], crop_area[1] + oldimageinfo[0])
                        print("crop_area",crop_area)
                        newcrop_nowimg = img.crop(crop_area)
                        newcrop_nowimg = cv2.cvtColor(np.array(newcrop_nowimg), cv2.COLOR_RGB2BGR)

                        # cv2.imshow("newcrop_nowimg",newcrop_nowimg)
                        print(im_num,size_pre, size_now)
                        print(oldimage.shape,newcrop_nowimg.shape)

                        # The size of the previous frame is the same as that of this frame
                        if im_num%geduoshaozhen==0:
                            optical_flow(oldimage, newcrop_nowimg, size_pre,im_num,mean_list,obj_dir)
                        print("=====================")
                    else:
                        continue
                        # print("----------------")
                        # print(im_num,size_pre, size_now)
                        # optical_flow(oldimage, frame,size,im_num)
                    # 注意oldimage存的是每一帧的，不是每30帧的，别搞混了。
                    oldimage = save_oldimage(flage=2,objs_pathes=objs_pathes, image=frame)
                # oldimage = save_oldimage(flage=2,objs_pathes,image=frame)

                # cv2.imshow('person_detection', frame)

                key = cv2.waitKey(100)& 0xFF
                if key == ord(' '):
                    cv2.waitKey(0)
                if key == ord('q'):
                    break
                j += 1

            i += 1

    time.sleep(3)
    # 生成归一化后的光流
    from utils import mean_average
    optival_path=curr_dir+"/{}/".format(video_name)+"opticalvalue"
    opticalvalue_files=os.listdir(optival_path)
    for opticalvalue_obj in opticalvalue_files:
        with open(optival_path+"/"+opticalvalue_obj, "r")as f:
            lines = f.readlines()
        json_list = []
        for line in lines:
            json_str = json.loads(line)
            json_list.append(json_str)
        print(type(json_list[0]))
        print(json_list)

        average_value = mean_average(json_list, block=5)
        print(average_value)
        if os.path.exists(optival_path+"/normal{}".format(opticalvalue_obj)):
            os.remove(optival_path+"/normal{}".format(opticalvalue_obj))

        for value in average_value:
            normalopticalvalue = float(value / max(average_value))
            with open(optival_path+"/normal{}".format(opticalvalue_obj), "a")as f:
                f.write(str(normalopticalvalue))
                f.write("\n")