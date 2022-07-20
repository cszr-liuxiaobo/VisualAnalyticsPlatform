"""
进行optical flow value的计算
"""
import opyf
import cv2
import numpy as np
import os
import json
# import torchvision
from PIL import Image
from my_utils.utils_dataprocess import MyEncoder

calcul_interval=5
change_coor_interval=1

def optical_flow(one, two,imgsize,im_num):
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

    with open("./dataprocess/opticalvalue/opticalvalue_{}.txt".format(calcul_interval), "a")as f:
        mean_json=json.dumps(mean,cls=MyEncoder)
        f.write(mean_json)
        f.write("\n")
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow

def save_oldimage(flage,image=None):
    if flage==0:
        print("1111111111111111111111")
        old_image = cv2.imread(dir_crop + images_name[0])
        return old_image
    print("22222222")
    old_image=image
    return old_image

def walk_dir(dir, fileinfo, topdown=True):
    for root, dirs, files in os.walk(dir, topdown):
        for name in files:
            fileinfo.append(os.path.join(root, name))

i=0
j=0
if __name__ == '__main__':
    curr_dir=os.getcwd()
    dir_ori=curr_dir+"/video_process/frames_ori/naverformworkb/frames_choose/103390-105146/"
    dir_crop = curr_dir+"/video_process/frames_patch/naverformworkb/all_patches/0/"
    all_coors="./video_process/all_coors.txt"
    # 这两者crop和ori不需要一一对应，最终的计算以其frame_num为准
    images_name = os.listdir(dir_crop)
    images_name.sort(key=lambda x: int(x.split("_")[0]))
    images_name_ori = os.listdir(dir_ori)
    images_name_ori.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    images_name_ori_numlist=[]
    for name in images_name_ori:
        num=name.split("_")[-1].split(".")[0]
        images_name_ori_numlist.append(num)

    if i==0:
        oldimage = save_oldimage(0)
    for im_num in range(len(images_name)):
        print("im_num",im_num)
        if (im_num % int(calcul_interval) == 0):
            frame = cv2.imread(dir_crop + images_name[im_num])
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
                # save_oldimage(flage=2,image=frame)
                # oldimage = cv2.resize(oldimage, size)
                if im_num % change_coor_interval == 0:
                    print("im_num:",im_num)
                    print(images_name[im_num])
                    # 获取crop_dir种对应顺序下元素的frame_num，根据这个frame找到ori_dir中的元素，这是一一对应的。
                    crop_dir_num=int(images_name[im_num].split("_")[0])
                    ori_correspond_index=images_name_ori_numlist.index(str(crop_dir_num))
                    print(crop_dir_num,images_name_ori_numlist,"\n",ori_correspond_index)
                    print(dir_ori + images_name_ori[ori_correspond_index])
                    newcrop_nowimg=cv2.imread(dir_ori + images_name_ori[ori_correspond_index])
                    # 附加一个鲁棒性操作，即如果crop_dir里有不连续的情况就加一个空格，
                    if int(crop_dir_num)-1 != int(images_name[im_num-1].split("_")[0]):
                        with open("./dataprocess/makeimages_{}.txt".format(calcul_interval), "a")as f:
                            f.write("\n")
                    # q用pre图的坐标剪切now图的patch
                    with open(all_coors,"r")as f:
                        lines=f.readlines()
                        line=lines[im_num-calcul_interval]
                        print("line",im_num-calcul_interval,line)
                        coor = line.split("_")[1]
                        x = int(coor.split("-")[0])
                        y = int(coor.split("-")[1])
                        xx = int(coor.split("-")[2])
                        yy = int(coor.split("-")[3])
                    img = Image.fromarray(cv2.cvtColor(newcrop_nowimg, cv2.COLOR_BGR2RGB))
                    crop_area = (x, y)
                    crop_area = (crop_area[0], crop_area[1], crop_area[0] + xx, crop_area[1] + yy)
                    print("crop_area",crop_area)
                    newcrop_nowimg = img.crop(crop_area)
                    newcrop_nowimg = cv2.cvtColor(np.array(newcrop_nowimg), cv2.COLOR_RGB2BGR)

                    cv2.imshow("newcrop_nowimg",newcrop_nowimg)
                    print(im_num,size_pre, size_now)
                    print(oldimage.shape,newcrop_nowimg.shape)

                    # The size of the previous frame is the same as that of this frame
                    optical_flow(oldimage, newcrop_nowimg, size_pre,im_num)
                    print("=====================")
                else:
                    print("----------------")
                    print(im_num,size_pre, size_now)
                    optical_flow(oldimage, frame,size,im_num)
                oldimage = save_oldimage(flage=2, image=frame)
            # oldimage = save_oldimage(flage=2,image=frame)

            # cv2.imshow('person_detection', frame)

            key = cv2.waitKey(100)& 0xFF
            if key == ord(' '):
                cv2.waitKey(0)
            if key == ord('q'):
                break
            j += 1

        i += 1

