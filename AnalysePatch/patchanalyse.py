"""
1.指定每个视频的矩形区域
2.矩形区域内做random获取patch，且能够自行指定每个patch的百分比（5,15,25,50,75,100）和数量（0-100个，单位为10）
3.获取然后存储到对应文件夹里，自行分类然后存储到对应的文件夹里
"""
import cmath
import os
import time

import cv2
from PIL import Image
from torchvision import transforms

from findcoordinate import getpointcoor
# from frame_dealing import prevent_rect_shaking

currdir = os.getcwd()
# >>此处设定video名字
videoname = 'video1.mp4'
videopath = os.path.join(currdir,"video",videoname)

# 规定每个work area 截取patch的数量
patch_number=100

# cropareas 需要进行随机生成，并根据计算出的总面积比例进行展示
def calculationarea(workarea_coor):
    # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    x_len=abs(workarea_coor[2]-workarea_coor[0])
    y_len=abs(workarea_coor[3]-workarea_coor[1])
    area=x_len*y_len
    area_percent5=area/20
    area_percent10=area/10
    area_percent25=area/4
    area_percent50=area/2
    # 后续完善成为一个选项，用户可以选择上述几种规格的patchsize
    # area_percent_list=[]

    # area_percents = {}
    # for areas in area_percents:
    #     area_percent_list.append(area_percent5)

    patch_sizes_dir= {
        "patchsize5%": area_percent5,
        "patchsize10%": area_percent10,
        "patchsize25%": area_percent25,
        "patchsize50%": area_percent50,
    }
    return patch_sizes_dir


# 获取视频的第一帧，然后通过点击获取坐标
def get_pointcoor():
    i = 0

    # 开始读视频
    videoCapture = cv2.VideoCapture(videopath)
    while True:
        success, frame = videoCapture.read()
        i += 1
        if (i==5):
            coor_list=getpointcoor(frame,draw_rect=True)
            videoCapture.release()
            return coor_list
        if not success:
            print('video is all read')
            break
    videoCapture.release()
    time.sleep(5)
a = input("是否需要获取video的坐标?请输入Y或者N： Y/N")
if a =="Y":
    # 初版只进行一次性定位，第二版再做好综合定位
    coor_list = get_pointcoor()
    print(coor_list)
# 指定工作区域
# workarea=coor_list
# v1_coor_list=[(34, 292), (658, 680), (668, 309), (1274, 663)]





dx_list = []
dy_list = []
def get_crop_area(workareapoint,area_num,list_xya,img,framenum):
    # 给视频加上防抖算法，然后对抽取出的每一张图进行核心区域的切割
    x = list_xya[0]
    y = list_xya[1]
    a = list_xya[2]
    dx_list.append(x)
    dy_list.append(y)
    dx = int(sum(dx_list))
    dy = int(sum(dy_list))
    # The x, y coordinates of the areas to be cropped. (x1, y1, x2, y2)
    crop_areas=[]
    for point in workareapoint:
        w = point[0]+dx
        h = point[1]+dy
        crop_areas.append((w,h))
    print(crop_areas)
    new_workareapoint = crop_areas[0]+crop_areas[1]
    print("new_workareapoint",new_workareapoint)
    # 对工作区域进行截图，并传递给patch截图层
    # cropname = 'frame{}'.format(framenum) + "crop{}".format(i) + '.jpg'
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cropped_workarea = im.crop(new_workareapoint)
    # 计算出需要分割的区域对应的面积及各patch的面积
    patch_sizes = calculationarea(new_workareapoint)

    # 在workarea中截取random patch,,此处选择的长宽是正方形的，可以修改成其他比例
    for patch_size_key,patch_size_value in patch_sizes.items():
        hight= patch_size_value ** 0.5
        width= patch_size_value ** 0.5
        for patch_i in range(patch_number):
            patch=transforms.RandomCrop((int(hight),int(width)))(cropped_workarea)
            patchpath=currdir+"\\"+"patches"+"\\"+videoname[:-4]+area_num+"\\"+"frame"+str(framenum)+"\\"+patch_size_key+"_{}x{}".format(int(width),int(hight))
            patchpath_concrete=currdir+"\\"+"patches"+"\\"+videoname[:-4]+area_num+"\\"+"frame"+str(framenum)+"\\"+patch_size_key+"_{}x{}".format(int(width),int(hight))+"\\"+"concrete"
            patchpath_framework=currdir+"\\"+"patches"+"\\"+videoname[:-4]+area_num+"\\"+"frame"+str(framenum)+"\\"+patch_size_key+"_{}x{}".format(int(width),int(hight))+"\\"+"framework"
            patchpath_rebar=currdir+"\\"+"patches"+"\\"+videoname[:-4]+area_num+"\\"+"frame"+str(framenum)+"\\"+patch_size_key+"_{}x{}".format(int(width),int(hight))+"\\"+"rebar"

            # 创建不存在的文件夹
            if not os.path.exists(patchpath):
                os.makedirs(patchpath)
            if not os.path.exists(patchpath_concrete):
                os.makedirs(patchpath_concrete)
            if not os.path.exists(patchpath_framework):
                os.makedirs(patchpath_framework)
            if not os.path.exists(patchpath_rebar):
                os.makedirs(patchpath_rebar)

            patch.save(patchpath+"\\patch{}.jpg".format(patch_i))


    # 在此处进行new_crop的切割

    # Loops through the "crop_areas" list and crops the image based on the coordinates in the list


def get_video_patch(workarea,area_num,capture):
    # 获取矩形框内的patch，设定比例，设定数量
    count = 50
    i = 0
    j = 0

    while True:
        success, frame = capture.read()
        if not success:
            print('video is all read')
            break
        i += 1
        if (i % count == 0):

            list_xya = prevent_rect_shaking(frame)
            get_crop_area(workarea,area_num,list_xya, frame,i)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            print(j)
            j += 1


if __name__ == '__main__':
    # 注意必须在此处拆分成两个点，且调整为左上右下角排列
    v1_coor_list = [    (79, 275),    (648, 619),    (676, 258),    (1265, 641)]
    v1_coor_list1 = [(79, 275),    (648, 619)]
    v1_coor_list2 = [(676, 258),    (1265, 641)]

    # 临时思路：每一帧截取一个workarea直接送入patch截取层；然后根据比例求出长宽，对每帧的workarea截取这一长和宽的patch
    # workarea,area_num=v1_coor_list1,"area1"
    workarea2,area_num2=v1_coor_list2,"area2"

    capture = cv2.VideoCapture(videopath)

    get_video_patch(workarea2,area_num2, capture)



    # #尝试采用transformer randomcrop
    # for
    # transforms.RandomCrop((avg_hight, avg_width))





