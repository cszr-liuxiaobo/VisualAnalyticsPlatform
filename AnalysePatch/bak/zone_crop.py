import os
import cv2
import random
import numpy as np
from PIL import Image

from frame_dealing import prevent_rect_shaking

currpath = os.getcwd()

crop_areas = [(414, 278)]

def main(video):
    print(currpath+"/video/{}.mp4".format(video))
    # 创建不存在的文件夹
    if not os.path.exists(currpath+"\\"+"frames"+"\\"+video):
        os.makedirs(currpath+"\\"+"frames"+"\\"+video)

    capture = cv2.VideoCapture(currpath+"/video/{}.mp4".format(video))
    return capture

# Crops the image and saves it as "new_filename"
def crop_image(img, crop_area, new_filename):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cropped_image = im.crop(crop_area)
    cropped_image.save(currpath+"\\"+"frames"+"\\"+video+"\\"+new_filename)

dx_list = []
dy_list = []
def get_crop_area(patchpoint,list_xya,img,framenum):
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
    for point in patchpoint:
        w = point[0]+dx
        h = point[1]+dy
        crop_areas.append((w,h))
    print(crop_areas)

    crop_list = []
    for each_crop in crop_areas:
        new_crop = each_crop + (each_crop[0] + 150, each_crop[1] + 150)
        crop_list.append(new_crop)

    # Loops through the "crop_areas" list and crops the image based on the coordinates in the list
    for i, crop_area in enumerate(crop_list):
        if framenum%50 ==0:
            cropname = 'frame{}'.format(framenum) + "crop{}".format(i) + '.jpg'
            crop_image(img, crop_area, cropname)


# 展示和截取crop区域
def plot_crop(img_cv,crop_coors):
    i=1
    for image_coor in crop_coors:
        leftup_point=(image_coor[0],image_coor[1])
        rightdown_point = (image_coor[0]+700,image_coor[1]+400)
        cv2.rectangle(img_cv,leftup_point,rightdown_point,(0,0,255),2)
        # cropimage = img_cv[leftup_point[0]:rightdown_point[0],leftup_point[1]:rightdown_point[1]]
        # # cv2.imwrite("./crops/{}_{}.png".format(str(filename),i), cropimage)
        # cv2.imwrite("{}.png".format(i),cropimage)
        i+=1
    cv2.imshow("crop_areas",img_cv)
    cv2.waitKey()



def single_patch(cropareas,image_name):

    image_name = os.path.join(currpath, image_name)
    img_cv = cv2.imread(image_name)
    plot_crop(img_cv,cropareas)

def get_video_patch(cropareas,capture):
    count = 5
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
            get_crop_area(cropareas,list_xya, frame,i)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            print(j)
            j += 1


if __name__ == '__main__':
    # # 单张图片演示一下获取的位置。
    # image_name = 'f1.mp41.jpg'
    # single_patch(crop_areas,image_name)
    # --------------------------------------
    video = 'video1'
    capture = main(video)
    get_video_patch(crop_areas,capture)