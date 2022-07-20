import os
import cv2
import random
import numpy as np
from PIL import Image

from myutils import affinetransformation

currpath = os.getcwd()
# Crops the image and saves it as "new_filename"
def crop_image(img, crop_area, new_filename):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cropped_image = img.crop(crop_area)
    cropped_image.save(currpath+"\\"+new_filename)
# 展示和截取crop区域
crop_coors=[]
def plot_crop(img_cv,crop_coors):
    i=1
    for image_coor in crop_coors:
        leftup_point=(image_coor[0],image_coor[1])
        rightdown_point = (image_coor[0]+400,image_coor[1]+400)
        cv2.rectangle(img_cv,leftup_point,rightdown_point,(0,0,255),2)
        # cropimage = img_cv[leftup_point[0]:rightdown_point[0],leftup_point[1]:rightdown_point[1]]
        # # cv2.imwrite("./crops/{}_{}.png".format(str(filename),i), cropimage)
        # cv2.imwrite("{}.png".format(i),cropimage)
        i+=1
    cv2.namedWindow('crop_areas', cv2.WINDOW_NORMAL)
    cv2.imshow("crop_areas",img_cv)
    cv2.waitKey()
def single_patch(cropareas,image_name):
    image_name = os.path.join(currpath, image_name)
    img_cv = cv2.imread(image_name)
    plot_crop(img_cv,cropareas)



# 这是目前采用的方法
def get_video_patch(src,savepath,num,sub_image_num_hight=5,sub_image_num_width = 8,flag=None):
    imgInfo = image.shape
    # cnt = 1
    # num = 1
    sub_images = []
    imgsize = (imgInfo[1], imgInfo[0])  # 获取图片宽高度信息
    print(imgsize)
    # --------------------------------------------------------
    src_height, src_width = src.shape[0], src.shape[1]
    sub_height = src_height // sub_image_num_hight
    sub_width = src_width // sub_image_num_width
    for j in range(sub_image_num_hight):
        for i in range(sub_image_num_width):
            if j < sub_image_num_hight - 1 and i < sub_image_num_width - 1:
                image_roi = src[j * sub_height: (j + 1) * sub_height, i * sub_width: (i + 1) * sub_width, :]
                # print(j * sub_height,i * sub_width, (j + 1) * sub_height,  (i + 1) * sub_width)
            elif j < sub_image_num_hight - 1:
                image_roi = src[j * sub_height: (j + 1) * sub_height, i * sub_width:, :]
            elif i < sub_image_num_width - 1:
                image_roi = src[j * sub_height:, i * sub_width: (i + 1) * sub_width, :]
            else:
                image_roi = src[j * sub_height:, i * sub_width:, :]
            sub_images.append(image_roi)
    for i, img in enumerate(sub_images):
        cv2.imwrite(savepath+'{}_{}_sub_img_'.format(1800*num,flag) + str(i) + '.jpg', img)
    # get_crop_area(cropareas, image)
    # plot_crop(src,sub_images)

if __name__ == '__main__':

    # 截图
    # image=cv2.imread("./patches/pourconcrete1806.jpg")
    # crop_area=(480,0,1920, 1080)
    # crop_image(image, crop_area, new_filename="new_pourconcrete1806.jpg")
    # plot_crop(image, crop_areas)


    # --------------------------------------
    # 按比例截取图像，w*h就是个数
    num=18
    image=cv2.imread("../Frames/NAVER1784_ch3_20190831080000_20190831090000-32249.jpg")
    result1,result2=affinetransformation(image)
    savepath=currpath+"/Patches/classification{}/".format(num)
    get_video_patch(result1,savepath,num,5,6,flag="a")
    get_video_patch(result2,savepath,num,5,6,flag="b")

