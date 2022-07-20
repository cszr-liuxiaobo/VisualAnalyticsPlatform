# -*- coding:utf8 -*-
import cv2
import os
import time

currpath=os.getcwd()
# 保存图片的路径
video_name="f_c"
video_path = currpath+'/video'
savedpath = currpath+'/patches/samples'
video_list = os.listdir(video_path)

# 保存图片的帧率间隔
count = 100
i = 0
j = 0

video_path_ = os.path.join(video_path, video_name)
# 开始读视频
videoCapture = cv2.VideoCapture(video_path_+".mp4")

while True:
    success, frame = videoCapture.read()
    i += 1
    if (i % count == 0):
        # 保存图片
        j += 1
        savedname = video_name + str(j) + '.jpg'
        cv2.imwrite(os.path.join(savedpath, savedname), frame)
        print('image of %s is saved' % (savedname))

    if not success:
        print('video is all read')
        break
videoCapture.release()
time.sleep(5)


