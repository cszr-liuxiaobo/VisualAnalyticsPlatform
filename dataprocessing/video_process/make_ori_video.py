import os
import cv2

from utils import listdir

video_name="NAVER1784_ch3_20190831080000_20190831090000"
track_imgs_list = []
track_imgs_path = "./frames_ori/{}/allframes".format(video_name)
trackimgs_list = listdir(track_imgs_path, track_imgs_list)
trackimgs_list.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
# 定义制作视频的参数
imageshape = cv2.imread(trackimgs_list[0]).shape
size = (imageshape[1], imageshape[0])
print(imageshape)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = 30
out = cv2.VideoWriter('./tracking_ori_video_{}.avi'.format(video_name), fourcc, fps, size)
print("trackimgs_list::::",len(trackimgs_list))
for i in range(0,32249):
    print(i)
    img = cv2.imread(trackimgs_list[i])
    out.write(img)
