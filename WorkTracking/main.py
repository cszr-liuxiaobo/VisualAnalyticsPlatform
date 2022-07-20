import os
from WorkTracking.mot_YOLOv3 import YOLOdetect

# 首先进行YOLO检测，获取全部person每一帧的patch并存储在指定文件夹
curr_dir=os.getcwd()
print(curr_dir)
base_dir=os.path.dirname(curr_dir)
# ###################注意修改videos为videos1,videos2...###########################
video_path=base_dir+"/dataprocessing/video_process/videos3/"
video_name=os.listdir(video_path)[0]
print("video_name:",video_name)
which_video=video_path+video_name
ptchsavepath = base_dir + "/dataprocessing/patches_process/frames_patch/{}/".format(video_name.split(".")[0])

print(which_video)
print(ptchsavepath)

YOLOdetect(which_video,ptchsavepath)
