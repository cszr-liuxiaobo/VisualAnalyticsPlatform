import cv2
import os

from utils import listdir

curr_path = os.getcwd()
video_name = "video202192_Trim"
video_mins=22
patch_pathes = curr_path + "/frames_patch_classify/{}/".format(video_name)

dirs = os.listdir(patch_pathes)
# dirs=["6"]
for dir in dirs:
    print("----------这是{}个人-----------".format(dir))
    img_paths = []
    y_true = []
    y_pred = []
    objdir_path = os.path.join(patch_pathes, dir)
    if os.path.isdir(objdir_path):
        print(objdir_path) # 1,2,3,4,5,6 full path
        obj_img_paths = listdir(objdir_path, img_paths)
        obj_img_paths.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
        i=1
        for i in range(1,video_mins):
            eff = 0
            idl = 0
            trav = 0
            for actionimg in obj_img_paths:
                # print(int(actionimg.split("\\")[-1].split("_")[0]))
                frame_num=int(actionimg.split("\\")[-1].split("_")[0])
                if "effective" in actionimg and (i-1)*1800<frame_num and frame_num<=i*1800:
                    eff+=1
                elif "idling" in actionimg and (i-1)*1800<frame_num and frame_num<=i*1800:
                    idl+=1
                elif "traveling" in actionimg and (i-1)*1800<frame_num and frame_num<=i*1800:
                    trav += 1
            # print("------{}MIN-------".format(i))
            print("{}-{}-{}".format(eff,idl,trav))
            # print("第{}分钟effective共计{}帧".format(i,eff))
            # print("第{}分钟idling共计{}帧".format(i,idl))
            # print("第{}分钟traveling共计{}帧".format(i,trav))






