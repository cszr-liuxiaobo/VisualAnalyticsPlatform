import json
import os

import cv2

from utils import listdir

curr_dir = os.getcwd()
base_dir = os.path.dirname(os.path.dirname(curr_dir))

video_name="NAVER1784_ch3_32249"
interval=30
classify_patch_path=base_dir+"/dataprocessing/patches_process/frames_patch_classify/{}".format(video_name)
frames_patch_arrange_path=base_dir+"/dataprocessing/patches_process/frames_patch_arrange/{}".format(video_name)

def write_actions_file():
    dirs = os.listdir(classify_patch_path)
    if os.path.exists("./frames_tracking/{}_Actions.txt".format(video_name)):
        os.remove("./frames_tracking/{}_Actions.txt".format(video_name))
    for dir in dirs:
        classify_patch_paths = []
        arrange_patch_paths=[]
        # 以dir为链接，找到frames_patch_arrange对应的dir中全部patch
        frames_patch_arrange_dir_path = os.path.join(frames_patch_arrange_path, dir)
        frames_patch_arrange_dir_imgname = listdir(frames_patch_arrange_dir_path, arrange_patch_paths)
        frames_patch_arrange_dir_imgname.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
        print(frames_patch_arrange_dir_imgname)
        # P:\project_bak\VisualAnalyticsPlatform/dataprocessing/patches_process/frames_patch_classify/NAVER1784_ch3_32249\0
        #P:\project_bak\VisualAnalyticsPlatform/dataprocessing/patches_process/frames_patch_classify/NAVER1784_ch3_32249\1

        predict_obj_path = os.path.join(classify_patch_path, dir)
        if os.path.isdir(predict_obj_path):
            print(predict_obj_path)
            obj_img_paths = listdir(predict_obj_path, classify_patch_paths)
            obj_img_paths.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
            for obj_img_path in obj_img_paths:
                if "effective" in obj_img_path:
                    # print("obj_img_path:",obj_img_path)
                    # 本帧在frames_patch_arrange_path每一个dir中的前30帧都是effective标签
                    effective_patch_name=os.path.basename(obj_img_path)
                    fullpath_inarrange=os.path.join(frames_patch_arrange_dir_path,effective_patch_name)
                    index=lambda x: frames_patch_arrange_dir_imgname.index(x)
                    print("index:",index(fullpath_inarrange))
                    # 此处对每个effective patch的前30帧做标注effective处理。
                    a=range(interval)
                    for i in reversed(a):
                        with open("./frames_tracking/{}_Actions.txt".format(video_name),"a")as f:
                            f.write("effective:"+frames_patch_arrange_dir_imgname[index(fullpath_inarrange)-i])
                            f.write("\n")
                if "idling" in obj_img_path:
                    # print("obj_img_path:",obj_img_path)
                    # 本帧在frames_patch_arrange_path每一个dir中的前30帧都是effective标签
                    idling_patch_name = os.path.basename(obj_img_path)
                    fullpath_inarrange = os.path.join(frames_patch_arrange_dir_path, idling_patch_name)
                    index = lambda x: frames_patch_arrange_dir_imgname.index(x)
                    print("index:", index(fullpath_inarrange))
                    # 此处对每个effective patch的前30帧做标注effective处理。
                    a = range(interval)
                    for i in reversed(a):
                        with open("./frames_tracking/{}_Actions.txt".format(video_name), "a") as f:
                            f.write("idling:" + frames_patch_arrange_dir_imgname[index(fullpath_inarrange) - i])
                            f.write("\n")
                if "traveling" in obj_img_path:
                    # print("obj_img_path:",obj_img_path)
                    # 本帧在frames_patch_arrange_path每一个dir中的前30帧都是effective标签
                    traveling_patch_name = os.path.basename(obj_img_path)
                    fullpath_inarrange = os.path.join(frames_patch_arrange_dir_path, traveling_patch_name)
                    index = lambda x: frames_patch_arrange_dir_imgname.index(x)
                    print("index:", index(fullpath_inarrange))
                    # 此处对每个effective patch的前30帧做标注effective处理。
                    a = range(interval)
                    for i in reversed(a):
                        with open("./frames_tracking/{}_Actions.txt".format(video_name), "a") as f:
                            f.write("traveling:" + frames_patch_arrange_dir_imgname[index(fullpath_inarrange) - i])
                            f.write("\n")

# 打开中间文件并以帧为中心对结果进行排序，对每一帧中的每个目标进行动作标注。
def make_video():
    track_imgs_list = []
    track_imgs_path = "./frames_tracking/tracking_{}".format(video_name)
    trackimgs_list = listdir(track_imgs_path, track_imgs_list)
    trackimgs_list.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    # 定义制作视频的参数
    imageshape=cv2.imread(trackimgs_list[0]).shape
    size=(imageshape[1],imageshape[0])
    print(imageshape)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 30
    out=cv2.VideoWriter('./frames_tracking/tracking_groundtruth_video.py_{}.avi'.format(video_name), fourcc, fps, size)

    with open("./frames_tracking/{}_Actions.txt".format(video_name),"r") as f:
        lines=f.read().splitlines()

    lines.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
    # 对相同的帧进行统一标注
    for i in range(1,int(lines[-1].split("\\")[-1].split("_")[0])+1):
        img=cv2.imread(trackimgs_list[i-1])
        for j in lines:
            if int(j.split("\\")[-1].split("_")[0]) == i:
                coor=(j.split(":")[0],int(j.split("\\")[-1].split("c")[1].split("-")[0]),int(j.split("\\")[-1].split("c")[1].split("-")[1]))
                cv2.rectangle(img, (coor[1],coor[2]-40), (coor[1]+120, coor[2]-14),
                             (255, 255, 255), cv2.FILLED)
                cv2.putText(img,coor[0],(coor[1],coor[2]-20),cv2.FONT_HERSHEY_COMPLEX,0.8, (0, 80, 200), 1)
            else:
                continue
        # 此处打开tracking_32249对应的第i帧
        print(i)
        # cv2.imshow("image:",img)
        out.write(img)


if __name__ == '__main__':
    # write_actions_file()
    make_video()
