import os
import cv2

from utils import listdir

interval = 30
curr_path = os.getcwd()
video_name="NAVER1784_ch3_20190831080000_20190831090000"

patch_pathes = curr_path + "/frames_patch_arrange/{}/".format(video_name)
label_pathes=curr_path + "/frames_patch_classify/{}/".format(video_name)
dirs = os.listdir(patch_pathes)
# dirs = ["0"]
for dir in dirs:
    img_paths = []
    predict_obj_path = os.path.join(patch_pathes, dir)
    if os.path.isdir(predict_obj_path):
        print(predict_obj_path)
        obj_img_paths = listdir(predict_obj_path, img_paths)
        obj_img_paths.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
        print(len(obj_img_paths))
        for img_path in range(len(obj_img_paths)):
            if img_path>0 and img_path%interval==0:
                img = cv2.imread(obj_img_paths[img_path])
                label_path=label_pathes+dir+"/"+os.path.basename(obj_img_paths[img_path])
                if not os.path.exists(os.path.dirname(label_path)):
                    os.makedirs(os.path.dirname(label_path))
                if not os.path.exists(label_pathes+dir+"/effective"):
                    os.makedirs(label_pathes+dir+"/effective")
                if not os.path.exists(label_pathes+dir+"/idling"):
                    os.makedirs(label_pathes+dir+"/idling")
                if not os.path.exists(label_pathes+dir+"/traveling"):
                    os.makedirs(label_pathes+dir+"/traveling")
                print(label_path)
                cv2.imwrite(label_path,img)


