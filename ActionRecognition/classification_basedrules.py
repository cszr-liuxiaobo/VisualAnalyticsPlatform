import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2

from utils import listdir

def calcul_confusion_matrix1(y_true,y_pred,exp_id,folder):
    #labels表示你不同类别的代号，比如这里的demo中有13个类别
    # labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    labels_ori = ['effective','idling','traveling']
    labels=[]
    yy = y_true + y_pred
    yy.sort()
    labels_list=list(set(yy))
    for i in labels_list:
        labels.append(labels_ori[i])
    print(labels)

    tick_marks = np.array(range(len(labels))) + 0.5

    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
        font2 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 20,
                 }
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title,fontdict=font2)
        cb=plt.colorbar()
        cb.ax.tick_params(labelsize=15)
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels)
        plt.tick_params(labelsize=20)
        plt.yticks(xlocations, labels)
        plt.tick_params(labelsize=20)

        plt.ylabel('True label',font2)
        plt.xlabel('Predicted label',font2)

    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        c_normal = cm_normalized[y_val][x_val]
        if c_normal>0.6:
            plt.text(x_val, y_val, c, color='white', fontsize=20, va='bottom', ha='center')
            plt.text(x_val, y_val, "{:.0f}%".format (c_normal*100,), color='white', fontsize=20, va='top', ha='center')

        else:
            # va='top', 'bottom', 'center', 'baseline', 'center_baseline'
            # ha='center', 'right', 'left'
            plt.text(x_val, y_val, c, color='black', fontsize=20, va='bottom', ha='center')
            plt.text(x_val, y_val, "{:.0f}%".format (c_normal*100,), color='black', fontsize=20, va='top', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Confusion_Matrix_{}_{}'.format(exp_id,folder))
    # show confusion matrix
    if not os.path.exists('./{}/result'.format(video_name)):
        os.makedirs('./{}/result'.format(video_name))
    plt.savefig('./{}/result/confusion_matrix_{}_{}.jpg'.format(video_name,folder,exp_id), format='jpg')
    # plt.show()

def imwriteimg(obj_img_paths,action,im_name,dir):
    # print(obj_img_paths)
    obj_img_paths.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
    # print(images_name)
    frame = cv2.imread(obj_img_paths[im_name])
    cv2.imshow("img", frame)
    if not os.path.exists("./{}/classification/{}/{}".format(video_name,dir,action)):
        os.makedirs("./{}/classification/{}/{}".format(video_name,dir,action))
    print("os.path.basename(obj_img_paths[im_name]:",os.path.basename(obj_img_paths[im_name]))
    # cv2.imwrite("./{}/classification/{}/{}/{}".format(video_name,dir,action,os.path.basename(obj_img_paths[im_name])),frame)
    print(im_name)
    # 制作真实标签
    if os.path.basename( os.path.dirname(obj_img_paths[im_name])) == "effective":
        y_true.append(0)
    elif os.path.basename( os.path.dirname(obj_img_paths[im_name])) == "idling":
        y_true.append(1)
    elif os.path.basename(os.path.dirname(obj_img_paths[im_name])) == "traveling":
        y_true.append(2)
    # 制作预测标签
    if action == os.path.basename(os.path.dirname(obj_img_paths[im_name])):
        if action=="effective":
            y_pred.append(0)
        elif action == "idling":
            y_pred.append(1)
        elif action == "traveling":
            y_pred.append(2)
    else:
        if action=="effective":
            y_pred.append(0)
        elif action == "idling":
            y_pred.append(1)
        elif action == "traveling":
            y_pred.append(2)

def classification(obj_img_paths,obj_num):
    with open("./{}/opticalvalue/normalopticalvalue_obj_{}.txt".format(video_name,obj_num),"r")as f:
        optical_flows=f.readlines()
    with open("./{}/ratiorate/normalratiorate_obj_{}.txt".format(video_name,obj_num),"r")as f:
        ratio_rates=f.readlines()
    # with open("./classification/cal_centercoor_distance_30.json","r")as f:
    #     distences=f.readlines()
    for i in range(len(obj_img_paths)):
        # ratio=float(ratios[i])
        # distance_d_optvalues=float(optival[i])
        optical_flow=float(optical_flows[i])
        ratio_rate=float(ratio_rates[i])
        if optical_flow>=0.4:
            print("frame_{}belong to traveling".format((i)))
            imwriteimg(obj_img_paths,"traveling",(i),obj_num)
        elif optical_flow<0.4 and ratio_rate>=0.01:
            print("frame_{}belong to effective".format((i)))
            imwriteimg(obj_img_paths,"effective",(i),obj_num)
        elif optical_flow<0.4 and ratio_rate<0.01:
            print("frame_{}belong to idling".format((i)))
            imwriteimg(obj_img_paths,"idling",(i),obj_num)
    # y_pred
        # else:
        #     print("frame_{}belong to traveling".format((i+1)*interval))
        #     imwriteimg(img_paths,"traveling",(i+1)*interval)

# all_img_paths=[]
all_y_true=[]
all_y_pred=[]
if __name__ == '__main__':
    # interval=30
    curr_path = os.getcwd()
    base_path = os.path.dirname(curr_path)
    video_name="video202192_Trim"
    patch_pathes = base_path + "/dataprocessing/patches_process/frames_patch_classify/{}/".format(video_name)


    dirs = os.listdir(patch_pathes)
    # dirs=["6"]
    for dir in dirs:
        img_paths = []
        y_true = []
        y_pred = []
        predict_obj_path=os.path.join(patch_pathes, dir)
        if os.path.isdir(predict_obj_path):
            print(predict_obj_path)
            obj_img_paths=listdir(predict_obj_path, img_paths)
            classification(obj_img_paths,dir)
            print(y_true,"\n",y_pred)
            calcul_confusion_matrix1(y_true,y_pred,exp_id="obj",folder=dir)
            # all_img_paths+=img_paths
            all_y_true+=y_true
            all_y_pred+=y_pred
    print(all_y_true)
    print(all_y_pred)
    calcul_confusion_matrix1(all_y_true, all_y_pred, exp_id="obj", folder="all")
