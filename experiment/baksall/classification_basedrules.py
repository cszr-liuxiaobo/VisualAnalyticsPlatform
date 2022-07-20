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
    plt.savefig('./confusion_matrix_{}_{}.jpg'.format(folder,exp_id), format='jpg')
    # plt.show()
# def devide_muti():
#     with open("./ratiointervals/ratio_{}.json".format(interval),"r")as f:
#         ratios=f.readlines()
#     with open("./distence/cal_centercoor_distance_{}.json".format(interval),"r")as f:
#         distance=f.readlines()
#
#     with open("../dataprocess/opticalvalueintervals/makeimages_{}.txt".format(interval),"r")as f:
#         optcal_values=f.readlines()
#
#     for i in range(len(ratios)):
#         if i >0:
#             # result=float(distance[i])*float(optcal_values[i-1])*float(ratios[i])
#             # result=float(ratios[i])*float(optcal_values[i-1])
#             result = (float(ratios[i]) - float(ratios[i-1]))
#             print(result)
#             if (os.path.exists("./ratio_changerate.json")):
#                 os.remove("./ratio_changerate.json")
#             with open("./ratio_changerate.json","a") as f:
#                 result=json.dumps(result)
#                 f.write(result)
#                 f.write("\n")
#             # result=float(optcal_values[i])
def imwriteimg(obj_img_paths,action,im_name):
    # print(obj_img_paths)
    obj_img_paths.sort(key=lambda x: int(x.split("\\")[-1].split("_")[0]))
    # print(images_name)
    frame = cv2.imread(obj_img_paths[im_name])
    cv2.imshow("img", frame)
    # cv2.imwrite("./classification/interval30/{}/{}".format(action,os.path.basename(img_paths[im_name])),frame)
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
    with open("./opticalvalue/normalopticalvalue_obj_{}.txt".format(obj_num),"r")as f:
        optical_flows=f.readlines()
    with open("./ratiorate/normalratiorate_obj_{}.txt".format(obj_num),"r")as f:
        ratio_rates=f.readlines()
    # with open("./classification/cal_centercoor_distance_30.json","r")as f:
    #     distences=f.readlines()
    for i in range(len(ratio_rates)):
        # ratio=float(ratios[i])
        # distance_d_optvalues=float(optival[i])
        optical_flow=float(optical_flows[i])
        ratio_rate=float(ratio_rates[i])
        if optical_flow>=0.4:
            print("frame_{}belong to traveling".format((i+1)*interval))
            imwriteimg(obj_img_paths,"traveling",(i+1)*interval)
        elif optical_flow<0.4 and ratio_rate>=0.05:
            print("frame_{}belong to effective".format((i+1)*interval))
            imwriteimg(obj_img_paths,"effective",(i+1)*interval)
        elif optical_flow<0.4 and ratio_rate<0.05:
            print("frame_{}belong to idling".format((i+1)*interval))
            imwriteimg(obj_img_paths,"idling",(i+1)*interval)
    # y_pred

        # else:
        #     print("frame_{}belong to traveling".format((i+1)*interval))
        #     imwriteimg(img_paths,"traveling",(i+1)*interval)

# all_img_paths=[]
all_y_true=[]
all_y_pred=[]
if __name__ == '__main__':
    interval=30
    curr_path = os.getcwd()
    base_path = os.path.dirname(curr_path)
    patch_pathes=base_path+"/dataprocessing/video_process/frames_patch_32249/NAVER1784_ch3_20190831080000_20190831090000/"
    # dirs = os.listdir(patch_pathes)
    dirs=["0","1"]
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

    calcul_confusion_matrix1(all_y_true, all_y_pred, exp_id="obj", folder="all")
