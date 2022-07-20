import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# 绘制如下3指标每一轮的结果
from logprocess.log_analyze import get_prfa_attri_lists, get_attri_lists


def mat_plot(x,accuracy_scores,Precisions_normal,Precisions_problem,
             Recalls_normal,Recalls_problem,f1_scores_normal,f1_scores_problem,name):

    plt.figure()
    #创建子图
    plt.subplot(221)
    plt.plot(x,accuracy_scores,color='r', mfc='w', label='accuracy_scores')
    plt.plot(x,Precisions_normal,color='g', mfc='w', label='Precisions_normal')
    plt.plot(x,Precisions_problem,color='b', mfc='w', label='Precisions_problem')
    plt.legend()  # 让图例生效
    plt.ylim(0, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('{}_Precisions'.format(name))
    plt.subplot(222)
    plt.plot(x,accuracy_scores,color='r', mfc='w', label='accuracy_scores')
    plt.plot(x,Recalls_normal,color='g', mfc='w', label='Recalls_normal')
    plt.plot(x,Recalls_problem,color='b', mfc='w', label='Recalls_problem')
    plt.legend()  # 让图例生效
    plt.ylim(0, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('{}_Recalls'.format(name))
    plt.subplot(223)
    plt.plot(x,accuracy_scores,color='r', mfc='w', label='accuracy_scores')
    plt.plot(x,f1_scores_normal,color='g', mfc='w', label='f1_scores_normal')
    plt.plot(x,f1_scores_problem,color='b', mfc='w', label='f1_scores_problem')
    plt.legend()  # 让图例生效
    plt.ylim(0, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('{}_f1_scores'.format(name))
    plt.savefig('./precision-recall-f1score_{}.jpg'.format(name), dpi=900)


# def calcul_confusion_matrix(lable_normal_num,lable_problem_num,predicted_nor_nor,
#                      predicted_nor_pro,predicted_pro_pro,predicted_pro_nor):
#
#     # guess = ["normal", "problem"]
#     classes = ["normal", "problem"]
#     print(classes)
#     # r1 = [[predicted_nor_nor, predicted_pro_nor], [predicted_nor_pro, predicted_pro_pro]]
#     r1 = [[predicted_nor_nor/lable_normal_num,predicted_pro_nor/lable_problem_num],
#           [predicted_nor_pro/lable_normal_num,predicted_pro_pro/lable_problem_num]]
#
#     plt.figure(figsize=(12, 10))  # 设置plt窗口的大小
#     confusion = r1
#     print("confusion", confusion)
#     plt.imshow(confusion, cmap=plt.cm.Blues)
#     indices = range(len(confusion))
#     indices2 = range(3)
#     plt.xticks(indices, classes, rotation=40, fontsize=18)
#     plt.yticks([0.00, 1.00], classes, fontsize=18)
#     plt.ylim(1.5, -0.5)  # 设置y的纵坐标的上下限
#
#     plt.title("Confusion matrix", fontdict={'weight': 'normal', 'size': 18})
#     # 设置color bar的标签大小
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=18)
#     plt.xlabel('Predict label', fontsize=18)
#     plt.ylabel('True label', fontsize=18)
#
#     print("len(confusion)", len(confusion))
#     for first_index in range(len(confusion)):
#         for second_index in range(len(confusion[first_index])):
#
#             if confusion[first_index][second_index] > 200:
#                 color = "w"
#             else:
#                 color = "black"
#             plt.text(first_index, second_index, confusion[first_index][second_index], fontsize=18, color=color,
#                      verticalalignment='center', horizontalalignment='center', )





def calcul_confusion_matrix1(y_true,y_pred,exp_id,folder):
    #labels表示你不同类别的代号，比如这里的demo中有13个类别
    # labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    labels_ori = ['Concre','Form','Rebar','C_R','F_C','R_F']
    labels=[]
    print(y_true)
    print(y_pred)
    yy = y_true + y_pred
    yy.sort()
    labels_list=list(set(yy))
    for i in labels_list:
        labels.append(labels_ori[i])
    print(labels)
    '''
    具体解释一下re_label.txt和pr_label.txt这两个文件，比如你有100个样本
    去做预测，这100个样本中一共有10类，那么首先这100个样本的真实label你一定
    是知道的，一共有10个类别，用[0,9]表示，则re_label.txt文件中应该有100
    个数字，第n个数字代表的是第n个样本的真实label（100个样本自然就有100个
    数字）。
    同理，pr_label.txt里面也应该有1--个数字，第n个数字代表的是第n个样本经过
    你训练好的网络预测出来的预测label。
    这样，re_label.txt和pr_label.txt这两个文件分别代表了你样本的真实label和预测label，然后读到y_true和y_pred这两个变量中计算后面的混淆矩阵。当然，不一定非要使用这种txt格式的文件读入的方式，只要你最后将你的真实
    label和预测label分别保存到y_true和y_pred这两个变量中即可。
    '''
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
        if str(c_normal)=="nan":
            c_normal=0.00
            print("c_normal", c_normal)

        if c_normal>0.5:
            plt.text(x_val, y_val, c, color='white', fontsize=20, va='bottom', ha='center')
            plt.text(x_val, y_val, "{:.0f}%".format (float(c_normal)*100,), color='white', fontsize=20, va='top', ha='center')

        else:
            # va='top', 'bottom', 'center', 'baseline', 'center_baseline'
            # ha='center', 'right', 'left'
            plt.text(x_val, y_val, c, color='black', fontsize=20, va='bottom', ha='center')
            plt.text(x_val, y_val, "{:.0f}%".format (float(c_normal)*100,), color='black', fontsize=20, va='top', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Confusion_Matrix_{}_{}'.format(exp_id,folder))
    # show confusion matrix
    plt.savefig('./Result/confusion_matrix_{}_{}.jpg'.format(folder,exp_id), format='jpg')
    # plt.show()


def prfa_mat(path,ylim_min=0.5,ylim_max=1):
    """
    通过日志数据绘制出全部指标变化图,precision,recall,f1-score,accuracy
    :return:
    """
    test_Precision_normal, test_Precision_problem, \
    test_Recall_normal, test_Recall_problem, \
    test_f1_score_normal, test_f1_score_problem, \
    train_Precision_normal, train_Precision_problem, \
    train_Recall_normal, train_Recall_problem, \
    train_f1_score_normal, train_f1_score_problem=get_prfa_attri_lists(path)
    print('test_Precision_normal',test_Precision_normal)
    x=[]
    for i in range(len(test_Precision_normal)):
        x.append(i)
    print(x)
    plt.figure()
    #创建子图
    plt.subplot(321)
    plt.plot(x,test_Precision_normal,color='g', mfc='w', label='test_Precisions_normal')
    plt.plot(x,test_Precision_problem,color='r', mfc='w', label='test_Precisions_problem')
    plt.legend()  # 让图例生效
    plt.ylim(ylim_min, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('test_Precisions')
    plt.subplot(323)
    plt.plot(x,test_Recall_normal,color='g', mfc='w', label='test_Recalls_normal')
    plt.plot(x,test_Recall_problem,color='r', mfc='w', label='test_Recalls_problem')
    plt.legend()  # 让图例生效
    plt.ylim(ylim_min, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('test_Recalls')
    plt.subplot(325)
    plt.plot(x,test_f1_score_normal,color='g', mfc='w', label='test_f1_scores_normal')
    plt.plot(x,test_f1_score_problem,color='r', mfc='w', label='test_f1_scores_problem')
    plt.legend()  # 让图例生效
    plt.ylim(ylim_min, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('test_f1_scores')
    plt.savefig('./test_precision-recall-f1score.jpg', dpi=900)

    #创建子图
    plt.subplot(322)
    plt.plot(x,train_Precision_normal,color='g', mfc='w', label='train_Precisions_normal')
    plt.plot(x,train_Precision_problem,color='r', mfc='w', label='train_Precisions_problem')
    plt.legend()  # 让图例生效
    plt.ylim(ylim_min, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('train_Precisions')
    plt.subplot(324)
    plt.plot(x,train_Recall_normal,color='g', mfc='w', label='train_Recalls_normal')
    plt.plot(x,train_Recall_problem,color='r', mfc='w', label='train_Recalls_problem')
    plt.legend()  # 让图例生效
    plt.ylim(ylim_min, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('train_Recalls')
    plt.subplot(326)
    plt.plot(x,train_f1_score_normal,color='g', mfc='w', label='train_f1_scores_normal')
    plt.plot(x,train_f1_score_problem,color='r', mfc='w', label='train_f1_scores_problem')
    plt.legend()  # 让图例生效
    plt.ylim(ylim_min, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('train_f1_scores')
    plt.savefig('./train_precision-recall-f1score.jpg', dpi=900)

    plt.show()

def loss_lr_mat(path):
    """
    绘制epoch为单位的loss和LR
    :param path:
    :return:
    """
    LearningRates,testbatavglosses,TestAccuracies,TrainAccuracies=get_attri_lists(path)
    print("LearningRates",LearningRates)
    x=[]
    for i in range(len(LearningRates)):
        x.append(i)
    print(x)
    plt.figure()
    plt.plot(x,LearningRates,color='g', mfc='w', label='LearningRates')
    plt.legend()  # 让图例生效
    plt.ylim(0,0.1)
    plt.xlabel('epoch_times')
    plt.ylabel('LearningRates')

    plt.figure()
    plt.plot(x,testbatavglosses,color='g', mfc='w', label='testbatavglosses')
    plt.legend()  # 让图例生效
    plt.ylim(0, 1)
    plt.xlabel('epoch_times')
    plt.ylabel('testbatavglosses')
    plt.show()

if __name__ == '__main__':
    current_path = os.path.split(os.path.realpath(__file__))[0]
    father_path=os.path.dirname(current_path)
    print(father_path)
    log_path = os.path.join(os.path.join(father_path, 'logprocess'), 'logs') # 存放log文件的路径
    path = os.path.join(log_path,'result_experiment5.log')
    loss_lr_mat(path)
    ylim_min=0.7
    ylim_max=1
    prfa_mat(path,ylim_min,ylim_max)
