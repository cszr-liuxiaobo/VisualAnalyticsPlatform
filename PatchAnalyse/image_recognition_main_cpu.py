import os
import time
import numpy as np
import torch
import shutil

from torchvision import transforms,datasets
from cnn_nets import resnet_net_50
from sklearn.metrics import confusion_matrix
import sklearn as sk
import torch.nn as nn
from logprocess import log
from utils.calculations import time_normalization
from utils.mat_plots import calcul_confusion_matrix1


class Myfolder(datasets.ImageFolder):
    # 实现对path每个图片路径的返回
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
def pre_parameters(root,folder):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model=resnet_net_50()
    if torch.cuda.device_count()>=1:
        print("Use {} GPU to train".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        model.cuda()
    exp_id="sample"

    pretrain = torch.load("./pths/experiment13_cnn_final.pth",map_location=torch.device('cpu'))
    new_state_dict = {}  # OrderedDict()
    for k, v in pretrain.items():
        if "classifier" in k:  # 最后分类层的参数是classeifer ,不需要这个模型参数
            continue
        new_state_dict[k[7:]] = v  # remove `module.`  #模型k 有module 不要
    model.load_state_dict(new_state_dict)

    # model.load_state_dict(torch.load('./pths/experiment13_cnn_final.pth'))
    model.eval()

    # label = np.array(['concrete','fromwork','rebar'])

    # 数据预处理
    # root='./AnalysePatch/patches'
    # folder="crop"

    # avg_width,avg_hight=GetAvgsize.avg_size(Path)
    # avg_hight=int(avg_hight/2)
    # avg_width=int(avg_width/2)
    avg_hight = 77
    avg_width = 95
    transform_test = transforms.Compose([
        transforms.Resize((avg_hight,avg_width)),  # 对图像进行随机的crop以后再resize成固定大小
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5874532, 0.49419284, 0.45221856), std=(0.15253681, 0.1479809, 0.14030135))
    ])


    # test_dataset = datasets.ImageFolder(root + '/{}'.format(folder), transform_test)

    test_dataset = Myfolder(root + '/{}'.format(folder), transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    return model,test_dataset,test_loader,exp_id,folder


def predict_nofilter(model,dataset,test_loader,exp_id,folder):
    #logger = log.#logger(filename="result_{}".format(exp_id))
    time_start=time.time()
    # 测试标志
    model.eval()
    correct = 0
    labels_np=[]
    predicted_np=[]
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(test_loader,0):
        # 获得数据和对应的标签
        # print(data)
        inputs, labels,paths = data
        # print("inputs====",inputs)
        # print("labels====",labels)
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        # 获得模型预测结果
        out = model(inputs)
        # print("out",out)
        # 获得最大值，以及最大值所在的位置(索引)
        _, predicted = torch.max(out, 1)
        # 测试集预测正确的数量，这些索引是一一对应的
        # .sum()是对数组中的相同元素的统计
        correct += (predicted == labels).sum()
        labels_cpu=labels.cpu().numpy()
        predicted_cpu = predicted.cpu().numpy()

        labels_np.append(labels_cpu)
        predicted_np.append(predicted_cpu)
        # if not os.path.exists("./wrongclassify/1concrete/"):
        #     os.makedirs("./wrongclassify/1concrete/")
        # if not os.path.exists("./wrongclassify/2formwork/"):
        #     os.makedirs("./wrongclassify/2formwork/")
        # if not os.path.exists("./wrongclassify/3rebar/"):
        #     os.makedirs("./wrongclassify/3rebar/")
        # if not os.path.exists("./wrongclassify/4con_bar/"):
        #     os.makedirs("./wrongclassify/4con_bar/")
        # if not os.path.exists("./wrongclassify/5form_crete/"):
        #     os.makedirs("./wrongclassify/5form_crete/")
        # if not os.path.exists("./wrongclassify/6rebar_form/"):
        #     os.makedirs("./wrongclassify/6rebar_form/")
        # # 将分对的样本提取出来
        # for entry_idx in np.where(labels_cpu!=predicted_cpu)[0]:
        #     if predicted_cpu[entry_idx]==0:
        #         shutil.copy(paths[entry_idx],"./wrongclassify/1concrete/"+os.path.basename(paths[entry_idx]))
        #     elif predicted_cpu[entry_idx]==1:
        #         shutil.copy(paths[entry_idx],"./wrongclassify/2formwork/"+os.path.basename(paths[entry_idx]))
        #     elif predicted_cpu[entry_idx]==2:
        #         shutil.copy(paths[entry_idx],"./wrongclassify/3rebar/"+os.path.basename(paths[entry_idx]))
        #     elif predicted_cpu[entry_idx]==3:
        #         shutil.copy(paths[entry_idx],"./wrongclassify/4con_bar/"+os.path.basename(paths[entry_idx]))
        #     elif predicted_cpu[entry_idx]==4:
        #         shutil.copy(paths[entry_idx],"./wrongclassify/5form_crete/"+os.path.basename(paths[entry_idx]))
        #     elif predicted_cpu[entry_idx]==5:
        #         shutil.copy(paths[entry_idx],"./wrongclassify/6rebar_form/"+os.path.basename(paths[entry_idx]))


    print("test acc: {0}".format(correct.item() / len(dataset)))

    labels_np_sum=[]
    predicted_np_sum=[]
    for i in range(0,len(labels_np)):
        labels_np_sum=labels_np_sum+labels_np[i].tolist()
        predicted_np_sum=predicted_np_sum+predicted_np[i].tolist()
    # exp_id_filtername=exp_id+"nofilter"
    calcul_confusion_matrix1(labels_np_sum, predicted_np_sum,exp_id=exp_id,folder=folder)

    # print('labels_np_sum',labels_np_sum)
    # print('predicted_np_sum',predicted_np_sum)
    print('accuracy_score',sk.metrics.accuracy_score(labels_np_sum, predicted_np_sum))
    print("Precision", sk.metrics.precision_score(labels_np_sum, predicted_np_sum, average=None))
    print("Recall", sk.metrics.recall_score(labels_np_sum, predicted_np_sum, average=None))
    print("f1_score", sk.metrics.f1_score(labels_np_sum, predicted_np_sum, average=None))

    Precision=sk.metrics.precision_score(labels_np_sum, predicted_np_sum, average=None)
    Recall=sk.metrics.recall_score(labels_np_sum, predicted_np_sum, average=None)
    f1_score=sk.metrics.f1_score(labels_np_sum, predicted_np_sum, average=None)
    accuracy_score=sk.metrics.accuracy_score(labels_np_sum, predicted_np_sum)

    #logger.get_#logger().info('{}_Precision:{}'.format(exp_id,Precision))
    #logger.get_#logger().info('{}_Recall:{}'.format(exp_id,Recall))
    #logger.get_#logger().info('{}_f1_score:{}'.format(exp_id, f1_score))
    #logger.get_#logger().info('{}_accuracy_score:{}'.format(exp_id,accuracy_score))

#小样本测试结果：做为normal正样本的，右边为problem负样本的
# Precision [0.9122807 1.       ]
# Recall [1.         0.83333333]
# f1_score [0.95412844 0.90909091]

    time_end=time.time()
    time_consum=time_end-time_start
    time_consumption=time_normalization(time_consum)
    eachimg_time_consum=time_consum/len(test_loader)
    eachimg_time_consuming=time_normalization(eachimg_time_consum)
    #logger.get_#logger().info('{}_time_consumption:{}'.format(exp_id,time_consumption))
    #logger.get_#logger().info('{}_eachimg_time_consum:{}'.format(exp_id,eachimg_time_consuming))

    return labels_np_sum, predicted_np_sum

if __name__ == '__main__':
    root = './AnalysePatch/Patches'
    folderlist = os.listdir(root)
    folders = []
    for folderi in folderlist:
        if "classification" in folderi:
            folders.append(folderi)
    labels_np_sum_all=[]
    predicted_np_sum_all=[]
    for folder in folders:
        model, test_dataset,test_loader, exp_id, folder=pre_parameters(root,folder)
        labels_np_sum, predicted_np_sum=predict_nofilter(model,test_dataset,test_loader,exp_id=exp_id,folder=folder)

        labels_np_sum_all.append(labels_np_sum)
        predicted_np_sum_all.append(predicted_np_sum)
    print("------------------------")
    print(len(sum(labels_np_sum_all,[])))

    print(sum(labels_np_sum_all,[]))
    print(len(sum(predicted_np_sum_all,[])))

    print(sum(predicted_np_sum_all,[]))
    calcul_confusion_matrix1(sum(labels_np_sum_all,[]), sum(predicted_np_sum_all,[]),exp_id="total",folder="sample")


