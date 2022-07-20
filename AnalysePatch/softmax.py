"""
注意：由于代码在计算时没有进行对batch_size的拆包，所以batch size暂时按照一张图片输入
"""
import json
import os
import re

from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import sklearn as sk
import sklearn.metrics

from cnn_nets import resnet_net_50
# from image_recognition_main_meanfilter import predict_meanfilters

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition

def validation(image_root,one_dataset,one_loader):
    model.eval()
    soft_concrete_num = 0
    soft_framework_num = 0
    soft_rebar_num = 0
    soft_con_bar=0
    soft_form_crete=0
    soft_rebar_form=0

    videoareaname = image_root.split("\\")[-3]
    framename = image_root.split("\\")[-2]
    patchsizename = image_root.split("\\")[-1]

    correct = 0
    labels_np=[]
    predicted_np=[]
    # 加载模型，方式1
    model.load_state_dict(torch.load('../pths/experiment9_cnn_epoch95.pth'))
    # 加载模型，方式2
    # model=torch.load("./pths/pts/model35.pt")
    # 加载模型，方式3
    # pretrain = torch.load("../experiment6_cnn_best.pth")
    # new_state_dict = {}  # OrderedDict()
    # for k, v in pretrain.items():
    #     if "classifier" in k:  # 最后分类层的参数是classeifer ,不需要这个模型参数
    #         continue
    #     new_state_dict[k[7:]] = v  # remove `module.`  #模型k 有module 不要
    # model.load_state_dict(new_state_dict)


    for i, data in enumerate(one_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        out = model(inputs)

        # softmax将两个预测值归一化到概率层面
        soft = softmax(out)
        # print(soft)
        _, predicted = torch.max(out, 1)
        # 测试集预测正确的数量，这些索引是一一对应的
        # .sum()是对数组中的相同元素的统计
        correct += (predicted == labels).sum()
        labels_cpu=labels.cpu().numpy()
        predicted_cpu = predicted.cpu().numpy()

        labels_np.append(labels_cpu)
        predicted_np.append(predicted_cpu)
        print(labels_np)
        print(predicted_np)


        soft_concrete=soft[0][0]
        soft_concrete=soft_concrete.item()
        soft_framework=soft[0][1]
        soft_framework=soft_framework.item()
        soft_rebar=soft[0][2]
        soft_rebar=soft_rebar.item()
        soft_con_bar=soft[0][3]
        soft_con_bar=soft_con_bar.item()
        soft_form_crete=soft[0][4]
        soft_form_crete=soft_form_crete.item()
        soft_rebar_form=soft[0][5]
        soft_rebar_form=soft_rebar_form.item()

        soft_list=[soft_concrete,soft_framework,soft_rebar,soft_con_bar,soft_form_crete,soft_rebar_form]

    #     # 记录并计算每个种类的占比，进而计算每个种类的准确率，即被分对的样本数除以所有的样本数
        if max(soft_list)==soft_list[0]:
            soft_concrete_num+=1
        elif max(soft_list)==soft_list[1]:
            soft_framework_num+=1
        elif max(soft_list)==soft_list[2]:
            soft_rebar_num+=1
        elif max(soft_list)==soft_list[3]:
            soft_con_bar+=1
        elif max(soft_list)==soft_list[4]:
            soft_form_crete+=1
        elif max(soft_list)==soft_list[5]:
            soft_rebar_form+=1
    print("soft_concrete_num",soft_concrete_num)
    print("soft_framework_num",soft_framework_num)
    print("soft_rebar_num",soft_rebar_num)
    print("soft_con_bar_num",soft_con_bar)
    print("soft_form_crete_num",soft_form_crete)
    print("soft_rebar_form_num",soft_rebar_form)
    # print("correct:",correct)
    print("validation acc: {0}".format(correct.item() / len(validation_dataset)))


    labels_np_sum=[]
    predicted_np_sum=[]
    for i in range(0,len(labels_np)):
        labels_np_sum=labels_np_sum+labels_np[i].tolist()
        predicted_np_sum=predicted_np_sum+predicted_np[i].tolist()
    accuracy_score=sk.metrics.accuracy_score(labels_np_sum, predicted_np_sum)
    accuracy_score=float('%.2f'%accuracy_score)
    print("accuracy_score：",accuracy_score)

    name=videoareaname+"-"+framename+"-"+patchsizename+":"

    return (name,accuracy_score)



if __name__ == '__main__':
    name_accs=[]
    avg_hight= avg_width=77
    transform = transforms.Compose([
        transforms.Resize((avg_hight,avg_width)),  # 对图像进行随机的crop以后再resize成固定大小
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8,contrast=0.8, saturation=0.5, hue=0.5)],p=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomRotation(30),  # 随机旋转角度
        # //transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5874532, 0.49419284, 0.45221856), std=(0.15253681, 0.1479809, 0.14030135))
    ])
    current_path = currdir = os.getcwd()
    patch_patch=current_path+"\\patches"
    image_roots = []
    for root, dirs, files in os.walk(patch_patch):
        # print(root,dirs,files)
        dirs.sort()
        # if dirs==["1concrete","2framework","3rebar","4con_bar","5form_crete","6rebar_form"]:
        if dirs==['1concrete', '2formwork', '3rebar', '4con_bar', '5form_crete', '6rebar_form'] :
            image_roots.append(root)
    print(image_roots)

    for image_root in image_roots:
        # print(image_root)
        validation_dataset = datasets.ImageFolder(image_root, transform)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1)

        model = resnet_net_50()
        if torch.cuda.device_count() >= 1:
            print("Use {} GPU to train".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            model.cuda()
        # model = nn.DataParallel(model).cuda()

        name_acc=validation(image_root,validation_dataset,validation_loader)
        name_accs.append(name_acc)
        print(name_accs)
    # 排序
    # bubbleSort(name_accs)
    name_accs=dict(name_accs)
    print(json.dumps(name_accs, sort_keys=True, indent=4, separators=(',', ':')))

    with open("a.json","a")as f:
        f.write(json.dumps(name_accs, sort_keys=True, indent=4, separators=(',', ':')))
