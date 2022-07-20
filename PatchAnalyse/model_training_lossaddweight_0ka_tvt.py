import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_nets import resnet_net_50
from logprocess import log
import sklearn.metrics
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold

from image_recognition_main import predict_nofilter
# from image_recognition_main_meanfilter import predict_meanfilters

from utils.calculations import total_consumption, time_predict, time_normalization

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    writer = SummaryWriter('./Result/experiment14')
    logger = log.Logger(filename="result_experiment14")
    time_start = time.time()

    epoch_times=300
    torch_save_epoch=20
    val_percent=0.1
    # ----------------------------------完成数据准备----------------
    # 数据预处理,获取更多元的数据
    # Path = './image'
    # avg_width,avg_hight=GetAvgsize.avg_size(Path)
    # avg_hight=int(avg_hight/2)
    # avg_width=int(avg_width/2)
    # print('avg_hight',avg_hight)
    # print('avg_width',avg_width)
    avg_hight = 77
    avg_width = 95
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
    transform_test = transforms.Compose([
        transforms.Resize((avg_hight,avg_width)),  # 对图像进行随机的crop以后再resize成固定大小
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5874532, 0.49419284, 0.45221856), std=(0.15253681, 0.1479809, 0.14030135))
    ])
    # 读取数据,数据为自我设定非常灵活变更数据库就可以进行更多测试
    root = 'image'
    train_dataset = datasets.ImageFolder(root + '/train', transform)
    test_dataset = datasets.ImageFolder(root + '/test', transform_test)
    # 交叉验证法，从训练集中剥离一部分给val集
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    print('train:', n_train,'val:', n_val, 'test:', len(test_dataset))
    train_dataset, val_db = torch.utils.data.random_split(train_dataset, [n_train, n_val])

    kf = KFold(n_splits=10, random_state=2001, shuffle=True)
    for train_idx, test_idx in kf.split(train_dataset):
        print("%s %s" % (len(train_idx), len(test_idx)))
    # 导入数据,分批打乱有利于增加训练复杂度  drop_last=True,num_workers=8,  pin_memory=True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=120,  num_workers=8,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=30,  num_workers=8,shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_db,    batch_size = 30,  num_workers=8,shuffle = True)

    # ---------------------------导入不同神经网络--------------
    model=resnet_net_50()

    accuracy_train_results_all={}
    LR1 = 0.1

    if torch.cuda.device_count()>=1:
        print("Use {} GPU to train".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        model.cuda()

    # 先前训练的挂载模型参数
    # model.load_state_dict(torch.load('./pths/experiment14_cnn_epoch3.pth'))
    # 定义代价函数三维图片给处理常用CrossEntropyLoss
    entropy_loss = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), 0.04, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.25,min_lr=0.0001,patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35,50,60], gamma=0.1, last_epoch=-1)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.25, min_lr=0.0001, patience=5, verbose=True)

    def train(epoch):
        # 训练标志
        model.train()
        labels_np=[]
        predicted_np=[]
        j=0
        running_loss=0
        totalloss_consumptions=0
        for i, data in enumerate(train_loader):
            batch_time_start=time.time()
            # 获得数据和对应的标签
            inputs, labels = data
            # print("labels",labels)
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            # 梯度清0
            optimizer.zero_grad()
            # 获得模型预测结果
            out = model(inputs)
            # 获得最大值，以及最大值所在的位置
            _, predicted = torch.max(out, 1)

            labels_cpu = labels.cpu().numpy()
            predicted_cpu = predicted.cpu().numpy()
            labels_np.append(labels_cpu)
            predicted_np.append(predicted_cpu)

            # 交叉熵代价函数out(batch,C),labels(batch)
            loss = entropy_loss(out, labels)
            # 计算梯度
            loss.backward()
            # 修改权值
            optimizer.step()
            running_loss += loss.item()
            writer.add_scalar('train_loss', loss, i + epoch * len(train_loader))
            batch_time_end = time.time()
            eachloss_consumptions, _ = total_consumption(batch_time_start, batch_time_end)
            totalloss_consumptions += eachloss_consumptions

            eachtrain_time_re = eachloss_consumptions * (len(train_loader) - i - 1)
            eachtrain_time_rest = time_normalization(eachtrain_time_re)
            total_time_res = eachloss_consumptions * (len(train_loader) + len(test_loader)) * (
                epoch_times-epoch) - eachtrain_time_re
            total_time_rest = time_normalization(total_time_res)

            if i % 50 == 0:
                print("本轮训练完还需:", eachtrain_time_rest)
                print("根据batch时间预测总时间还需:", total_time_rest)
                # 100个输出一次
                print('train_loss:', loss.item())
                if time_rest_number != 0:
                    time_rest_numbers = time_rest_number - totalloss_consumptions
                    time_rest_total = time_normalization(time_rest_numbers)
                    print("根据epoch时间预测总时间还需：", time_rest_total)
            j = int(i)
        running_loss_batchavg=running_loss/int(j+1)

        # 获得训练集的全部测试结果标签以及对应的正确label。
        labels_np_sum = []
        predicted_np_sum = []
        for i in range(0, len(labels_np)):
            labels_np_sum = labels_np_sum + labels_np[i].tolist()
            predicted_np_sum = predicted_np_sum + predicted_np[i].tolist()
        accuracy_train=pth_test_scalar(labels_np_sum,predicted_np_sum,epoch,name='train')

        return running_loss_batchavg,accuracy_train

    def pth_test_scalar(labels_np_sum,predicted_np_sum,epoch,name=None):
        """
        对测试集或者训练集进行测试
        :param loader:
        :return:
        """
        accuracy_score = sklearn.metrics.accuracy_score(labels_np_sum, predicted_np_sum)
        Precision = sklearn.metrics.precision_score(labels_np_sum, predicted_np_sum, average=None)
        Recall = sklearn.metrics.recall_score(labels_np_sum, predicted_np_sum, average=None)
        f1_score = sklearn.metrics.f1_score(labels_np_sum, predicted_np_sum, average=None)
        print("Precision：",Precision)
        print("Recall:",Recall)
        print("f1_score:",f1_score)

        # writer.add_scalar('{}_Precision_Concrete'.format(name), Precision[0], epoch)
        # writer.add_scalar('{}_Recall_Concrete'.format(name), Recall[0], epoch)
        # writer.add_scalar('{}_f1_score_Concrete'.format(name), f1_score[0], epoch)
        # writer.add_scalar('{}_Precision_Formwork'.format(name), Precision[1], epoch)
        # writer.add_scalar('{}_Recall_Formwork'.format(name), Recall[1], epoch)
        # writer.add_scalar('{}_f1_score_Formwork'.format(name), f1_score[1], epoch)
        # writer.add_scalar('{}_Precision_Rebar'.format(name), Precision[2], epoch)
        # writer.add_scalar('{}_Recall_Rebar'.format(name), Recall[2], epoch)
        # writer.add_scalar('{}_f1_score_Rebar'.format(name), f1_score[2], epoch)

        # writer.add_scalar('{}_Precision_Concrete_rebar'.format(name), Precision[3], epoch)
        # writer.add_scalar('{}_Recall_Concrete_rebar'.format(name), Recall[3], epoch)
        # writer.add_scalar('{}_f1_score_Concrete_rebar'.format(name), f1_score[3], epoch)
        # writer.add_scalar('{}_Precision_Formwork_concrete'.format(name), Precision[4], epoch)
        # writer.add_scalar('{}_Recall_Formwork_concrete'.format(name), Recall[4], epoch)
        # writer.add_scalar('{}_f1_score_Formwork_concrete'.format(name), f1_score[4], epoch)
        # writer.add_scalar('{}_Precision_Rebar_formwork'.format(name), Precision[5], epoch)
        # writer.add_scalar('{}_Recall_Rebar_formwork'.format(name), Recall[5], epoch)
        # writer.add_scalar('{}_f1_score_Rebar_formwork'.format(name), f1_score[5], epoch)

        writer.add_scalar('{}_accuracy_score'.format(name), accuracy_score, epoch)

        accuracy_scores.append(accuracy_score)
        Precisions_Concrete.append(Precision[0])
        Precisions_Formwork.append(Precision[1])
        Precisions_Rebar.append(Precision[2])
        Recalls_Concrete.append(Recall[0])
        Recalls_Formwork.append(Recall[1])
        Recalls_Rebar.append(Recall[2])
        f1_scores_Concrete.append(f1_score[0])
        f1_scores_Formwork.append(f1_score[1])
        f1_scores_Rebar.append(f1_score[2])
        logger.get_logger().info('experiment14_round{}_{}_Precision:{}'.format(epoch, name,Precision))
        logger.get_logger().info('experiment14_round{}_{}_Recall:{}'.format(epoch, name,Recall))
        logger.get_logger().info('experiment14_round{}_{}_f1_score:{}'.format(epoch,name, f1_score))
        logger.get_logger().info('experiment14_round{}_{}_accuracy_score:{}'.format(epoch, name,accuracy_score))

        print('{}_accuracy_score'.format(name), accuracy_score)
        print("{}_Precision".format(name), Precision)
        print("{}_Recall".format(name), Recall)
        print("{}_f1_score".format(name), f1_score)
        return accuracy_score

    def val(epoch):
        # 测试模式标志
        model.eval()

        labels_np=[]
        predicted_np=[]
        running_loss=0
        totalloss_consumptions=0
        for i, data in enumerate(val_loader):
            # 获得数据和对应的标签
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            # 获得模型预测结果
            out = model(inputs)
            _, predicted = torch.max(out, 1)
            labels_cpu = labels.cpu().numpy()
            predicted_cpu = predicted.cpu().numpy()

            labels_np.append(labels_cpu)
            predicted_np.append(predicted_cpu)

            loss = entropy_loss(out, labels)
            running_loss += loss.item()
            writer.add_scalar('val_loss', loss, i+epoch*len(train_loader))
            if i % 10 == 0:
                # 100个输出一次
                print('val_loss:', loss.item())
                if time_rest_number != 0:
                    time_rest_numbers = time_rest_number - totalloss_consumptions
                    time_rest_total = time_normalization(time_rest_numbers)
                    print("time_rest_total", time_rest_total)

            j=int(i)
        running_loss_batchavg=running_loss/int(j+1)

        labels_np_sum = []
        predicted_np_sum = []
        for i in range(0, len(labels_np)):
            labels_np_sum = labels_np_sum + labels_np[i].tolist()
            predicted_np_sum = predicted_np_sum + predicted_np[i].tolist()

        # 测试集合作为测试样本
        accuracy_val=pth_test_scalar(labels_np_sum,predicted_np_sum,epoch,name='val')
        # 训练集合作为测试样本
        # accuracy_train=pth_val_scalar(epoch,name='train')

        return running_loss_batchavg,accuracy_val

    def test(epoch):
        # 测试模式标志
        model.eval()

        labels_np=[]
        predicted_np=[]
        running_loss=0
        for i, data in enumerate(test_loader):
            # 获得数据和对应的标签
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            # 获得模型预测结果
            out = model(inputs)
            # print("out::::::::",out)
            _, predicted = torch.max(out, 1)
            labels_cpu = labels.cpu().numpy()
            predicted_cpu = predicted.cpu().numpy()

            labels_np.append(labels_cpu)
            predicted_np.append(predicted_cpu)

            loss = entropy_loss(out, labels)
            running_loss += loss.item()
            writer.add_scalar('test_loss', loss, i+epoch*len(train_loader))
            if i % 10 == 0:
                # 100个输出一次
                print('test_loss:', loss.item())

            j=int(i)
        running_loss_batchavg=running_loss/int(j+1)

        labels_np_sum = []
        predicted_np_sum = []
        for i in range(0, len(labels_np)):
            labels_np_sum = labels_np_sum + labels_np[i].tolist()
            predicted_np_sum = predicted_np_sum + predicted_np[i].tolist()

        # 测试集合作为测试样本
        accuracy_test=pth_test_scalar(labels_np_sum,predicted_np_sum,epoch,name='test')
        # 训练集合作为测试样本
        # accuracy_train=pth_test_scalar(epoch,name='train')

        return running_loss_batchavg,accuracy_test

    # ------------------进行epoch轮的训练----------------------------------------
    accuracy_val_best = 0
    # epoch是训练的次数，每一次训练都将更新一次网络权重，适当更新就会避免欠拟合和过拟合。
    # 其本质就是一张张图喂进去罢了，管他重复不重复。
    accuracy_train_results = []
    accuracy_val_results = []

    accuracy_scores = []
    Precisions_Concrete = []
    Precisions_Formwork=[]
    Precisions_Rebar=[]

    Recalls_Concrete = []
    Recalls_Formwork = []
    Recalls_Rebar = []

    f1_scores_Concrete = []
    f1_scores_Formwork = []
    f1_scores_Rebar = []

    time_rest_number=0
    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    logger.get_logger().info("Now_time_is:{},----------------starting_a_experiment14-New-epoches---------------".format(now_time))
    for epoch in range(0,epoch_times):
        print('epoch:', epoch+1)
        time_epoch_start = time.time()

        # 训练集开始
        train_loss_batchavg, accuracy_train = train(epoch)
        # 把数据写入log
        lr=optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate', lr, epoch)
        logger.get_logger().info('experiment14_round:{}_LearningRate:{}'.format(epoch, format(lr, '.8f')))
        logger.get_logger().info('experiment14_round:{}_trainbatavgloss:{}'.format(epoch, format(train_loss_batchavg, '.6f')))

        # 对验证集数据进行整合
        val_loss_batchavg,accuracy_val = val(epoch)
        scheduler.step(val_loss_batchavg)

        logger.get_logger().info('experiment14_round:{}_valbatavgloss:{}'.format(epoch, format(val_loss_batchavg, '.6f')))

        logger.get_logger().info('experiment14_round:{}_valAccuracy:{}'.format(epoch,  format(accuracy_val, '.6f')))
        logger.get_logger().info('experiment14_round:{}_TrainAccuracy:{}'.format(epoch, format(accuracy_train, '.6f')))

        # 分别保存最优模型，指定间隔模型，最终轮模型
        if accuracy_val > accuracy_val_best:
            accuracy_val_best = accuracy_val
            # 经历过train()整个model已经训练完成，后边save(pth)只是存储这个模型
            torch.save(model.state_dict(), 'experiment14_cnn_best.pth')
        if epoch % torch_save_epoch == 0:
            torch.save(model.state_dict(), './pths/experiment14_cnn_epoch{}.pth'.format(epoch))
            _, _ = test(epoch)
            predict_nofilter(model,val_db,val_loader,exp_id="experiment14",folder="val")

        torch.save(model.state_dict(), './pths/experiment14_cnn_final.pth')

        accuracy_train_results.append(accuracy_train)
        accuracy_val_results.append(accuracy_val)
        time_epoch_end=time.time()
        time_rest,time_rest_number=time_predict(time_epoch_start,time_epoch_end,epoch,epoch_times)
        print("time_rest:",time_rest)
        logger.get_logger().info('experiment14_time_rest:{}'.format(time_rest))

        if epoch==epoch_times-1:
            predict_nofilter(model,test_dataset,test_loader,exp_id="experiment14",folder="test")
            # predict_meanfilters(model, val_dataset,val_loader,flag="0",exp_id="experiment14",folder="test")

    time_end = time.time()
    time_consuming,HMS=total_consumption(time_start,time_end)
    print('time_consuming',HMS)
    logger.get_logger().info(HMS)

