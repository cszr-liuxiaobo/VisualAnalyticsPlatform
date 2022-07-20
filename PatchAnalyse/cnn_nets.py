import torch
from torchvision import models

def vgg_net():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.vgg16(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False

    # 构建新的全连接层(此处决定分类数量)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 100),
                                           torch.nn.ReLU(),
                                           torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(100, 2))
    return model




def googlenet_net():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.googlenet(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=1024, out_features=2))
    return model

def densenet201_net():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.densenet201(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.classifier = torch.nn.Sequential(torch.nn.Linear(1920, 2))
    return model

def inception_v3_net():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.inception_v3(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=2048, out_features=2))
    return model

def alexnet_net():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.alexnet(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 2))
    return model

def mobilenet_v2_net():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.mobilenet_v2(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=False),
                                            torch.nn.Linear(1280, 2))
    return model

# -------------resnet_net_series-------------------------
def resnet_nest_50():
    from localmodels.timm.models import resnest50d
    model = resnest50d(pretrained=True)
    model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=2048, out_features=2))
    return model

def resnet_nest_50_2():
    from localmodels.timm.models import resnest50d
    model = resnest50d(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=2048, out_features=1024),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(in_features=1024, out_features=2)
    )
    return model
def resnet_nest_101():
    from localmodels.timm.models import resnest101e
    model = resnest101e(pretrained=True)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=2048, out_features=1024),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(in_features=1024, out_features=2)
    )
    return model
# # -------------------delete---------------
#
# def resnet_nest_50_backup():
#     from resnest.torch import resnest50
#     model = resnest50(pretrained=True)
#     # print(model)
#     model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=2048, out_features=2))
#     return model
#
# def resnet_nest_50_trace():
#     from resnest.torch import resnest50
#     model = resnest50(pretrained=True)
#     # print(model)
#     model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=2048, out_features=2),torch.nn.Softmax(1))
#     return model
#
# def resnet_nest_50_1_bak():
#     from resnest.torch import resnest50
#     model = resnest50(pretrained=True)
#     # print(model)
#     model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=2048, out_features=2))
#     return model
#
# def resnet_nest_50_2_bak():
#     from resnest.torch import resnest50
#     model = resnest50(pretrained=True)
#     # print(model)
#     model.fc = torch.nn.Sequential(
#         torch.nn.Linear(in_features=2048, out_features=1024),
#         torch.nn.Dropout(0.5),
#         torch.nn.ReLU(inplace=True),
#         torch.nn.Linear(in_features=1024, out_features=2)
#     )
#     return model
#
# def resnet_nest_101_bak():
#     from resnest.torch import resnest101
#     model = resnest101(pretrained=True)
#     model.fc = torch.nn.Sequential(
#         torch.nn.Linear(in_features=2048, out_features=1024),
#         torch.nn.Dropout(0.5),
#         torch.nn.ReLU(inplace=True),
#         torch.nn.Linear(in_features=1024, out_features=2)
#     )
#     # print(model)
#     return model
# # --------------------------------------------------------------
def resnet_net_50():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.resnet50(pretrained=True)
    # print(model)

    # model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 6))
    return model

def resnet_net_101():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.resnet101(pretrained=True)
    # print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 2))
    return model

def resnet_net_152():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.resnet152(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 2))
    return model

def resnet_net_32x8d():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.resnext101_32x8d(pretrained=True)
    # print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 2))
    return model

def resnet_net_50_2():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.wide_resnet50_2(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 2))
    return model

def resnet_net_101_2():
    # 带上与训练参数，是为了获得其初始权重，更多的训练展开会更准确
    model = models.wide_resnet101_2(pretrained=True)
    print(model)

    # 如果我们想只训练模型的全连接层
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # 构建新的全连接层
    model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 2))
    return model

