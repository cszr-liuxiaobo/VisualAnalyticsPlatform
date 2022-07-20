import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
current_path = os.path.split(os.path.realpath(__file__))[0]
father_path=os.path.dirname(current_path)

data_transform = transforms.Compose([
    transforms.Resize((77, 77)),
    transforms.ToTensor(),
])

testdataset = datasets.ImageFolder(root=father_path+'/image', transform=data_transform)
dataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=True)
mean = torch.zeros(3)
std = torch.zeros(3)
print('Compute mean and variance for training data.')
print(len(dataloader))
print('==> Computing mean and std..')
def calculate():
    for inputs, targets in dataloader:
        for i in range(3):
            # print(inputs)
            # if torch.cuda.is_available():
            #     inputs, labels = inputs.cuda(), targets.cuda()
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(testdataset))

    std.div_(len(testdataset))
    print("mean:",mean.numpy())
    print("std",std.numpy())

if __name__ == '__main__':
    calculate()