import xml.etree.ElementTree as ET
import pickle
import time
import os
from os import listdir, getcwd
from os.path import join
import random
curr_dir=os.getcwd()
# 创建voc的训练集和测试集
def createvoc_train_test():
    trainval_percent = 0.01
    train_percent = 1
    xmlfilepath = curr_dir+'/VOCdevkit/VOC2007/Annotations'
    txtsavepath = curr_dir+'/VOCdevkit/VOC2007/ImageSets/Main'
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open('./VOCdevkit/VOC2007/ImageSets/Main/trainval.txt', 'w')
    ftest = open('./VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'w')
    ftrain = open('./VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'w')
    fval = open('./VOCdevkit/VOC2007/ImageSets/Main/val.txt', 'w')
    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftest.write(name)
            else:
                fval.write(name)
        else:
            ftrain.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

# voc转label
sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["person"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
def voc_to_yolo():
    wd = getcwd()

    for year, image_set in sets:
        if not os.path.exists('./VOCdevkit/VOC%s/labels/'%(year)):
            os.makedirs('./VOCdevkit/VOC%s/labels/'%(year))
        image_ids = open('./VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        list_file = open('%s_%s.txt'%(year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
            convert_annotation(year, image_id)
        list_file.close()

if __name__ == '__main__':
    createvoc_train_test()
    time.sleep(3)
    voc_to_yolo()