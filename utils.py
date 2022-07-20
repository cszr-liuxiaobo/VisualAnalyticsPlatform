import os
import json

def listdir(path, img_path):
    # 获取指定文件夹下图片的绝对路径
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, img_path)
            # print(file_path)
        elif os.path.splitext(file_path)[1] == '.jpg':
            # 此处获取带有图片文件的全部路径
            img_path.append(file_path)
    return  img_path

def images_onlyname(images_fullpath,images_onlynames):
    images_num=[]
    for imagefullpath in images_fullpath:
        img=os.path.basename(imagefullpath)
        images_onlynames.append(img)
        images_num.append(int(img.split("_")[0]))
    images_onlynames.sort(key=lambda x: int(x.split("_")[0]))
    images_num.sort()
    return images_onlynames,images_num

# ---------------------------------
def mean_average(json_list,block=5):
    average_optical = []
    if block!=0 and block%2==0:
        raise Exception("it has to be 0 or odd")
    if len(json_list)>int(block/2+1):
        for halfbl in range(int(block/2)):
            average_optical.append(json_list[halfbl])
        for j in range(len(json_list)):
            if j >int(block/2)-1 and j<len(json_list)-int(block/2):
                json_block=[]
                for jj in range(block):
                    json_block.append(json_list[j+int(block/2)-jj])
                print(json_block)
                average_optical.append(sum(json_block)/3)
        length=len(average_optical)
        for halfbl in range(int(block/2)):
            average_optical.append(json_list[length+halfbl])
        print("average_optical",average_optical)
    else:
        average_optical=json_list
    return average_optical


