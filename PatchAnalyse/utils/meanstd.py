import cv2, os, argparse
import numpy as np
from tqdm import tqdm

def get_img_list(dir, firelist, ext=None):
    newdir = dir
    if os.path.isfile(dir):  # 如果是文件
        if ext is None:
            firelist.append(dir)
        elif ext in dir[-3:]:
            firelist.append(dir)
    elif os.path.isdir(dir):  # 如果是目录
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            get_img_list(newdir, firelist, ext)

    return firelist

def main(image_path):
    imglist = get_img_list(image_path, [], 'jpg')
    imgall = []
    m_list, s_list = [], []

    for imgpath in imglist:
        # print(imgpath)
        imaname = os.path.split(imgpath)[1]  # 分离文件路径和文件名后获取文件名（包括了后缀名）
        # print(imaname)
        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))

    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print(m[0][::-1])
    print(s[0][::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    # main()
    path = '../AnalysePatch/Patches'
    main(path)