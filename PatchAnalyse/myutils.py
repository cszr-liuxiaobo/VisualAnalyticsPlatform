import torch
import cv2
import numpy as np

# 挂载GPU
def load_GPU(*args):
    if torch.cuda.is_available():
        if len(args)==1:
            args[0].cuda()
            return None
        inputs, labels = args[0].cuda(), args[1].cuda()
    return inputs,labels


def affinetransformation(image):
    # 获取crop区域
    result3 = image.copy()

    img = cv2.GaussianBlur(image,(3,3),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    cv2.imwrite("canny.jpg", edges)
    # 上右，下右
    # area1
    src1 = np.float32([ (713, 872), (1248, 628),(991, 1062),(1565, 761)])
    dst1 = np.float32([[0, 0],[852, 0],  [0, 434], [852, 434]])
    m1 = cv2.getPerspectiveTransform(src1, dst1)
    result1= cv2.warpPerspective(result3, m1, (852, 434))
    cv2.imshow("result1", result1)

    # area2
    src2 = np.float32([(1385, 381),(1683, 272),(1510, 528),(1856, 331)])
    dst2 = np.float32([[0, 0],[471, 0],  [0, 256], [471, 256]])
    m2 = cv2.getPerspectiveTransform(src2, dst2)
    result2= cv2.warpPerspective(result3, m2, (471, 256))
    cv2.imshow("result2", result2)
    return result1,result2