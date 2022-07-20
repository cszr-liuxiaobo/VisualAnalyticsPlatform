import os
import cv2
import numpy as np
currpath=os.getcwd()

list_xy=[]
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print((x, y))
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255,0,255), thickness = 2)
        list_xy.append((x, y))
        cv2.imshow("image", img)
        return list_xy



def getpointcoor(img,draw_rect=False):
    # 鼠标点击获取坐标
    img=cv2.imread(img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN,img)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return list_xy

if __name__ == '__main__':
    image = "./asd.jpg"
    getpointcoor(image)