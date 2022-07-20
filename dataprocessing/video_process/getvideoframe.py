# -*- coding:utf8 -*-
import cv2
import os
import time

def video_information(videoCapture):
    # *"MJPG"
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_all = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    print("[INFO] 视频时长: {}s".format(frame_all / fps))
    frame_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("[INFO] 视频帧大小: {}s".format(frame_size))

def getframe():
    # 保存图片的帧率间隔
    count = 1
    i = 1
    j = 0
    for index, video_name in enumerate(video_list):
        video_path_ = os.path.join(video_path, video_name)
        # 开始读视频
        videoCapture = cv2.VideoCapture(video_path_)
        print("正在处理第{}个视频，总共{}个视频".format(index + 1, len(video_list)))
        video_information(videoCapture)

        while True:
            print("i",i)
            success, frame = videoCapture.read()
            # if i>=103000 and i<=107000:
            # 保存图片
            j += 1
            savedname = '{}-'.format(video_name.split(".")[0]) + str(i) + '.jpg'
            savedpath =curr_dir+'/frames_ori/{}/allframes'.format(video_name.split(".")[0])

            if not os.path.exists(savedpath):
                os.makedirs(savedpath)
            # 同时生成VOCdevkit目录结构

            # cv2.imwrite(os.path.join(savedpath, savedname),frame)
            cv2.imwrite(os.path.join(savedpath, savedname),frame, [int(cv2.IMWRITE_JPEG_QUALITY),20])
            print('image of %s is saved' % (savedname))
            i += 1
            if not success:
                print('video is all read')
                break
        videoCapture.release()
        time.sleep(5)

if __name__ == '__main__':
    curr_dir=os.getcwd()
    # 注意修改video对应的文件夹
    video_path = r'./videos2'
    video_list = os.listdir(video_path)
    getframe()