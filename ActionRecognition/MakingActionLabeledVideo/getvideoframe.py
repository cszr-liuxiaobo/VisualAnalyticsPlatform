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

def getframe(video_path_):
    # 保存图片的帧率间隔
    i = 1
    # 开始读视频
    videoCapture = cv2.VideoCapture(video_path_)
    video_information(videoCapture)
    video_name=os.path.basename(video_path_)
    while True:
        print("i",i)
        success, frame = videoCapture.read()
        if not success:
            print('video is all read')
            break
        # if i>=103000 and i<=107000:
        # 保存图片
        # if not os.path.exists(video_name.split(".")[0]):
        #     os.makedirs(video_name.split(".")[0])
        savedname = '{}-'.format(video_name.split(".")[0]) + str(i) + '.jpg'
        savedpath =curr_dir+'/frames_tracking/{}'.format(video_name.split(".")[0])

        if not os.path.exists(savedpath):
            os.makedirs(savedpath)
        # cv2.imwrite(os.path.join(savedpath, savedname),frame)
        cv2.imwrite(os.path.join(savedpath, savedname),frame, [int(cv2.IMWRITE_JPEG_QUALITY),50])
        print('image of %s is saved' % (savedname))
        i += 1

    videoCapture.release()
    time.sleep(5)

if __name__ == '__main__':
    curr_dir=os.getcwd()
    # 注意修改video对应的文件夹
    video_path = r'./tracking_32249.avi'
    getframe(video_path)