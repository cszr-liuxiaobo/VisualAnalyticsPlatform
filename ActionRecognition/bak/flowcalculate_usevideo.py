import opyf
import cv2
import numpy as np
import os
from PIL import Image

# analyzer_img=opyf.frameSequenceAnalyzer("image_folder/make_video_full/stationary/acting_a")
# G:\Projects\Opticalflow\opyflow\image_folder\make_video_full\stationary\acting_a
# analyzer_video=opyf.videoAnalyzer("video/file/path")
# analyzer_img.extractGoodFeaturesAndDisplacements()

def optical_flow(one, two,imgsize,video_name):
    # one= fromarray(cv2.cvtColor(one, cv2.COLOR_BGR2RGB))
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((imgsize[1],imgsize[0], 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:,:,1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean=np.mean(mag)
    middle=np.median(mag)
    with open("./dataprocess/frame1/{}.txt".format(video_name.split(".")[0]),"a")as f:
        f.write(str(mean))
        f.write("\n")

    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow

def save_oldimage(image=None):
    old_image=image
    return old_image

def walk_dir(dir, fileinfo, topdown=True):
    for root, dirs, files in os.walk(dir, topdown):
        for name in files:
            fileinfo.append(os.path.join(root, name))

if __name__ == '__main__':
    curr_dir=os.getcwd()

    # paths=['coco2014','Stanford','vehicleplate']
    homedir = curr_dir+"/makevideo"
    fileinfo=[]
    walk_dir(homedir, fileinfo)
    print(fileinfo)

    for file in fileinfo:
        capture = cv2.VideoCapture(file)
        video_name=os.path.basename(file)
        print(video_name)
        # G:\Projects\ActionRecognition\Opticalflow\opyflow\image_folder\Action-video\acting\acting_a.avi
        # G:\Projects\ActionRecognition\Opticalflow\opyflow\image_folder\Action-video\idling\idling_b.avi
        # G:\Projects\ActionRecognition\Opticalflow\opyflow\image_folder\Action-video\traveling\traveling_a.avi
        i=0
        j=0
        count=1    #第28行
        frame=0

        while True:
            oldimage=save_oldimage(frame)
            success, frame = capture.read()
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            # print(size)
            # cv2.imshow('person_detection', frame)

            if not success:
                print('video is all read')
                break
            save_oldimage(image=frame)

            if (i % count == 0):
                if i > 0:
                    optical_flow(oldimage, frame,size,video_name)

                cv2.imshow('person_detection', frame)
                # 保存视频
                # 写入一帧

                key = cv2.waitKey(100)& 0xFF
                if key == ord(' '):
                    cv2.waitKey(0)
                if key == ord('q'):
                    break
                j += 1
            i += 1






