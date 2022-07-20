import datetime
import json
import os
import cv2 as cv
from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks
import numpy as np
# curr_dir=os.getcwd()

def save_oldimage(image=None):
    old_image=image
    return old_image

def get_centerpoint(tracker):
    IDpoint_dict={}
    for trk in tracker:
        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]

        xcentroid, ycentroid = int(xmin + 0.5 * width), int(ymin + 0.5 * height)
        IDpoint_dict[trk_id]=(xcentroid, ycentroid)
    return IDpoint_dict
def draw_arrow(IDpoint_dict_pre,IDpoint_dict_now,image_old,out2):
    for key,val in IDpoint_dict_pre.items():
        if key in IDpoint_dict_now:
            cv.arrowedLine(image_old,
                            pt1=IDpoint_dict_pre[key],
                            pt2=IDpoint_dict_now[key],
                            color=(0, 0, 255),
                            thickness=3,
                            line_type=cv.LINE_8,
                            shift=0,
                            tipLength=0.5)

    cv.imshow('arrow_image', image_old)
    out2.write(image_old)
    cv.waitKey(1)


class DateEnconding(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(DateEnconding, self).default(obj)


def crop_patch(image,tracker,ptchsavepath):
    for trk in tracker:

        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]
        xmax, ymax = int(xmin + width), int(ymin +height)
        # 先高后宽

        cropImg = image[ymin:ymax,xmin:xmax]
        person_ID=trk[1]
        # print("prev_tracker[0]",trk[0])
        # print("prev_tracker[1]",trk[1])
        save_dir=ptchsavepath+"/{}/".format(str(person_ID))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # print("asasasasa",str(trk[0])+"_"+str(person_ID)+ ".jpg")
        try :
            # cv.imwrite(save_dir + str(trk[0])+"_"+str(person_ID)+ ".jpg", cropImg)
            # trk[0]是frame_num
            cv.imwrite(save_dir + str(trk[0])+"_"+str(person_ID)+"c{}-{}-{}-{}c".format(xmin,ymin,width,height)+ ".jpg", cropImg)

        except Exception as e:
            # 有可能出现问题的原因是：tracker中会存储旧有的数据越100帧，这期间如果出现屏幕申缩就会导致点会出现在界外，进而导致无法切割。
            # 或者还有另一种可能，到100帧阈值后会清空对应一直没检测到的数据，然后值留下一个[]空，此时还没把键删除掉。
            # 真实原因：出现了-1...这特么搞笑呢吧。
            print("error:",e)
            continue
    # cv.putText(image, label, (bb[0], y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)


trackers_dir={}
def main(video_path, model, tracker,ptchsavepath,drawarrow=False):
    cap = cv.VideoCapture(video_path)
    updated_tracks = []
    # 保存视频
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    out = cv.VideoWriter('tracking.avi', fourcc, fps, size)
    # out2 = cv.VideoWriter('arrow_tracking.avi', fourcc, fps, size)
    updated_image=0
    image_num=0
    trackers_dir = {}

    while True:
        image_old=save_oldimage(updated_image)
        ok, image = cap.read()

        if not ok:
            print("Cannot read the video feed.")
            break
        updated_tracks=list(set(updated_tracks))
        # image = cv.resize(image, (700, 500))
        # 经过检测获得框和class_id等，这里应该可以操作，只获取关于person的信息
        abc = model.detect(image)

        # if abc==None:
        #     print("没有检测到，pass")
        #     out.write(image)
        #     continue
        # 在这里就获得了track到每个obj的ID
        bboxes, confidences, class_ids=abc

        # --------改编并保存tracker的json数据
        # prev_tracker = tracker.get_tracker()
        # print(prev_tracker)
        # trackers_dir["tracker_{}".format(image_num)]=prev_tracker
        # with open('../../dataprocessing2/trackerdata.txt', 'w') as FD:
        #     FD.write(json.dumps(dict(trackers_dir),cls=DateEnconding))

        updated_image = model.draw_bboxes(image.copy(), bboxes, confidences, class_ids)

        # IDpoint_dict_pre=get_centerpoint(prev_tracker)
        # 更新前
        # 注意：tracker的初始是0，但是从一开始输入第一张图片的识别值后，其frame计数就是1了。
        tracks = tracker.update(bboxes, confidences, class_ids,updated_tracks=updated_tracks)
        # 更新后
        # -----------根据tracker内容进行截图
        now_tracker = tracks
        print("now_tracker",now_tracker)
        crop_patch(image.copy(), now_tracker,ptchsavepath)

        # 画箭头
        # IDpoint_dict_now=get_centerpoint(now_tracker)
        # if image_num>2:
        #     if drawarrow==True:
        #         draw_arrow(IDpoint_dict_pre,IDpoint_dict_now,image_old,out2)
        #     else:
        #         pass

        updated_image = draw_tracks(updated_image, tracks)
        # 更新图片
        save_oldimage(updated_image)

        cv.imshow("image", updated_image)
        # 左上角添加帧数
        font = cv.FONT_HERSHEY_SIMPLEX
        updated_image = cv.putText(updated_image, str(image_num), (50, 50), font, 1, (255, 255, 255),
                          2)  # #添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        out.write(updated_image)

        # if image_num % 1==0:
        #     savedname = 'frame_' + str(image_num) + '.jpg'
        #     cv.imwrite("../../dataprocessing2/videoprocess/frames_patch/naverformworkb/all_patches/{}".format(savedname), updated_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        image_num+=1
    cap.release()
    cv.destroyAllWindows()

curr_dir=os.getcwd()
base_dir=os.path.dirname(os.path.dirname(curr_dir))

if __name__ == '__main__':
    ptchsavepath = base_dir + "/dataprocessing2/video_process/frames_patch/naverformworkb/all_patches/"
    which_video=base_dir+"/dataprocessing2/video_process/video_seg/naverformworkb/naverformworkb.avi"
    print(which_video)
    import argparse

    parser = argparse.ArgumentParser(
        description='Object detections in input video using YOLOv3 trained on COCO dataset.'
    )

    parser.add_argument(
        '--video', '-v', type=str, default="{}".format(which_video), help='Input video path.')

    parser.add_argument(
        '--weights', '-w', type=str,
        default="./../pretrained_models/yolo_weights/yolov3_last.weights",
        help='path to weights file of YOLOv3 (`.weights` file.)'
    )

    parser.add_argument(
        '--config', '-c', type=str,
        default="./../pretrained_models/yolo_weights/yolov3.cfg",
        help='path to config file of YOLOv3 (`.cfg` file.)'
    )

    parser.add_argument(
        '--labels', '-l', type=str,
        default="./../pretrained_models/yolo_weights/coco_names.json",
        help='path to labels file of coco dataset (`.names` file.)'
    )

    parser.add_argument(
        '--gpu', type=bool,
        default=True, help='Flag to use gpu to run the deep learning model. Default is `False`'
    )

    parser.add_argument(
        '--tracker', type=str, default='IOUTracker',
        help="Tracker used to track objects. Options include ['CentroidTracker', 'CentroidKF_Tracker', 'SORT']")

    args = parser.parse_args()

    if args.tracker == 'CentroidTracker':
        tracker = CentroidTracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'CentroidKF_Tracker':
        tracker = CentroidKF_Tracker(max_lost=0, tracker_output_format='mot_challenge')
    elif args.tracker == 'SORT':
        tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
    elif args.tracker == 'IOUTracker':
        tracker = IOUTracker(max_lost=100, iou_threshold=0.01, min_detection_confidence=0.1, max_detection_confidence=0.9,
                             tracker_output_format='mot_challenge')
    else:
        raise NotImplementedError

    model = YOLOv3(
        weights_path=args.weights,
        configfile_path=args.config,
        labels_path=args.labels,
        confidence_threshold=0.5,
        nms_threshold=0.1,
        draw_bboxes=True,
        use_gpu=args.gpu
    )

    main(args.video, model, tracker,ptchsavepath,drawarrow=False)
