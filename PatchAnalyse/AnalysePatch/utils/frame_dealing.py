import ast
import cv2
import numpy as np

def rect(img,lines_true,lines_IOU,num):
    IOU=float(lines_IOU[num])
    truth_coordinate = ast.literal_eval(lines_true[num])
    xmin=truth_coordinate[0]
    ymin=truth_coordinate[1]
    xmax=truth_coordinate[2]
    ymax=truth_coordinate[3]
    cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
    cv2.putText(img, 'Groundtruth', (xmax, ymax), 1, 2, (255, 255, 255), 2)
    cv2.putText(img, 'IOU={:.3f}'.format(IOU), (xmax, ymax+22), 1, 2, (255, 255, 255), 2)
    return img


def truth_rec(img,frame_num):
    """
    为图片加上真实人工标注的框
    """
    num = int(frame_num/5)
    with open('G:/Projects/Mask_RCNN/utils/tempfiles/coor_person1_truth.txt','r') as f:
        lines_true1 = f.readlines()
    with open('G:/Projects/Mask_RCNN/utils/tempfiles/person1_IOU.txt','r') as f:
        lines_IOU1 = f.readlines()

    with open('G:/Projects/Mask_RCNN/utils/tempfiles/coor_person2_truth.txt','r') as f:
        lines_true2 = f.readlines()
    with open('G:/Projects/Mask_RCNN/utils/tempfiles/person2_IOU.txt','r') as f:
        lines_IOU2 = f.readlines()

    img = rect(img,lines_true1,lines_IOU1,num)

    img = rect(img,lines_true2,lines_IOU2,num)

    cv2.imshow("img_rec",img)
    return img

prev_list=[]
def prevent_rect_shaking(pc_frame):
    """
    此处主要是使用光流法的角点检测部分，目的是防止工作区域的框出现shaking的情况
    原理是检测出帧与帧之间角点移动的方向和平均距离，然后将结果返回，对框的点进行调整
    """
    # Convert frame to grayscale
    prev_list.append(pc_frame)
    if len(prev_list) >= 2:
        prev = prev_list[-2]
        curr = prev_list[-1]
        if len(prev_list) >= 3:
            for li in range(len(prev_list)-2):
                prev_list[li]=0
    else:
        return [0,0,0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)
    # Convert to grayscale
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Sanity check
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Find transformation matrix   https://www.pythonheidong.com/blog/article/612065/ee01c795369bf3ae8532/
    # 当最后一个参数为fullAffine=False时，使用cv2.estimateAffinePartial2D替代
    # 当最后一个参数为fullAffine=True时，使用cv2.estimateAffine2D替代
    m,_ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  # will only work with OpenCV-3 or less

    # Extract traslation综合评估下移动的距离和角度
    dx = m[0, 2]
    dy = m[1, 2]
    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])
    # print("[dx, dy, da]",[dx, dy, da])

    return [dx, dy, da]
