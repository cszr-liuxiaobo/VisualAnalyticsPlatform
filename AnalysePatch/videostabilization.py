# Import numpy and OpenCV
from operator import gt

import numpy as np
import cv2
SMOOTHING_RADIUS=1

# Sample Video VID_20160427_101151.mp4
cap = cv2.VideoCapture("./videos/persondetection1.avi")
# n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
n_frames=10
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

# 对于视频稳定，我们需要捕捉视频的两帧，估计帧之间的运动，最后校正运动。
# Read first frame
_, prev = cap.read()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
print("n_frames",n_frames)
transforms = np.zeros((n_frames - 1, 3), np.float32)
for i in range(n_frames - 2):  # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                       maxCorners=200,
                                       qualityLevel=0.01,
                                       minDistance=30,
                                       blockSize=3)
    # Read next frame
    success, curr = cap.read()
    if not success:
        break
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

    # Store transformation
    transforms[i] = [dx, dy, da]

    # Move to next frame
    prev_gray = curr_gray

    print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

# Write n_frames-1 transformed frames
for i in range(n_frames - 2):
    # Read next frame
    success, frame = cap.read()
    if not success:
        break

    cv2.imshow("Before and After", frame_out)
    cv2.waitKey(10)
    out.write(frame_out)

