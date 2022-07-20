from motrackers.utils.misc import iou_xywh as iou
from motrackers.tracker import Tracker

class IOUTracker(Tracker):
    """
    Intersection over Union Tracker.

    References
    ----------
    * Implementation of this algorithm is heavily based on https://github.com/bochinski/iou-tracker

    Args:
        max_lost (int): Maximum number of consecutive frames object was not detected.
        tracker_output_format (str): Output format of the tracker.
        min_detection_confidence (float): Threshold for minimum detection confidence.
        max_detection_confidence (float): Threshold for max. detection confidence.
        iou_threshold (float): Intersection over union minimum value.
    """

    def __init__(
            self,
            max_lost=100,
            iou_threshold=0.01,
            min_detection_confidence=0.1,
            max_detection_confidence=0.9,
            tracker_output_format='mot_challenge'
    ):
        self.iou_threshold = iou_threshold
        self.max_detection_confidence = max_detection_confidence
        self.min_detection_confidence = min_detection_confidence

        super(IOUTracker, self).__init__(max_lost=max_lost, tracker_output_format=tracker_output_format)


    # 重写Tracker的update
    def update(self, bboxes, detection_scores, class_ids,updated_tracks=None):
        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)
        self.frame_count += 1

        # 此处针对前一帧的obj——ID进行处理，通过iou确定连续帧的具体的某个人拥有某个确定的ID
        track_ids = list(self.tracks.keys())
        # print("updated_tracks:",updated_tracks)
        # print("frame_count:{}---track_ids:{}".format(self.frame_count,track_ids))
        for track_id in track_ids:
            if len(detections) > 0:
                # x就是detections挨个取出来，下边的函数相当于省略了一个for循环。获得的就是上一帧的某个obj_ID的bbox和本帧所有的bbox进行一一对比
                # 找到最大的iou，赋予同一个obj_ID.一旦中断了检测上的连续，就歇菜了。所以需要进行改进。
                idx, best_match = max(enumerate(detections), key=lambda x: iou(self.tracks[track_id].bbox, x[1][0]))
                # print("idx:",idx)
                (bb, cid, scr) = best_match
                # print("self.iou_threshold",self.iou_threshold)
                if iou(self.tracks[track_id].bbox, bb) > self.iou_threshold:
                    self._update_track(track_id, self.frame_count, bb, scr, class_id=cid,
                                       iou_score=iou(self.tracks[track_id].bbox, bb))
                    updated_tracks.append(track_id)
                    # 在这里将已经被检测出与上一帧iou重合高的obj相关信息从detections中给删掉,防止循环结束后的那一步track中再添加一条字典数据。
                    del detections[idx]
            # 兼顾有无和阈值,对删除条件加以限制，主要是解决IOU对过去20帧的IOU交叉对比问题，如果有交叉那么就视作同一个image。总之20帧内不删除。
            if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

        # 整个走到这里的逻辑是：新一帧检测出的obj如果和前一帧的obj有大于阈值的iou则继承同一个obj_ID,然后将这个obj从detections中删除；
        # 没有检测到的则留下，然后在下一步添加新的obj-ID。
        for bb, cid, scr in detections:
            # next_track_id就是给每个obj-ID进行递增获取，detections到的每一个新的obj，添加上一个ID。将新检测到的对象添加到队列中。
            self._add_track(self.frame_count, bb, scr, class_id=cid)
        # 此处将tracks字典中存储的全部获取然后返回，用于展示。
        outputs = self._get_tracks(self.tracks)
        return outputs

    # 单纯获取一下tracker数据
    def get_tracker(self):
        outputs = self._get_tracks(self.tracks)
        return outputs