import cv2
from time import time
from collections import deque

from YOLO.yolo.detector import PersonDetector
from POSE.pose.pose_estimator import PoseEstimator
from POSE.pose.utils import filter_person_detections

video = input("Enter: ")

obje = PersonDetector("/Users/masuryui/Workspace/IRPose/YOLO/models/yolov6s_base_bs1.onnx")
pose = PoseEstimator("/Users/masuryui/Workspace/IRPose/POSE/models/pose_resnet.onnx", conf_thres=0.15)


video_capture_dummy = cv2.VideoCapture(video)
fps = video_capture_dummy.get(cv2.CAP_PROP_FPS)
ret, oriImg = video_capture_dummy.read()
shape_tuple = tuple(oriImg.shape[1::-1])
video_capture_dummy.release()


video_capture = cv2.VideoCapture(video)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_out = cv2.VideoWriter("output_video.mp4", fourcc, fps, shape_tuple)

proc_frame_list = []
oriImg_list = []
while True:
    try:
        ret, oriImg = video_capture.read()
        if not ret:
            break
        oriImg_list.append(oriImg)
    except :
        break
video_capture.release()

print("フレーム数:",len(oriImg_list))

count = 0
for oriImg in oriImg_list:
    detections = obje(oriImg)
    ret, person_detections = filter_person_detections(detections)
    total_heatmap, peaks = pose(oriImg, person_detections)
    output_img = pose.draw_pose(oriImg)
    vid_out.write(output_img)

print(">>> 終了 <<<")