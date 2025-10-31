import cv2
import time
from collections import deque

from YOLO.yolov7.detector import PersonDetector
from POSE.pose.pose_estimator import PoseEstimator
from POSE.pose.utils import filter_person_detections

video = input("Enter: ")

obje = PersonDetector("/Users/masuryui/Workspace/IRPose/YOLO/models/best.onnx")
pose = PoseEstimator("POSE/models/irpose.onnx", conf_thres=0.2)


video_capture_dummy = cv2.VideoCapture(video)
fps = video_capture_dummy.get(cv2.CAP_PROP_FPS)
ret, oriImg = video_capture_dummy.read()
shape_tuple = tuple(oriImg.shape[1::-1])
video_capture_dummy.release()


video_capture = cv2.VideoCapture(video)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid_out = cv2.VideoWriter("output_video.mp4", fourcc, fps, shape_tuple)

oriImg_list = []
while True:
    ret, oriImg = video_capture.read()
    if not ret:
        break
    oriImg_list.append(oriImg)
video_capture.release()

print("フレーム数:", len(oriImg_list))

# FPS計測用
start_time = time.time()
processed_frames = 0

for oriImg in oriImg_list:
    try:
        detections = obje(oriImg)
        ret, person_detections = filter_person_detections(detections)
        total_heatmap, peaks = pose(oriImg, person_detections)
        output_img = pose.draw_pose(oriImg)
        vid_out.write(output_img)
        processed_frames += 1
    except Exception as e:
        print("フレーム処理中にエラー:", e)
        continue

end_time = time.time()
vid_out.release()

elapsed_time = end_time - start_time
if processed_frames > 0:
    avg_fps = processed_frames / elapsed_time
else:
    avg_fps = 0

print(f"処理時間: {elapsed_time:.2f}秒")
print(f"処理できたフレーム数: {processed_frames}")
print(f"平均FPS: {avg_fps:.2f}")
print(">>> 終了 <<<")