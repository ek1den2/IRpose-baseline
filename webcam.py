import cv2
from time import time
from collections import deque

from YOLO.yolo.detector import PersonDetector
from POSE.pose.pose_estimator import PoseEstimator
from POSE.pose.utils import filter_person_detections

cap = cv2.VideoCapture(0)

obje = PersonDetector("/Users/masuryui/Workspace/IRPose/YOLO/models/yolov6s_base_bs1.onnx")
pose = PoseEstimator("./POSE/models/pose_resnet.onnx", conf_thres=0.3)

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
frame_times = deque(maxlen=60)
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()
    start = time()
    if not ret:
        break

    # Detect People in the image
    detections = obje(frame)
    ret, person_detections = filter_person_detections(detections)

    # Estimate the pose in the image
    total_heatmap, peaks = pose(frame, person_detections)



    if ret:
        
        # Draw Model Output
        output_img = obje.draw_detections(frame)
        output_img = pose.draw_pose(output_img)
        end = time()

        frame_times.append(end - start)
        if len(frame_times) > 0:
            avg = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg
        else:
            fps = 0.0
        cv2.putText(output_img, f"FPS:{fps:3.2f}", (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_4)
        cv2.imshow("Model Output", output_img)

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break