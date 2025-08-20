import argparse
import cv2
from pathlib import Path

from YOLO.yolo.detector import PersonDetector
from POSE.pose.pose_estimator import PoseEstimator
from POSE.pose.utils import filter_person_detections


parser = argparse.ArgumentParser(description='Settings')
parser.add_argument('--model', type=str, default='models/simple.onnx', help='onnxファイルのパス')
parser.add_argument('--thres', type=float, default=0.2, help='検出のしきい値')
parser.add_argument('--image', type=str, default='./demo/demo.jpeg', help='入力画像のパス')
parser.add_argument('--output', type=str, default='./demo/outputs/', help='出力画像のパス')
args = parser.parse_args()



# モデルの初期化
obje = PersonDetector("/Users/masuryui/Workspace/IRPose/YOLO/models/yolov6s_base_bs1.onnx")
pose = PoseEstimator(args.model, conf_thres=args.thres)



# 画像の推論
np_img = cv2.imread(args.image)
detections = obje(np_img)

ret, person_detections = filter_person_detections(detections)

# 可視化
if ret:

    # Estimate the pose in the image
    total_heatmap, peaks = pose(np_img, person_detections)

    # Draw Model Output
    img = pose.draw_pose(np_img)

    # Draw detections
    # img = person_detector.draw_detections(img)

cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
cv2.imshow("Model Output", img)
cv2.imwrite("doc/img/output.jpg", img)
cv2.waitKey(0)