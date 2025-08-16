import argparse
import cv2
from pathlib import Path

from irpose.pose_estimator import PoseEstimator


parser = argparse.ArgumentParser(description='Settings')
parser.add_argument('--model', type=str, default='models/simple.onnx', help='onnxファイルのパス')
parser.add_argument('--thres', type=float, default=0.2, help='検出のしきい値')
parser.add_argument('--image', type=str, default='./demo/demo.jpeg', help='入力画像のパス')
parser.add_argument('--output', type=str, default='./demo/outputs/', help='出力画像のパス')
args = parser.parse_args()

# モデルの初期化
pose = PoseEstimator(args.model, conf_thres=args.thres)

# 画像の推論
np_img = cv2.imread(args.image)
total_heatmap, peaks = pose(np_img)

# 可視化
output_img = pose.draw_all(np_img)
# output_img = pose.draw_heatmap(np_img)

path = Path(args.image)
filename = path.stem
cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
cv2.imshow("Model Output", output_img)
cv2.imwrite(args.output + filename + "_heatmap.jpeg", output_img)
cv2.waitKey(0)