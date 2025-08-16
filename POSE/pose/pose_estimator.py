import time
import cv2
import numpy as np
import onnxruntime

from POSE.pose.utils import draw_skeletons, draw_heatmap, resize_with_padding

class PoseEstimator:
    def __init__(self, path, conf_thres=0.7, search_region_ratio=0.1):
        self.conf_threshold = conf_thres
        self.search_region_ratio = search_region_ratio

        self.initialize_model(path)


    def __call__(self, image, detections=None):
        if detections is None:
            return self.update(image)
        else:
            return self.update_with_detections(image, detections)


    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path,
            providers=[
                'CUDAExecutionProvider',
                'MPSExecutionProvider',
                'CPUExecutionProvider'
            ]
        )

        self.get_input_details()
        self.get_output_details()


    def update_with_detections(self, image, detections):
        full_height, full_width = image.shape[:2]
        boxes, scores, class_ids = detections

        if len(scores) == 0:
            self.total_heatmap, self.poses = None, None
            return self.total_heatmap, self.poses
        
        poses = []
        total_heatmap = np.zeros((full_height, full_width))

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = box
            box_width, box_height = x2 - x1, y2 - y1

            # 推定範囲の拡大
            x1 = max(int(x1 - box_width * self.search_region_ratio), 0)
            x2 = min(int(x2 + box_width * self.search_region_ratio), full_width)
            y1 = max(int(y1 - box_height * self.search_region_ratio), 0)
            y2 = min(int(y2 + box_height * self.search_region_ratio), full_height)

            crop = image[y1:y2, x1:x2]
            body_heatmap, body_pose = self.update(crop)

            mask = ~( (body_pose[:,0] == 0) & (body_pose[:,1] == 0) )
            body_pose[mask] += np.array([x1, y1])
            poses.append(body_pose)

            total_heatmap[y1:y2, x1:x2] += body_heatmap

        self.total_heatmap = total_heatmap
        self.poses = poses

        return self.total_heatmap, self.poses


    def update(self, image):

        self.img_height, self.img_width = image.shape[:2]

        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        # Process output data
        self.total_heatmap, self.poses = self.process_output(outputs)

        return self.total_heatmap, self.poses


    def prepare_input(self, image):

        # input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        # input_img = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)

        input_img, padding_info = resize_with_padding(image, self.input_width, self.input_height)
        self.padding_info = padding_info
        cv2.imshow("input_img", input_img)
        cv2.waitKey(1)

        # Scale input pixel values to 0 to 1
        mean = 0.5
        std = 0.5
        input_img = ((input_img / 255.0 - mean) / std)
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs


    # def process_output(self, heatmaps):

    #     total_heatmap = cv2.resize(heatmaps.sum(axis=1)[0], (self.img_width, self.img_height))
    #     map_h, map_w = heatmaps.shape[2:]

    #     # Find the maximum value in each of the heatmaps and its location
    #     max_vals = np.array([np.max(heatmap) for heatmap in heatmaps[0, ...]])
    #     peaks = np.array([np.unravel_index(heatmap.argmax(), heatmap.shape)
    #                       for heatmap in heatmaps[0, ...]])
    #     peaks[max_vals < self.conf_threshold] = np.array([0, 0], dtype=peaks.dtype)

    #     # Scale peaks to the image size
    #     peaks = peaks[:, ::-1] * np.array([self.img_width / map_w,
    #                                       self.img_height / map_h])

    #     return total_heatmap, peaks
    
    def process_output(self, heatmaps):
        # 合計ヒートマップを作成（可視化用）
        summed_heatmap = heatmaps[0].sum(axis=0)
        info = self.padding_info

        crop_left  = int(round(info["pad_left"]  * summed_heatmap.shape[1] / self.input_width))
        crop_top   = int(round(info["pad_top"]   * summed_heatmap.shape[0] / self.input_height))
        crop_right = int(round((info["pad_left"] + info["new_width"])  * summed_heatmap.shape[1] / self.input_width))
        crop_bottom= int(round((info["pad_top"]  + info["new_height"]) * summed_heatmap.shape[0] / self.input_height))

        cropped_heatmap = summed_heatmap[crop_top:crop_bottom, crop_left:crop_right]
        total_heatmap = cv2.resize(cropped_heatmap, (self.img_width, self.img_height))

        # ピーク検出（ベクトル化）
        num_joints, map_h, map_w = heatmaps.shape[1:]
        reshaped = heatmaps[0].reshape(num_joints, -1)
        idx = reshaped.argmax(axis=1)
        max_vals = reshaped.max(axis=1)
        peaks_y, peaks_x = np.divmod(idx, map_w)
        peaks = np.stack([peaks_y, peaks_x], axis=1)

        # 座標変換
        final_peaks = self.transform_coords(peaks, info, (map_h, map_w),
                                            (self.input_width, self.input_height),
                                            (self.img_width, self.img_height))

        # 信頼度でマスク
        final_peaks[max_vals < self.conf_threshold] = 0

        return total_heatmap, final_peaks

    def transform_coords(self, peaks, info, heatmap_shape, input_size, img_size):
        map_h, map_w = heatmap_shape
        input_w, input_h = input_size
        img_w, img_h = img_size
        
        scale_x_padded = input_w / map_w
        scale_y_padded = input_h / map_h
        x_padded = peaks[:, 1] * scale_x_padded
        y_padded = peaks[:, 0] * scale_y_padded

        x_unpadded = x_padded - info["pad_left"]
        y_unpadded = y_padded - info["pad_top"]

        scale_x_final = img_w / info["new_width"]
        scale_y_final = img_h / info["new_height"]

        final_x = x_unpadded * scale_x_final
        final_y = y_unpadded * scale_y_final
        
        return np.stack([final_x, final_y], axis=1)


    def draw_pose(self, image):

        if self.poses is None:
            return image
        return draw_skeletons(image, self.poses)


    def draw_heatmap(self, image, mask_alpha=0.4):
        if self.poses is None:
            return image
        return draw_heatmap(image, self.total_heatmap, mask_alpha)


    def draw_all(self, image, mask_alpha=0.4):
        if self.poses is None:
            return image
        return self.draw_pose(self.draw_heatmap(image, mask_alpha))


    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]


    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    model_path = "/Users/masuryui/Workspace/IRpose_train/checkpoints/pose_resnet.onnx"
    model = PoseEstimator(model_path, conf_thres=0.3)

    img = input("img_path: ")
    img = cv2.imread(img)

    total_heatmap, poses = model(img)

    output_img = model.draw_all(img)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", output_img)
    cv2.waitKey(0)
