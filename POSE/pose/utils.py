import cv2
import numpy as np

colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), 
          (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
          (0, 170, 255), (0, 85, 255), (0, 0, 255), (255, 0, 255)]

skeleton = [[0, 1], [0, 4], [1, 4], [1, 2], [4, 5], [2, 3], [5, 6],
            [1, 7], [4, 10], [7, 10], [7, 8], [10, 11], [8, 9], [11, 12]]


# cocoデータセット17ポイント
# colors = [(255, 0, 127), (254, 37, 103), (251, 77, 77), (248, 115, 51),
#                (242, 149, 25), (235, 180, 0), (227, 205, 24), (217, 226, 50),
#                (206, 242, 76), (193, 251, 102), (179, 254, 128), (165, 251, 152),
#                (149, 242, 178), (132, 226, 204), (115, 205, 230), (96, 178, 255),
#                (78, 149, 255), (59, 115, 255), (39, 77, 255), (18, 37, 255), (0, 0, 255)]

# skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
#                  [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
#                  [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
#                  [3, 5], [4, 6]]


def filter_person_detections(detections):
    boxes, scores, class_ids = detections

    boxes = boxes[class_ids == 0]
    scores = scores[class_ids == 0]
    class_ids = class_ids[class_ids == 0]

    return len(scores)>0, [boxes, scores, class_ids]


def valid_point(point):
    return point[0] >= 0 and point[1] >= 0


def draw_skeletons(img, keypoints):
    output_img = img.copy()

    if type(keypoints) != list:
        return draw_skeleton(output_img, keypoints)

    for keypoint in keypoints:
        output_img = draw_skeleton(output_img, keypoint)

    return output_img


def draw_skeleton(img, keypoints):

    scale = 1/150
    thickness = min(int(img.shape[0]*scale), int(img.shape[1]*scale))

    for i, segment in enumerate(skeleton):
        point1_id, point2_id = segment

        point1 = keypoints[point1_id]
        point2 = keypoints[point2_id]
        # print(f"point1: {point1}, point2: {point2}")

        color = colors[i]

        if point1[0] == 0 and point1[1] == 0:
            continue
        if point2[0] == 0 and point2[1] == 0:
            continue

        if valid_point(point1):

            cv2.circle(img, (int(point1[0]), int(point1[1])), radius=int(thickness*1.2), color=color, thickness=-1, lineType=cv2.LINE_AA)

        if valid_point(point2):
            cv2.circle(img, (int(point2[0]), int(point2[1])), radius=int(thickness*1.2), color=color, thickness=-1, lineType=cv2.LINE_AA)

        if not valid_point(point1) or not valid_point(point2):
            continue
        img = cv2.line(img, (int(point1[0]), int(point1[1])),
                       (int(point2[0]), int(point2[1])),
                       color, thickness=thickness, lineType=cv2.LINE_AA)

    return img


def draw_heatmap(img, heatmap, mask_alpha=0.4):
    # Normalize the heatmap from 0 to 255
    min, max = np.min(heatmap), np.max(heatmap)
    heatmap_norm = np.uint8(255 * (heatmap - min) / (max - min))

    # Apply color to the heatmap
    color_heatmap = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_MAGMA)

    # Resize to match the image shape
    color_heatmap = cv2.resize(color_heatmap, (img.shape[1], img.shape[0]))

    # Fuse both images
    if mask_alpha == 0:
        combined_img = np.hstack((img, color_heatmap))
    else:
        combined_img = cv2.addWeighted(img, mask_alpha, color_heatmap, (1 - mask_alpha), 0)

    return combined_img

def resize_with_padding(image, target_width, target_height):
    """
    リサイズ&パディング

    """
    original_height, original_width = image.shape[:2]
    ratio_w = target_width / original_width
    ratio_h = target_height / original_height

    if ratio_w < ratio_h:
        new_width = target_width
        new_height = int(original_height * ratio_w)
    else:
        new_height = target_height
        new_width = int(original_width * ratio_h)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 3 and image.shape[2] == 3:
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    else:
        padded_image = np.zeros((target_height, target_width), dtype=np.uint8)

    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    
    padded_image[top:top + new_height, left:left + new_width] = resized_image

    padding_info = {
        "new_width": new_width,
        "new_height": new_height,
        "pad_top": top,
        "pad_left": left
    }
    
    return padded_image, padding_info