from ultralytics import YOLO
import cv2
import numpy as np
from inference import draw_bounding_boxes,draw_bounding_boxes_w_gt
import os
val_image_dir = './data_real'
save_dir = './runs/detect/train3/real_results'
os.makedirs(save_dir, exist_ok=True)
model = YOLO('./runs/detect/train3/weights/best.pt')  # load a model

for val_image_name in os.listdir(val_image_dir):
    val_image_path = os.path.join(val_image_dir, val_image_name)
    result_image = os.path.join(save_dir, val_image_name)
    results = model(val_image_path)  # run inference
    # print(results[0])
    #draw_bounding_boxes_w_gt(val_image_path, results[0], val_label_path, result_image)
    draw_bounding_boxes(val_image_path, results[0], result_image)

