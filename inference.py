from ultralytics import YOLO
import cv2
import numpy as np
# Function to convert YOLO format to pixel coordinates
def yolo_to_bbox(img_width, img_height, x_center, y_center, width, height):
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)
    return xmin, ymin, xmax, ymax

# Function to draw bounding boxes
def draw_bounding_boxes_w_gt(image_path, bounding_boxes,gt_txt_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    # Draw predicted bounding boxes (green)
    for box in bounding_boxes:
        # Extract the box coordinates (xmin, ymin, xmax, ymax) and class ID
        xmin, ymin, xmax, ymax, confidence, class_id = box

        # Convert floating points to integers
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax) # predictions

        # Draw a rectangle on the image (in green)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Label the bounding box with the class ID and confidence
        label = f"Class {int(class_id)}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw the label background
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + label_size[0], ymin), (0, 255, 0), -1)
        # Draw the label text
        cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Read the ground truth bounding boxes from the .txt file
    with open(gt_txt_path, 'r') as file:
        lines = file.readlines()
    # Draw ground truth bounding boxes (red)
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x_center, y_center, width, height = map(float, data[1:])

        # Convert normalized YOLO coordinates to pixel coordinates
        xmin, ymin, xmax, ymax = yolo_to_bbox(img_width, img_height, x_center, y_center, width, height) # ground truth

        # Draw a rectangle on the image (in red)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Label the ground truth bounding box with the class ID
        label = f"GT Class {class_id}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw the label background
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + label_size[0], ymin), (0, 0, 255), -1)
        # Draw the label text
        cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # Save the image with bounding boxes
    cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")
# Function to draw bounding boxes
def draw_bounding_boxes(image_path, bounding_boxes, output_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Loop through the bounding boxes
    for box in bounding_boxes:
        # Extract the box coordinates (xmin, ymin, xmax, ymax) and class ID
        xmin, ymin, xmax, ymax, confidence, class_id = box

        # Convert floating points to integers
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # Draw a rectangle on the image (in BGR format, e.g., (0, 255, 0) is green)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Label the bounding box with the class ID
        label = f"Class {int(class_id)}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw the label background
        cv2.rectangle(img, (xmin, ymin - 20), (xmin + label_size[0], ymin), (0, 255, 0), -1)
        # Draw the label text
        cv2.putText(img, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save the image with bounding boxes
    cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")

model = YOLO('./runs/detect/train3/weights/best.pt')  # load a model

test_image = './switch_data_v1/images/test/images_original_3-png_f2418e6a-86aa-4bda-9750-2f1fc4d5277e_png.rf.8a115b70ff122482ac387d78d5621202.jpg'
train_image = './switch_data_v1/images/train/7_png_jpg.rf.6359c820c8baf4d1b5fe572116c9272a.jpg'
result_image = './results_test.jpg'
results = model(test_image)  # run inference
# results = model('./test.jpg')  # run inference
print(results[0])
draw_bounding_boxes(test_image, results[0], result_image)