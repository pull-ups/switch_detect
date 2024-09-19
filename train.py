from ultralytics import YOLO
# ultralytics.checks()

model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
results = model.train(data="./switch_data_v1/data.yaml", epochs=100, imgsz=640, device=0)



