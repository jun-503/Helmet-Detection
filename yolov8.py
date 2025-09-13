from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
