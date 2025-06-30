from ultralytics import YOLO

model = YOLO('/root/tmp/ultralytics-main/C-YES.yaml')  

model.train(
    data='/root/tmp/ultralytics-main/data.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    patience=100,
    deterministic=True,
    seed=0,
    workers=4,
    save=True,
    exist_ok=True,
    project='runs/train',
    device=0,
    verbose=True,
)
