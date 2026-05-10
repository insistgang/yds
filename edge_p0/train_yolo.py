from ultralytics import YOLO
import os

# Load pretrained YOLO11n model
model = YOLO('yolo11n.pt')

# Train on LinkAble P0 dataset
results = model.train(
    data='datasets/public_p0_yolo/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    project='runs/train',
    name='linkable_p0_baseline',
    patience=20,
    save=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    label_smoothing=0.1,
    nbs=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    auto_augment='randaugment',
    erasing=0.4,
    crop_fraction=1.0
)

print(f"Training complete! Best mAP: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
