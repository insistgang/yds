#!/usr/bin/env python3
"""快速性能测试 - 获取 FPS 和延迟"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_model_path() -> Path | str:
    candidates = [
        Path("runs/train/p0_mangdaojiance/weights/best.pt"),
        Path("best.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"使用训练模型: {candidate}")
            return candidate

    print("使用默认模型: yolo11n.pt")
    return "yolo11n.pt"


def pick_test_image() -> Any:
    candidates = [
        Path("datasets/p0_yolo/images/val"),
        Path("datasets/p0_yolo/images/test"),
    ]
    for directory in candidates:
        if not directory.exists():
            continue
        images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if images:
            print(f"使用测试图片: {images[0]}")
            return str(images[0])

    import numpy as np

    print("未找到验证/测试图片，使用随机噪声测试")
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


def print_runtime_info() -> None:
    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"PyTorch环境信息读取失败: {exc}")


model_path = pick_model_path()
model = YOLO(str(model_path))
test_image = pick_test_image()
print_runtime_info()

# 预热
print("\n预热中...")
for _ in range(5):
    model.predict(source=test_image, imgsz=640, verbose=False)

# 正式测试
print("\n开始性能测试...")
num_frames = 100
times = []

for i in range(num_frames):
    start = time.perf_counter()
    results = model.predict(source=test_image, imgsz=640, verbose=False)
    end = time.perf_counter()
    times.append((end - start) * 1000)  # 转换为毫秒

# 计算统计
times_sorted = sorted(times)
n = len(times_sorted)
avg_ms = sum(times) / n
fps = 1000.0 / avg_ms
p50 = times_sorted[n // 2]
p99 = times_sorted[int(n * 0.99)]
min_ms = times_sorted[0]
max_ms = times_sorted[-1]

print("\n" + "=" * 60)
print("性能测试结果")
print("=" * 60)
print(f"测试帧数: {num_frames}")
print(f"平均延迟: {avg_ms:.2f} ms")
print(f"FPS: {fps:.1f}")
print(f"P50 延迟: {p50:.2f} ms")
print(f"P99 延迟: {p99:.2f} ms")
print(f"最小延迟: {min_ms:.2f} ms")
print(f"最大延迟: {max_ms:.2f} ms")
print("=" * 60)

# 检查是否达标
print("\nAGENTS.md 指标检查:")
print(f"  推理帧率 >= 30 FPS: {'✅ 达标' if fps >= 30 else '❌ 未达标'} ({fps:.1f} FPS)")
print(f"  单帧推理延迟 < 100 ms: {'✅ 达标' if avg_ms < 100 else '❌ 未达标'} ({avg_ms:.2f} ms)")
