#!/usr/bin/env python3
"""快速性能测试 - 获取 FPS 和延迟"""

import time
from pathlib import Path
from ultralytics import YOLO

# 加载模型
model_path = Path("runs/train/p0_mangdaojiance/weights/best.pt")
if not model_path.exists():
    model_path = "yolo11n.pt"
    print(f"使用默认模型: {model_path}")
else:
    print(f"使用训练模型: {model_path}")

model = YOLO(str(model_path))

# 测试图片
test_image = "datasets/p0_yolo/images/val/000001.jpg"  # 使用验证集第一张
if not Path(test_image).exists():
    # 如果不存在，使用随机噪声
    import numpy as np
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    print("使用随机噪声测试")
else:
    print(f"使用测试图片: {test_image}")

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
