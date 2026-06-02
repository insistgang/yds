# mangdaojianceYOLO 数据集整合指南

## 概述

本文档说明如何将购买的 mangdaojianceYOLO 数据集（16类）整合到 LinkAble 项目中，用于训练 P0 四类目标检测模型。

## 数据集信息

### mangdaojianceYOLO 数据集
- **类别数**: 16 类
- **训练图片**: 7,432 张
- **验证图片**: 928 张
- **测试图片**: 930 张
- **数据格式**: YOLO 格式

### 16 类定义
```yaml
['blind track','ashcan','car','bicycle','person','spherical_roadblock',
 'pole','fire_hydrant','stop_sign','truck','dog','motorbike',
 'warning_column','bus','tricycle','reflective_cone']
```

## 类别映射关系

### P0 四类目标
| 类别 | 说明 | 目标占比 |
|------|------|----------|
| `blind_road_occupied` | 盲道占用 | 35% |
| `stairs` | 台阶 | 20% |
| `ramp` | 坡道 | 20% |
| `road_obstacle` | 路面障碍 | 25% |

### 映射配置
| 原始类别 | 映射到 | 说明 |
|----------|--------|------|
| `blind track` | `blind_road_occupied` | 直接映射 |
| `ashcan` | `road_obstacle` | 垃圾桶是障碍物 |
| `car` | `road_obstacle` | 车辆是障碍物 |
| `bicycle` | `road_obstacle` | 自行车是障碍物 |
| `person` | `road_obstacle` | 行人可能成为障碍 |
| `spherical_roadblock` | `road_obstacle` | 球形路障 |
| `pole` | `road_obstacle` | 杆子是障碍物 |
| `fire_hydrant` | `road_obstacle` | 消防栓是障碍物 |
| `truck` | `road_obstacle` | 卡车是障碍物 |
| `dog` | `road_obstacle` | 狗是障碍物 |
| `motorbike` | `road_obstacle` | 摩托车是障碍物 |
| `warning_column` | `road_obstacle` | 警示柱是障碍物 |
| `bus` | `road_obstacle` | 公交车是障碍物 |
| `tricycle` | `road_obstacle` | 三轮车是障碍物 |
| `reflective_cone` | `road_obstacle` | 反光锥是障碍物 |
| `stop_sign` | 不映射 | P1 阶段处理 |

### 缺失类别
**注意**: mangdaojianceYOLO 数据集不包含 `stairs` 和 `ramp` 类别，需要从其他数据源补充。

## 整合步骤

### 步骤 1: 数据转换

运行数据转换脚本，将 mangdaojianceYOLO 转换为 P0 四类格式：

```bash
cd D:\000\yds\edge_p0
python scripts/convert_mangdaojiance_to_p0.py
```

**输出目录**: `edge_p0/datasets/p0_yolo/`

### 步骤 2: 补充缺失数据

需要从以下数据源补充 `stairs` 和 `ramp` 类别：

1. **Zenodo accessibility_barriers 数据集**
   - 包含 stairs 和 ramp 类别
   - 运行: `python edge_p0/scripts/prepare_public_p0_datasets.py`

2. **Roboflow stairs_detection 数据集**
   - 专门的台阶检测数据集
   - 需要手动下载并转换

3. **HuggingFace RampNet 数据集**
   - 专门的坡道检测数据集
   - 需要手动下载并转换

### 步骤 3: 训练模型

使用整合后的数据集训练 P0 四类模型：

```bash
cd D:\000\yds\edge_p0
python scripts/train_p0_model.py --mode train
```

**训练配置**:
- 模型: YOLOv11n (轻量级，适合边缘部署)
- Epochs: 150
- Batch size: 16
- 图像尺寸: 640x640
- 优化器: AdamW
- 学习率: 0.01

**输出目录**: `edge_p0/runs/train/p0_mangdaojiance/`

### 步骤 4: 评估模型

```bash
python scripts/train_p0_model.py --mode evaluate
```

**评估指标**:
- mAP50: 目标 >= 0.8
- mAP50-95: 目标 >= 0.6
- 推理速度: 目标 >= 30 FPS (Jetson Nano)

### 步骤 5: 导出模型

```bash
python scripts/train_p0_model.py --mode export
```

**支持格式**:
- TorchScript (PyTorch)
- ONNX (通用格式)
- TensorRT (Jetson 部署)
- OpenVINO (Intel 设备)

## 代码更新

### 1. detector.py 更新

新增配置选项 `model_type`，支持三种模式：

```python
config = YoloDetectorConfig(
    model_path="path/to/model.pt",
    model_type="mangdaojiance",  # 使用 mangdaojianceYOLO 类别映射
)
```

**支持的 model_type**:
- `coco`: COCO 预训练模型（默认）
- `mangdaojiance`: mangdaojianceYOLO 16类模型
- `p0_custom`: P0 四类自定义模型

### 2. 事件与语义策略

`mangdaojianceYOLO` 的原始 16 类会先映射成 P0 四类，再进入 `EventBuilder` 和 `semantics.py`。当前阶段不保留 16 类直出播报模板，避免偏离 AGENTS.md 的 P0 四类主线。

## 使用示例

### 使用转换后的 P0 模型推理

优先使用标准 P0 演示入口：

```bash
cd D:\000\yds\edge_p0
python -m linkable_edge.image_demo --model runs/train/p0_mangdaojiance/weights/best.pt --source test.jpg --audio print
python -m linkable_edge.video_demo --model runs/train/p0_mangdaojiance/weights/best.pt --source test.mp4 --audio print
```

历史 16 类直出推理脚本已归档到 `edge_p0/scripts/archive/mangdaojiance_inference_demo.py`，不作为当前主入口。

### 在代码中使用

```python
from linkable_edge.detector import YoloDetector, YoloDetectorConfig
from linkable_edge.event_builder import EventBuilder
from linkable_edge.semantics import render_event_text

# 创建检测器
config = YoloDetectorConfig(
    model_path="path/to/mangdaojiance_model.pt",
    model_type="mangdaojiance",
)
detector = YoloDetector(config)

# 推理
frame_detections = detector.predict_image("test.jpg")

# 处理事件
event_builder = EventBuilder()
events = event_builder.process_frame(frame_detections)

# 生成语音提示
for event in events:
    text = render_event_text(event)
    print(text)
```

## 文件清单

### 新增文件
1. `edge_p0/configs/label_mapping.yaml` - 类别映射配置
2. `edge_p0/configs/p0_data.yaml` - P0 四类数据集配置
3. `edge_p0/scripts/convert_mangdaojiance_to_p0.py` - 数据转换脚本
4. `edge_p0/scripts/train_p0_model.py` - 整合训练脚本
5. `edge_p0/scripts/archive/mangdaojiance_inference_demo.py` - 历史 16 类直出推理脚本（归档）
6. `edge_p0/docs/mangdaojiance_integration.md` - 本文档

### 更新文件
1. `edge_p0/src/linkable_edge/detector.py` - 新增类别映射支持
2. `edge_p0/src/linkable_edge/event_builder.py` - 只处理 P0 四类和 P1 隔离
3. `edge_p0/src/linkable_edge/semantics.py` - 只维护 P0/P1 模板化中文提示

## 下一步计划

### 短期（本周）
1. [x] 完成数据转换脚本
2. [ ] 运行数据转换
3. [ ] 补充 stairs 和 ramp 数据
4. [ ] 开始训练 P0 四类模型

### 中期（2 周内）
1. [ ] 完成模型训练和评估
2. [ ] 导出 TensorRT 模型
3. [ ] 在 Jetson Nano 上测试推理速度
4. [ ] 集成到演示流程

### 长期（1 个月内）
1. [ ] 收集更多自采数据
2. [ ] 优化模型性能
3. [ ] 完成校园最后 20 米演示
4. [ ] 准备比赛提交材料

## 常见问题

### Q: mangdaojianceYOLO 数据集不包含 stairs 和 ramp 怎么办？
A: 需要从其他数据源补充：
   - Zenodo accessibility_barriers 数据集
   - Roboflow stairs_detection 数据集
   - HuggingFace RampNet 数据集

### Q: 如何使用 mangdaojianceYOLO 的 16 类模型直接推理？
A: 设置 `model_type="mangdaojiance"`，系统会自动映射到 P0 四类。

### Q: 训练需要多长时间？
A: 使用 GPU (RTX 3060) 大约需要 2-4 小时。使用 CPU 会更慢。

### Q: 模型大小是多少？
A: YOLOv11n 模型约 5MB，适合边缘部署。

### Q: 推理速度能达到多少？
A: 在 Jetson Nano 上预计 15-25 FPS，在桌面 GPU 上可达 100+ FPS。

## 参考资料

1. [LinkAble PRD v2.2](../LinkAble_PRD_v2.2.md)
2. [AGENTS.md](../AGENTS.md)
3. [mangdaojianceYOLO 数据集说明](../../mangdaojianceYOLO/README.md)
4. [YOLOv11 官方文档](https://docs.ultralytics.com/models/yolov11/)
