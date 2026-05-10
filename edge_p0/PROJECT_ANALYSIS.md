# LinkAble Edge P0 项目分析总结

> 最后更新：2026-04-29

---

## 一、项目概述

**LinkAble** 是面向视障与行动不便人群的**边缘智能无障碍辅助通行系统**。

**核心定位：**
- 运行在 NVIDIA Jetson 边缘设备上
- 使用 USB 摄像头采集实时画面
- 通过 YOLO 检测障碍物
- 生成中文语音提示辅助通行
- 聚焦"校园最后 20 米"场景

**当前阶段：** 工程原型验证期，已完成端到端链路预演，正在进入数据采集与模型训练阶段。

---

## 二、系统架构

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐    ┌─────────────┐
│ USB 摄像头   │───>│ YOLO 检测器   │───>│ EventBuilder   │───>│ 中文模板生成  │───>│ 语音播报     │
│ /dev/video0 │    │ yolo11n.pt   │    │ 去抖+cooldown  │    │ semantics.py │    │ MiniMax TTS │
│ 640x480@30  │    │ detector.py  │    │ event_builder  │    │              │    │ audio.py    │
└─────────────┘    └──────────────┘    └────────────────┘    └──────────────┘    └─────────────┘
                                                │
                                                v
                                        ┌──────────────┐
                                        │ 云端 POST    │
                                        │ publisher.py │
                                        └──────────────┘
```

**数据流：**
1. USB 摄像头捕获帧（640x480, MJPG, 30 FPS）
2. YOLO 检测目标，映射到 P0 类别
3. EventBuilder 连续帧去抖 + cooldown 控制
4. semantics.py 生成中文提示文本
5. audio.py 播报语音（MiniMax TTS / 打印输出）
6. 可选：POST 到云端 API

---

## 三、核心模块分析

### 3.1 数据模型 (`models.py`) — 完成度 100%

三个核心 dataclass：

| 类 | 职责 |
|---|---|
| `Detection` | 单帧检测结果：label, confidence, bbox, distance, direction |
| `FrameDetections` | 一帧的所有检测结果 |
| `DetectionEvent` | 去抖后的事件：含优先级、时间戳、序列化方法 |

使用 `slots=True` 优化内存，提供 `to_dict()` 和 `to_api_dict()` 序列化。

### 3.2 检测器 (`detector.py`) — 完成度 95%

**P0 四类目标：**
```python
P0_LABELS = {
    "blind_road_occupied",  # 盲道占用
    "stairs",               # 台阶
    "ramp",                 # 坡道
    "road_obstacle",        # 路面障碍
}
```

**COCO 代理映射（当前链路预演用）：**
```python
DEFAULT_COCO_LABEL_MAP = {
    "backpack": "road_obstacle",
    "bench": "road_obstacle",
    "bicycle": "road_obstacle",
    "chair": "road_obstacle",
    "motorcycle": "road_obstacle",
    "person": "road_obstacle",
    "potted plant": "road_obstacle",
    "suitcase": "road_obstacle",
}
```

**方向推断：** 基于 bbox 水平三等分推断 left_front / front / right_front。

**缺失：** TensorRT 推理路径（明确列为待完成）。

### 3.3 事件去抖 (`event_builder.py`) — 完成度 100%

**核心逻辑：**
- 每帧取每类最强检测
- 连续帧计数（`min_consecutive_frames=2`）
- 发射冷却（`emit_cooldown_frames=5`）
- 优先级排序（stairs=100 > traffic_red=95 > road_obstacle=80 > ...）
- 未出现的标签自动重置 streak

### 3.4 中文模板 (`semantics.py`) — 完成度 100%

**六种事件的中文提示：**

| 事件 | 提示示例 |
|---|---|
| `road_obstacle` | 前方有障碍，请注意避让。 |
| `blind_road_occupied` | 前方盲道被占用，请注意绕行。 |
| `stairs` | 前方有台阶，请减速。 |
| `ramp` | 前方有坡道。 |
| `traffic_red` | 前方是红灯，请在路口前等待。 |
| `crosswalk` | 前方有人行横道，请沿横道通过。 |

**方向处理：** 已修复"前方左前方"重复问题，road_obstacle 使用独立的位置归一化逻辑。

### 3.5 音频输出 (`audio.py`) — 完成度 100%

**三种后端：**

| 后端 | 用途 |
|---|---|
| `PrintAudioOutput` | 打印文本，不调用 TTS |
| `Pyttsx3AudioOutput` | 本地 pyttsx3 引擎 |
| `MiniMaxAudioOutput` | MiniMax TTS API + 本地缓存 |

**MiniMax 特性：**
- 模型：`speech-2.8-hd`
- 默认音色：`presenter_female`
- 缓存：SHA256 哈希，`~/.cache/linkable_edge/tts_minimax/`
- 播放器：自动检测 mpg123 / aplay
- 离线模式：有缓存时无需 API key

### 3.6 主管线 (`pipeline.py`) — 完成度 100%

`EdgePipeline` 串联完整链路：
- EventBuilder 去抖
- semantics 生成中文提示
- audio 播报（带独立 cooldown）
- publisher 发布（失败不影响主流程）
- audio 失败自动重置冷却状态

### 3.7 USB 演示入口 (`usb_demo.py`) — 完成度 100%

**实时循环核心：**
```
USB frame -> YOLO -> EventBuilder -> 中文模板 -> MiniMax/print 播报 -> 可选 POST
```

**已验证参数：**
- 设备：`/dev/video0`
- 格式：MJPG 640x480
- 帧率：30 FPS
- v4l2：`exposure_dynamic_framerate=0`

**功能：**
- 自动配置 v4l2 参数
- 帧保存（可关闭）
- FPS 统计（每 30 帧）
- SafePublisher 防崩溃
- 支持 `--show` 本地预览

### 3.8 其他模块

| 模块 | 完成度 | 说明 |
|---|---|---|
| `publisher.py` | 100% | StdoutPublisher / HttpPublisher |
| `camera.py` | 100% | CSI 摄像头 GStreamer pipeline |
| `camera_check.py` | 100% | 摄像头诊断工具 |
| `demo.py` | 100% | Mock 数据演示 |
| `image_demo.py` | 100% | 单图 YOLO 演示 |

---

## 四、P0 事件类别定义

| 类别 | 英文 | 优先级 | 说明 |
|---|---|---|---|
| 台阶 | `stairs` | 100 | 最高优先级 |
| 路面障碍 | `road_obstacle` | 80 | 临时障碍物 |
| 盲道占用 | `blind_road_occupied` | 70 | 盲道被物体占用 |
| 坡道 | `ramp` | 60 | 坡道或无障碍入口 |

**P1 类别（当前不实现）：** `traffic_light`、`crosswalk`

---

## 五、测试覆盖

### 5.1 单元测试

| 测试文件 | 覆盖模块 | 用例数 |
|---|---|---|
| `test_detector.py` | normalize_label, infer_direction_from_bbox | 4 |
| `test_event_builder.py` | 连续帧去抖, cooldown | 2 |
| `test_semantics.py` | 全部 6 种事件 + 方向/距离组合 | 8 |
| `test_publisher.py` | API payload 格式 | 1 |
| `test_usb_demo.py` | FakeCapture/FakeDetector mock 全链路 | 2 |
| `test_tts_minimax.py` | 缓存命中、API 构造、成本估算 | 5 |

**总计：22 个测试用例，全部通过。**

### 5.2 端到端验证

**Jetson dry-run 结果（2026-04-28）：**
- 触发事件：`road_obstacle: 17`
- 中文提示：14x 前方有障碍 + 3x 右前方有障碍
- spoken/cooldown：10/7
- FPS：后续稳定 ~30
- WARN/ERROR：0/0

---

## 六、数据集状态

### 6.1 自采数据 (`datasets/linkable_p0_raw/`)

| 类别 | 目标 | 当前 |
|---|---|---|
| `road_obstacle` | 100-200 张 | 0 张 |
| `stairs` | 80-150 张 | 0 张 |
| `ramp` | 60-120 张 | 0 张 |
| `blind_road_occupied` | 80-150 张 | 0 张 |
| **合计** | **400-600 张** | **0 张** |

**采集工具已就绪：** `scripts/capture_p0_frames.py`

### 6.2 公开数据 (`datasets/public_p0_yolo/`)

| 数据源 | 类别 | 状态 |
|---|---|---|
| Mendeley Obstacles Avoidance | `road_obstacle` | ✅ 已转换：1627 框 |
| Zenodo Accessibility Barriers | `stairs`, `ramp` | ⏳ 未下载 |
| Roboflow Tactile Paving | 盲道上下文 | ⏳ 未下载 |

**转换工具已就绪：** `scripts/prepare_public_p0_datasets.py`

### 6.3 TTS 样本 (`tts_samples/`)

- 4 音色 × 5 事件 = 20 个 mp3
- 用于人工试听选音色

---

## 七、工具脚本

| 脚本 | 功能 | 行数 |
|---|---|---|
| `capture_p0_frames.py` | USB 摄像头离散图片采集 | 245 |
| `check_p0_dataset.py` | 数据集数量和 manifest 检查 | 111 |
| `prepare_public_p0_datasets.py` | 公开数据初始化/转换/统计 | 409 |
| `test_mendeley_road_obstacle_dataset.py` | Mendeley 数据 QA 验证 | 293 |
| `tts_minimax_test.py` | TTS 音色评测和缓存管理 | 422 |

---

## 八、文档体系

| 文档 | 内容 |
|---|---|
| `current_phase_status.md` | 5 阶段进度表、材料口径 |
| `phase4_dryrun_report.md` | dry-run 实验报告 |
| `demo_route_plan.md` | 校园最后 20 米 A-F 点位路线 |
| `demo_video_shotlist.md` | 4:30 演示视频分镜 |
| `p0_data_collection_plan.md` | P0 数据采集详细计划 |
| `annotation_guidelines_p0.md` | 标注规则说明 |
| `field_capture_checklist.md` | 现场采集 checklist |
| `public_dataset_candidates.md` | 6 个公开数据集候选 |
| `public_dataset_ingestion_plan.md` | 公开数据导入转换计划 |
| `public_mendeley_conversion_report.md` | Mendeley 转换报告 |
| `dataset_build_next_steps.md` | 数据集构建下一步 |

---

## 九、已验证结果

### 9.1 端到端链路预演

**环境：** Jetson + HP w300 USB 摄像头

**参数：**
- 模型：`yolo11n.pt`（COCO 代理类）
- 分辨率：640x480
- 帧率：30 FPS
- 音频：`--audio print`

**结果：**
```
[EVENT] road_obstacle spoken -> 前方有障碍，请注意避让。
[EVENT] road_obstacle spoken -> 右前方有障碍，请注意避让。
[STATS] fps=30.00 frames=2520 events=17 speeches=10
```

### 9.2 MiniMax TTS 缓存

- 首次请求：成功生成 mp3
- 离线播放：有缓存时无需 API key
- 缓存路径：`~/.cache/linkable_edge/tts_minimax/`

### 9.3 文案归一化

**修复前：** `前方右前方有障碍，请注意避让。`（不自然）
**修复后：** `右前方有障碍，请注意避让。`（自然、可缓存）

---

## 十、已知限制

| 限制 | 影响 | 状态 |
|---|---|---|
| 无 TensorRT 推理 | 推理性能未优化 | 待完成 |
| P0 自训练模型未完成 | 当前用 COCO 代理类 | **核心瓶颈** |
| 自采数据为零 | 无法训练自定义模型 | **核心瓶颈** |
| 云端/移动端未联调 | HttpPublisher 未验证 | 待完成 |
| audio.py 无独立测试 | 测试覆盖不完整 | 改进项 |
| pyproject.toml 无依赖声明 | 安装不便 | 改进项 |

---

## 十一、完成度评估

| 模块 | 权重 | 完成度 | 说明 |
|---|---|---|---|
| 核心代码 | 30% | 98% | 全链路完整，仅缺 TensorRT |
| 测试覆盖 | 10% | 75% | 核心逻辑覆盖好，缺独立集成测试 |
| 数据集 | 25% | 35% | 公开 road_obstacle 就绪，自采为零 |
| 脚本工具 | 10% | 100% | 采集/检查/转换/评测工具齐全 |
| 文档 | 10% | 100% | 采集计划、标注规范、演示分镜完整 |
| 演示验证 | 15% | 80% | dry-run 通过，正式路线待固定 |
| **总计** | **100%** | **~78%** | |

---

## 十二、下一步优先级

### P0 紧急

1. **开始自采数据** — 使用 `capture_p0_frames.py` 按 A-F 点位采集 400-600 张
2. **转换 Zenodo stairs/ramp 数据** — 补齐公开 cold-start
3. **训练 P0 baseline** — 合并数据训练 `yolo11n`，输出 mAP50/Recall/FPS

### P1 重要

4. **固定演示路线** — 实地走查 A-F 点位
5. **录制演示视频** — 按分镜脚本录制 4:30 视频
6. **准备提交材料** — 论文、展架、PPT

### P2 改进

7. 补充 audio.py / pipeline.py 独立单元测试
8. 完善 pyproject.toml 依赖声明
9. TensorRT 加速（如需性能数据）

---

## 十三、材料口径

**可以说：**
- LinkAble 是面向视障与行动不便人群的边缘智能无障碍辅助通行系统
- 当前完成了 Jetson + USB 摄像头 + YOLO + P0 事件管线 + 中文提示的稳定预演
- `yolo11n.pt` + COCO 代理类用于验证端到端链路
- 已有 P0 数据采集和公开数据转换工具链

**不能说：**
- 已经是成熟导盲设备
- 可以替代盲杖或导盲犬
- 已经完成完整 P0 自训练模型
- 核心是 Flutter、云端、地图或志愿者系统

---

## 十四、关键文件索引

```
edge_p0/
├── src/linkable_edge/
│   ├── models.py              # 数据模型
│   ├── detector.py            # YOLO 检测器
│   ├── event_builder.py       # 事件去抖
│   ├── semantics.py           # 中文模板
│   ├── audio.py               # 音频输出
│   ├── pipeline.py            # 主管线
│   ├── publisher.py           # 云端发布
│   ├── usb_demo.py            # 主入口
│   ├── camera.py              # CSI 摄像头
│   ├── camera_check.py        # 摄像头诊断
│   ├── demo.py                # Mock 演示
│   └── image_demo.py          # 单图演示
├── tests/                     # 22 个单元测试
├── scripts/
│   ├── capture_p0_frames.py   # 数据采集
│   ├── check_p0_dataset.py    # 数据检查
│   ├── prepare_public_p0_datasets.py  # 公开数据转换
│   ├── test_mendeley_road_obstacle_dataset.py  # QA 验证
│   └── tts_minimax_test.py    # TTS 评测
├── docs/                      # 11 份文档
├── datasets/
│   ├── linkable_p0_raw/       # 自采数据（空）
│   ├── public_p0_sources/     # 公开数据暂存
│   └── public_p0_yolo/        # 转换后 YOLO 数据
└── runs/                      # 运行产物
```
