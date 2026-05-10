# LinkAble Edge P0

> **基于边缘智能的无障碍辅助通行原型系统**
> 
> 对齐版本：PRD v2.2 / AGENTS.md  
> 一句话记忆：**无人驾驶让车看懂路，LinkAble 让弱势行人听懂路、让管理者看见盲区。**

---

## 项目定位

LinkAble 当前版本是一套**基于边缘智能的无障碍辅助通行原型系统**，不是成熟产品，也不是完整 App。

它借鉴自动驾驶中的环境感知与边缘推理思想，但把感知对象从机动车道迁移到人行无障碍空间，把输出从车辆控制转化为面向弱势行人（视障及行动不便人群）的语音辅助提示，并通过结构化事件记录支撑城市无障碍设施的长效治理。

**核心能力：**
- 识别前方盲道是否被占用
- 识别前方是否有台阶
- 识别前方是否有坡道
- 识别路面是否有临时障碍
- 听到简短、明确、可执行的语音提示

**系统主链路：**

```
摄像头/图片/视频输入
  -> YOLO 目标检测
  -> 事件聚合与去抖
  -> 模板化中文提示（<50字）
  -> presenter_female 本地语音播报
  -> 结构化事件记录
  -> [P1] 云端大模型长文本报告生成（Kimi/Qwen）
```

---

## P0 / P1 / P2 范围

| 优先级 | 内容 | 状态 |
|--------|------|------|
| **P0** | blind_road_occupied, stairs, ramp, road_obstacle 检测 | 代码就绪，数据待采集 |
| **P0** | 连续帧事件聚合与去抖 | 完成 |
| **P0** | 模板化中文提示（<50字） | 完成 |
| **P0** | presenter_female 本地语音播报 | 完成 |
| **P0** | 结构化事件记录 | 完成 |
| **P0** | 校园最后 20 米演示路线（A-D 点位） | 待执行 |
| **P1** | 检测结果上报（FastAPI） | 待完成 |
| **P1** | 云端大模型语义增强 | 待完成 |
| **P1** | Flutter 展示页 | 待完成 |
| **P2** | 实时大模型对话、动态路线规划、志愿者系统 | 本期不做 |

---

## 目录结构

```
edge_p0/
  src/linkable_edge/          # 核心源码
    inputs.py                 # 统一输入抽象（CSI/USB/视频/图片）
    detector.py               # YOLO 检测器
    event_builder.py          # 事件聚合与去抖
    pipeline.py               # 主链路编排
    semantics.py              # 中文语义生成
    audio.py                  # 音频底层
    audio_manager.py          # 音频管理（降级链+全局锁+预加载）
    benchmark.py              # 性能基准（FPS/延迟/JSON/CSV）
    video_demo.py             # 视频/图片序列演示
    usb_demo.py               # USB 摄像头演示
    image_demo.py             # 单张图片演示
    demo.py                   # Mock 数据演示
    publisher.py              # 结构化事件上报
    models.py                 # 数据模型
  datasets/                   # 数据集
    linkable_p0_raw/          # 自采数据（四类 + negative）
    public_p0_sources/        # 公共数据源
    public_p0_yolo/           # YOLO 格式数据集
  docs/                       # 项目文档
    annotation_guidelines_p0.md   # P0 标注规范
    demo_route_plan.md            # A-D 点位路线
    field_capture_checklist.md    # 现场采集清单
    p0_data_collection_plan.md    # 数据采集计划
  scripts/                    # 工具脚本
    capture_p0_frames.py      # 数据采集脚本
    check_p0_dataset.py       # 数据集检查
    prepare_public_p0_datasets.py  # 公共数据集转换
    tts_minimax_test.py       # MiniMax TTS 测试
  tests/                      # 单元测试
  tts_samples/                # MiniMax 音频缓存
```

---

## 快速开始

### 环境要求

- Python >= 3.10
- PyTorch + CUDA（Jetson 端）
- Ultralytics（YOLO）
- OpenCV（视频/图片输入）
- requests（MiniMax TTS API）

### 安装依赖

```bash
pip install ultralytics opencv-python requests pyttsx3
```

### 1. Mock 演示（无摄像头，验证链路）

```bash
# Windows
$env:PYTHONPATH="D:\000\yds\edge_p0\src"
python -m linkable_edge.demo --audio print

# Linux / Jetson
export PYTHONPATH=/path/to/edge_p0/src
python3 -m linkable_edge.demo --audio print
```

### 2. 单张图片演示

```bash
python -m linkable_edge.image_demo \
  --model yolo11n.pt \
  --source path/to/image.jpg \
  --audio print
```

使用 COCO 代理验证路障检测：

```bash
python -m linkable_edge.image_demo \
  --model yolo11n.pt \
  --source path/to/bicycle.jpg \
  --label-map bicycle=blind_road_occupied \
  --audio print
```

### 3. USB 摄像头实时演示

```bash
# Jetson（推荐）
python3 -m linkable_edge.usb_demo \
  --device-idx 0 \
  --model yolo11n.pt \
  --audio minimax \
  --voice presenter_female \
  --imgsz 640

# 离线模式（不调用云端 TTS）
python3 -m linkable_edge.usb_demo \
  --device-idx 0 \
  --model yolo11n.pt \
  --offline \
  --imgsz 640
```

### 4. 视频文件/图片序列演示

```bash
# 视频文件
python -m linkable_edge.video_demo \
  --source video.mp4 \
  --model yolo11n.pt \
  --audio print \
  --benchmark

# 图片序列
python -m linkable_edge.video_demo \
  --source-dir frames/ \
  --model yolo11n.pt \
  --audio print
```

### 5. 性能基准测试

```bash
python -m linkable_edge.video_demo \
  --source video.mp4 \
  --benchmark \
  --output-dir ./runs/benchmark
```

输出 `benchmark.json` 和 `benchmark.csv`，包含 preprocess/inference/postprocess 耗时、FPS、p50/p99。

---

## AGENTS.md 硬约束合规

| 规范 | 状态 | 说明 |
|------|------|------|
| P0 四类检测 | 已定义 | blind_road_occupied, stairs, ramp, road_obstacle |
| P1 隔离 | 已隔离 | traffic_red/crosswalk 不可触发 |
| COCO 冷启动诚实表述 | 已注释 | stairs/ramp/blind_road 需自定义模型训练 |
| 输入 fallback | 已实现 | CSI→USB→视频→图片（`auto_select_source()`） |
| 事件聚合去抖 | 已实现 | 连续帧检测 + 发射冷却 |
| 低信息密度 | 已实现 | 一帧只播最高优先级事件 |
| 中文提示 | 已实现 | <50字，明确，可执行，无不确定词 |
| 音频降级链 | 已实现 | MiniMax→pyttsx3→print |
| 全局音频锁 | 已实现 | `threading.Lock()` |
| 启动预加载 | 已实现 | `preload(ALL_TEMPLATE_TEXTS)` |
| 默认音色 | 已设置 | presenter_female |
| 默认模型 | 已设置 | speech-2.8-hd |
| 缓存目录 | 已设置 | `~/.cache/linkable_edge/tts_minimax/` |
| 隐私保护 | 已实现 | 默认不存原始帧（`save_interval_sec=0.0`） |
| 离线模式 | 已实现 | `--offline` 禁用云端依赖 |
| 性能基准 | 已实现 | JSON/CSV 输出 |
| 结构化事件 | 已实现 | 不上传原始图像 |

---

## 数据采集（Phase 3 当前）

### 目标分布（AGENTS.md）

| 类别 | 目标数量 | 当前数量 |
|------|---------|---------|
| blind_road_occupied | ~210 (35%) | 0 |
| stairs | ~120 (20%) | 0 |
| ramp | ~120 (20%) | 0 |
| road_obstacle | ~150 (25%) | 0 |
| **合计** | **>=600** | **0** |

### 本周底线任务

```bash
# 执行数据采集
python scripts/capture_p0_frames.py \
  --category blind_road_occupied \
  --location "A点位" \
  --output datasets/linkable_p0_raw/blind_road_occupied/images
```

即使每类只有 20 张，也要拍，证明数据采集已启动。

---

## 演示场景（校园最后 20 米）

| 点位 | 场景 | 语音提示 |
|------|------|----------|
| A | 盲道占用 | "前方盲道被占用，请注意绕行" |
| B | 台阶 | "前方有台阶，请减速" |
| C | 坡道 | "前方有坡道，位于右侧" |
| D | 路面障碍 | "前方有障碍，请注意避让" |

---

## 提交材料清单

| 材料 | 格式 | 要求 |
|------|------|------|
| 技术论文 | PDF + DOCX | 6000-8000 字，匿名 |
| 演示视频 | MP4/AVI/WMV | <=5 分钟，1920x1080，<=100M |
| 展示照片 | JPG | 5 张，每张 <=2M |
| 门型展架 | JPG | 80x180cm，<=30M |
| 答辩 PPT | PPTX | 选题/原理/方案/成果/价值 |
| 作品提交 | 百度网盘 | 命名"参赛学校-作品名称" |

---

## 参考文档

- [AGENTS.md](../agents.md) — 项目协作规范（PRD v2.2 执行版）
- [LinkAble_PRD_v2.2.md](../LinkAble_PRD_v2.2.md) — 产品需求文档
- [docs/plans/linkable-competition-master-checklist.md](docs/plans/linkable-competition-master-checklist.md) — 比赛总清单
- [docs/plans/jetson-troubleshooting.md](docs/plans/jetson-troubleshooting.md) — Jetson 故障排查
- [docs/annotation_guidelines_p0.md](docs/annotation_guidelines_p0.md) — P0 标注规范

---

## 项目状态

### 我们已经做了什么

**Phase 1-2 已完成（代码层 100%）：**
- [x] Jetson 环境搭建（USB 摄像头验证通过，MJPG 640x480@30FPS）
- [x] YOLO 推理链路（单图/视频/摄像头三种输入）
- [x] P0 四类检测定义（blind_road_occupied/stairs/ramp/road_obstacle）
- [x] COCO 冷启动代理（诚实标注：仅 road_obstacle 可用，其余需自定义训练）
- [x] 事件聚合与去抖（连续帧检测 + 发射冷却 + 优先级排序）
- [x] 模板化中文提示（28 条模板，<50 字，明确可执行）
- [x] 音频系统（presenter_female + speech-2.8-hd + 缓存 + 降级链 + 全局锁）
- [x] 输入 fallback（CSI→USB→视频→图片，auto_select_source）
- [x] 隐私保护（默认不存原始帧，--offline 模式）
- [x] 性能基准（benchmark.py，JSON/CSV 输出）
- [x] 结构化事件记录（不上传原始图像）
- [x] 代码 Review 三次（10 个对齐问题全部修复）

**文档与工具：**
- [x] AGENTS.md 硬约束合规（17/17 项）
- [x] 比赛总清单、Jetson 故障排查、相机诊断记录
- [x] 系统架构图（ARCHITECTURE.txt）
- [x] 项目文件说明（PROJECT_FILES.txt）
- [x] 数据采集脚本、数据集检查脚本、TTS 测试脚本

---

### 还有哪些事情要做（当前阻塞）

**数据层（致命瓶颈）：**
- [ ] 自采真实照片：目标 >=600 张，当前 0 张
- [ ] AI 生成图分类：342 张在 pic/ 目录，需人工分类到 4 个类别
- [ ] 数据标注：分类后的图片需用 labelImg 画框标注
- [ ] 数据集划分：train/val/test = 70/20/10，测试集需含 >=20% 真实照片

**代码收尾（轻度债务）：**
- [ ] AudioManager 接入 demo（当前仍用旧 build_audio_output()）
- [ ] 删除 camera.py 死代码（33 行未引用）
- [ ] A-D 点位演示脚本（创建 demo_route_campus.py 或 route config）

**演示准备：**
- [ ] 校园 A-D 点位实地走查（固定路线、确认场景可触发）
- [ ] 演示道具准备（三脚架、充电宝、备用线材）
- [ ] 演示视频录制（<=5 分钟，1920x1080，匿名）

**提交材料：**
- [ ] 技术论文（6000-8000 字）
- [ ] 答辩 PPT（15-20 页）
- [ ] 门型展架（80x180cm）
- [ ] 展示照片（5 张，<=2M）

---

### 后续执行计划

**本周（5.10-5.11）—— 数据破冰：**
1. 人工分类 342 张 AI 生成图到 4 个类别目录
2. 拿手机/相机拍摄真实场景：每类 >=20 张（底线 100 张）
3. 标注首批 100 张图片（labelImg，生成 .txt）
4. 运行 check_p0_dataset.py 验证数据状态

**第 1 周（5.12-5.18）—— 数据扩充：**
1. 继续采集真实照片，目标 400+ 张
2. 下载 Zenodo accessibility_barriers.zip 并转换
3. 验证 Mendeley 路障数据集转换
4. 合并 AI 补充 + 真实照片 + 公共数据

**第 2 周（5.19-5.25）—— 训练 baseline：**
1. 运行 train_yolo.py，epochs=100
2. 输出 mAP50/Recall/FPS 指标
3. 对比 COCO 代理 vs 自定义模型
4. 调优（数据增强、超参数）

**第 3 周（5.26-6.1）—— 演示固定：**
1. A-D 点位实地走查，确认触发稳定
2. 录制演示视频素材（多机位、分镜脚本）
3. 视频剪辑（<=5 分钟，匿名审查）

**第 4 周（6.2-6.8）—— 材料收口：**
1. 论文初稿（引言/相关工作/架构/实验/结论）
2. PPT 初稿（选题/原理/方案/成果/价值）
3. 展架设计（80x180cm，核心图表）

**第 5 周（6.9-6.15）—— 评审优化：**
1. 论文/PPT/展架内审修改
2. 演示视频最终版
3. 匿名合规检查（无校名/人脸/车牌）

**第 6 周（6.16-6.20）—— 提交截止：**
1. 百度网盘打包上传
2. 官网提交确认
3. 备份所有材料

---

> 最后更新：2026-05-10  
> 对齐版本：PRD v2.2 / AGENTS.md  
> 当前状态：代码就绪，数据待采集，距提交截止 42 天
