# P0 数据集构建下一步

## 当前阶段

当前处于数据采集初始化阶段。已准备：

- `datasets/linkable_p0_raw/` 原始数据目录结构。
- `manifest.csv` 采集记录模板。
- `scripts/capture_p0_frames.py` 离散图片采集脚本。
- `scripts/check_p0_dataset.py` 数据数量和 manifest 检查脚本。
- 现场采集 checklist、标注规则和本说明文档。

当前尚未完成 P0 自训练模型。不能在材料中写成已经完成完整 P0 模型。

## 下一阶段流程

1. 实地采集 400-600 张 P0 正样本。
2. 完成人工隐私筛查，删除或打码敏感样本。
3. 使用 P0 四类标注为 YOLO 格式。
4. 按 train/val/test = 70/20/10 划分数据集。
5. 训练 `yolo11n` baseline。
6. 输出 mAP50、Recall、FPS、失败案例。

## 公开数据 cold-start 补充

公开数据只能作为 cold-start 和补充，不能替代校园最后 20 米自采数据。

优先候选：

- Zenodo Accessibility Barriers：补 `stairs` 和 `ramp`。
- Mendeley Obstacles Avoidance：补 `road_obstacle`。
- Roboflow tactile paving：只能作为盲道上下文辅助，不能直接映射为 `blind_road_occupied`。

相关文档和脚本：

- `docs/public_dataset_candidates.md`
- `docs/public_dataset_ingestion_plan.md`
- `scripts/prepare_public_p0_datasets.py`

公开数据转换后仍必须统一成四类 P0，不加入 `traffic_light`、`crosswalk`、`vehicle`、`person`。

## 数据采集目标

| 类别 | 目标数量 |
|---|---:|
| `road_obstacle` | 100-200 张 |
| `stairs` | 80-150 张 |
| `ramp` | 60-120 张 |
| `blind_road_occupied` | 80-150 张 |

第一阶段总目标为 400-600 张正样本。`negative` 样本用于误报检查，不作为 YOLO 类别写入 `data.yaml`。

## YOLO 数据集生成原则

- 原始数据目录 `datasets/linkable_p0_raw/` 保留采集样本和 manifest。
- 衍生训练目录建议使用 `datasets/linkable_p0_yolo/`。
- `data.yaml` 只包含四类 P0：
  - `road_obstacle`
  - `stairs`
  - `ramp`
  - `blind_road_occupied`
- 不加入 `traffic_light`、`crosswalk`、`vehicle`、`person` 等类别。
- train/val/test 按场景拆分，避免同一连续采集片段同时进入训练和测试。

## 实验输出要求

baseline 训练完成后至少记录：

- 数据集版本和样本数量。
- train/val/test 划分比例。
- mAP50。
- Recall。
- Jetson 端推理 FPS。
- 典型成功案例。
- 典型失败案例。
- 与当前 `yolo11n.pt` + COCO 代理类链路预演的边界差异。

## 材料口径

可以说：

- 当前已经完成 `road_obstacle` 端到端工程预演。
- 当前已经初始化 P0 数据采集和标注流程。
- 后续将通过四类 P0 数据训练自定义 baseline。

不能说：

- 已经完成完整 P0 自训练模型。
- `yolo11n.pt` + COCO 代理类就是最终 P0 模型。
- LinkAble 已经是成熟导盲设备。
- 系统可以替代盲杖或导盲犬。
