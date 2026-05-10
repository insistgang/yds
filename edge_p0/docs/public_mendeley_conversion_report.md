# Mendeley road_obstacle Public Dataset Conversion Report

## 结论

Mendeley Data 的 `Obstacles Avoidance Assistance for Visually Impaired` 已作为 `road_obstacle` 公开 cold-start 数据源下载并转换到 P0-only YOLO 格式。

本数据只能作为 `road_obstacle` cold-start 补充，不能替代校园最后 20 米自采数据，也不能说明完整 P0 自训练模型已经完成。

## 数据源

| 项目 | 内容 |
|---|---|
| 数据源 | Obstacles Avoidance Assistance for Visually Impaired |
| URL | https://data.mendeley.com/datasets/xwhnp82rhk/1 |
| DOI | 10.17632/xwhnp82rhk.1 |
| License | CC BY 4.0 |
| Source format | YOLO |
| 原始类别 | `bump`, `fence`, `hole`, `pole` |
| P0 映射 | 全部折叠为 `road_obstacle` |

## Jetson 路径

| 项目 | 路径 |
|---|---|
| 下载 zip | `datasets/public_p0_sources/raw/mendeley_obstacles/xwhnp82rhk-1.zip` |
| 解压目录 | `datasets/public_p0_sources/raw/mendeley_obstacles/extracted/Obstacles Avoidance Assistance for Visually Impair/` |
| 转换输出 | `datasets/public_p0_yolo/` |
| data.yaml | `datasets/public_p0_yolo/data.yaml` |

## 源数据统计

| Split | Images | Label files | Boxes |
|---|---:|---:|---:|
| train | 1000 | 1000 | 1306 |
| valid | 250 | 250 | 321 |
| total | 1250 | 1250 | 1627 |

源类别框数：

| Source class | Train boxes | Valid boxes | Total |
|---|---:|---:|---:|
| `bump` | 397 | 82 | 479 |
| `fence` | 292 | 76 | 368 |
| `hole` | 329 | 90 | 419 |
| `pole` | 288 | 73 | 361 |

## 转换结果

| Split | Images | Label files | P0 boxes |
|---|---:|---:|---:|
| train | 1000 | 1000 | 1306 |
| val | 250 | 250 | 321 |
| test | 0 | 0 | 0 |
| total | 1250 | 1250 | 1627 |

P0 class id 检查：

```text
CLASS_ID_COUNTS={'0': 1627}
NON_P0_OR_NON_ROAD_CLASS_LINES=0
```

`prepare_public_p0_datasets.py summary` 输出：

```text
road_obstacle: 1627
stairs: 0
ramp: 0
blind_road_occupied: 0
```

## data.yaml

```yaml
path: datasets/public_p0_yolo
train: images/train
val: images/val
test: images/test
names:
  0: road_obstacle
  1: stairs
  2: ramp
  3: blind_road_occupied
```

## QA 要求

训练前仍需人工抽查：

- `bump/fence/hole/pole` 是否确实影响行人通行。
- 框是否贴合障碍物本身。
- 是否存在清晰人脸、车牌、校名、姓名牌等隐私风险。
- 是否存在与校园最后 20 米场景差异过大的样本。

## 当前边界

可以说：

- 已完成 Mendeley `road_obstacle` 公开 cold-start 数据转换。
- 转换结果是 P0-only YOLO 格式。
- 所有源类别均被折叠为 `road_obstacle`。

不能说：

- 已完成完整 P0 自训练模型。
- 公开数据已经覆盖 `blind_road_occupied`、`stairs`、`ramp`。
- `person`、`vehicle` 或其他 COCO 类别是 LinkAble P0 类别。
