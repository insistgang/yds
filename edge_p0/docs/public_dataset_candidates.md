# P0 公开数据集候选清单

## 当前结论

公开数据可以作为 P0 自训练模型的冷启动补充，但不能替代校园最后 20 米自采数据。当前优先补齐：

- `stairs`
- `ramp`
- `road_obstacle`

`blind_road_occupied` 是“盲道 + 被占用物体”的关系型场景，公开数据很少直接覆盖。触觉铺装/盲道本体数据只能作为背景或上下文辅助，不能直接当作 `blind_road_occupied`。

## P0 边界

最终 `data.yaml` 只允许四类：

```yaml
names:
  0: road_obstacle
  1: stairs
  2: ramp
  3: blind_road_occupied
```

禁止加入：

- `traffic_light`
- `crosswalk`
- `vehicle`
- `person`

也不能把 COCO 代理类写成最终 P0 类别。

## 候选数据源

| 优先级 | 数据源 | 许可 | 格式 | 建议 P0 用途 | 处理建议 | 风险 |
|---:|---|---|---|---|---|---|
| 1 | [Image Dataset of Accessibility Barriers](https://zenodo.org/records/6382090) | CC BY 4.0 | CVAT XML + images | `stairs`, `ramp` | 优先下载和转换；`stairs/ramps` 可直接映射；`steps` 需要人工审核后再并入 `stairs` | 8.0GB，单人标注，需抽查 |
| 2 | [Obstacles Avoidance Assistance for Visually Impaired](https://data.mendeley.com/datasets/xwhnp82rhk/1) | CC BY 4.0 | YOLO | `road_obstacle` | 将 `pole/fence/bump/hole` 统一折叠为 `road_obstacle`，不要保留原始类别 | 来源来自公开 Roboflow，经二次整理，需抽查 |
| 3 | [Tactile-Paving-Blind-Assist](https://universe.roboflow.com/labeling-qcirk/tactile-paving-blind-assist) | CC BY 4.0 | Roboflow object detection | 盲道上下文辅助 | 可用于盲道/触觉铺装背景理解，不直接映射为 `blind_road_occupied` | 只有 Dot-Stop/Line-Go，不是占用关系 |
| 4 | [Roboflow stair detection workspace](https://universe.roboflow.com/stair-detection-itfjj) | 逐项目确认 | Roboflow object detection | `stairs` | 作为补充候选，下载前确认具体项目版本和 license | Roboflow 项目质量和许可差异大 |
| 5 | [TW0521/Obstacle-Dataset](https://github.com/TW0521/Obstacle-Dataset) | 未确认 | VOC + YOLO | `road_obstacle` 候选 | 只做内部候选，不进入正式 baseline，除非确认许可 | 仓库页面未见清晰 license |
| 6 | [Project Sidewalk RampNet Dataset](https://huggingface.co/datasets/projectsidewalk/rampnet-dataset) | MIT | Hugging Face parquet | `ramp` 研究候选 | 数据量大，标注形态需自定义转换；不作为第一优先级 | 街景域差异大，转换成本高 |

## 推荐落地顺序

1. 先处理 Zenodo Accessibility Barriers，补 `stairs/ramp`。
2. 再处理 Mendeley Obstacles Avoidance，补 `road_obstacle`。
3. 自采 `blind_road_occupied`，不等待公开数据。
4. 用 Roboflow 和 Hugging Face 候选数据做补充，不作为第一批 baseline 依赖。

## 公开数据不能替代的部分

仍必须自采：

- 校园最后 20 米路面材质。
- HP w300 USB 摄像头视角。
- Jetson 实际演示路线光照。
- 盲道被箱子、背包、椅子、自行车占用的关系型样本。

## 引用和材料口径

可以说：

- 公开数据用于 P0 冷启动训练和补充。
- 校园自采数据用于最终场景适配和验证。
- `yolo11n.pt + COCO` 仍只是端到端链路预演。

不能说：

- 公开数据已经覆盖完整校园 P0 场景。
- 已经完成完整 P0 自训练模型。
- LinkAble 已经是成熟导盲设备。
