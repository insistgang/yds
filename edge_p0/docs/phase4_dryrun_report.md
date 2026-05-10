# Phase 4 Dry-run Report

## 结论

第 3A 关校园最后 20 米端到端 dry-run 已达到阶段目标。Jetson + USB 摄像头 + YOLO + P0 事件管线 + 中文提示 + 日志输出可以稳定运行。

本结论只表示端到端链路预演稳定，不表示已经完成完整 P0 自训练模型，也不表示 LinkAble 是成熟导盲设备。

## 运行环境

| 项目 | 结果 |
|---|---|
| 边缘设备 | Jetson |
| 摄像头 | HP w300 USB 摄像头 |
| 摄像头节点 | `/dev/video0` |
| 编码 | MJPG |
| 分辨率 | 640x480 |
| 帧率 | 30 FPS |
| 关键 v4l2 参数 | `exposure_dynamic_framerate=0` |
| 模型 | `yolo11n.pt` |
| 音频模式 | `--audio print` |
| 保存原始帧 | `--save-interval 0` |

## 当前定位

本轮使用 `yolo11n.pt` 和 COCO 代理类验证端到端链路。当前可以证明：

- USB 摄像头输入稳定。
- YOLO 推理能进入 P0 事件管线。
- `road_obstacle` 可以触发中文提示。
- EventBuilder cooldown 生效。
- 日志可以作为演示取证材料。

当前不能夸大为：

- 完整 P0 自训练模型已经完成。
- `blind_road_occupied`、`stairs`、`ramp` 已完成真实识别闭环。
- 系统已经成为成熟导盲设备或可以替代盲杖、导盲犬。

## 第 3A 日志

| 项目 | 结果 |
|---|---|
| 日志路径 | `runs/phase4_dryrun/route_print_dryrun_20260428_193356.log` |
| 日志大小 | 5.9K |
| 事件总数 | 17 |
| 音频/文本输出次数 | 10 |
| WARN | 0 |
| ERROR | 0 |

## 事件统计

| P0 label | 次数 |
|---|---:|
| `road_obstacle` | 17 |

本轮只实际触发了 `road_obstacle`。其余 P0 类别需要通过真实 P0 数据采集、标注和模型训练补齐。

## 中文提示统计

| 中文提示 | 次数 |
|---|---:|
| 前方有障碍，请注意避让。 | 14 |
| 右前方有障碍，请注意避让。 | 3 |

## spoken/cooldown 统计

| 状态 | 次数 |
|---|---:|
| spoken | 10 |
| cooldown | 7 |

该结果说明同类事件在冷却时间内不会反复刷屏，适合现场演示。

## FPS 摘要

初始阶段存在模型加载和预热开销，后续 FPS 稳定接近 30。最后 5 条统计如下：

```text
[STATS] fps=30.00 frames=2400 events=16 speeches=10
[STATS] fps=30.00 frames=2430 events=17 speeches=10
[STATS] fps=30.01 frames=2460 events=17 speeches=10
[STATS] fps=29.99 frames=2490 events=17 speeches=10
[STATS] fps=30.00 frames=2520 events=17 speeches=10
```

## WARN/ERROR 摘要

本轮日志中：

- `WARN_COUNT=0`
- `ERROR_COUNT=0`

没有观察到摄像头、YOLO、事件管线或 print 输出相关异常。

## 隐私说明

本轮 dry-run 使用 `--save-interval 0`，未保存原始视频或连续图像流。后续取证应继续优先保存结构化事件日志、FPS 摘要、必要的短片段演示素材，并避免出现可识别个人隐私、导师、学校、队员等匿名违规信息。

## 限制说明

当前阶段仍有以下限制：

- `road_obstacle` 是通过 `yolo11n.pt` + COCO 代理类完成的端到端预演。
- `blind_road_occupied`、`stairs`、`ramp` 尚未完成真实 P0 自定义模型验证。
- 还需要固定校园最后 20 米路线，采集 P0 四类真实数据，并形成标注与训练记录。
- 语音机制已作为可用项，不再阻塞本轮路线和数据主线。
