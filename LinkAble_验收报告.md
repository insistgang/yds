# LinkAble 技术验收报告

> 生成日期：2026-06-03
> 验收依据：根目录 `LinkAble_开发文档.md` V1.0
> 验收范围：按 8 层技术架构核对 `edge_p0`，只补齐 P0 本地闭环和证据链缺口，不扩展实时大模型、完整 App、量产穿戴能力。

---

## 1. 本轮结论

LinkAble 当前边缘端主链路已经具备稳定演示基础：

```text
输入源 -> YOLO 检测 -> EventBuilder 去抖/冷却 -> 中文模板 -> Pipeline 仲裁
      -> AudioManager 本地播报/降级 -> SafePublisher 结构化上报/缓存
```

本轮已完成的补缺：

- 输入层新增 `FallbackFrameSource`：启动失败和读帧失败都能自动切到下一级输入源。
- 事件层新增消融开关：`enable_debounce` / `enable_cooldown`，并记录连续帧 `source_frame_ids`。
- Pipeline 已有“一帧只播最高优先级事件”，本轮补测试证据。
- `build_audio_output("minimax")` 已切到 `AudioManager`，具备 MiniMax -> pyttsx3 -> print 三级降级和全局音频锁。
- `SafePublisher` 已补本地缓存、重试入口、后台发布模式，USB/视频 demo 已启用后台上报，避免拖垮播报主链路。
- `BenchmarkCollector` 已补 `end_to_end_ms` 字段，并写入 JSON/CSV。
- 语义层已收敛为固定 P0 模板：P0 渲染结果必定属于 `ALL_TEMPLATE_TEXTS` 的 28 条缓存模板。
- P1 `report_gen.py` 已补为离线结构化事件报告模块，不进入实时链路。
- P1 报告样本脚本默认改为离线模板生成，只有显式 `--use-llm` 才调用 Kimi/Qwen。
- 测试集从 30 个增加到 35 个，当前全通过。

---

## 2. 模块对照

| 规格层 | 要求模块 | 现状 |
|---|---|---|
| 输入层 | `inputs.py` | 存在，支持 CSI / USB / 视频 / 图片序列，已补读帧失败回退 |
| 检测层 | `detector.py` | 存在，YOLO 推理、P0 过滤、COCO 冷启动代理、异常空结果降级已实现 |
| 事件层 | `event_builder.py` | 存在，去抖、冷却、TTL、P1 隔离、优先级、连续帧证据已实现 |
| 语义层 | `semantics.py` | 存在，P0 中文模板和方向/距离格式化已实现 |
| Pipeline | `pipeline.py` | 存在，主链路串联和最高优先级播报已实现 |
| 音频层 | `audio_manager.py` | 存在，音频锁、预加载、三级降级已实现，并已接入 `minimax` 音频入口 |
| 上报层 | `publisher.py` | 存在，结构化 JSON、HTTP POST、失败缓存、补传、后台发布已实现 |
| 性能层 | `benchmark.py` | 存在，JSON/CSV、FPS、分段延迟、端到端延迟字段已实现 |
| P1 报告 | `report_gen.py` | 已存在，基于结构化事件生成离线周报；`scripts/generate_report_sample.py` 作为样本产出入口 |

额外存在但未画入 8 层图的工程文件：

| 文件 | 作用 |
|---|---|
| `audio.py` | 音频底层实现：MiniMax、pyttsx3、print、缓存 key、播放器 |
| `image_demo.py` / `video_demo.py` / `usb_demo.py` | 单图、视频/图片序列、USB 摄像头演示入口 |
| `demo.py` | Mock pipeline 演示入口 |
| `camera_check.py` | 摄像头诊断 |
| `models.py` | `Detection` / `FrameDetections` / `DetectionEvent` 数据结构 |

---

## 3. 八层验收表

| 层 | 判定 | 证据 | 本轮补缺动作 |
|---|---|---|---|
| 第 1 层：输入 | `[PASS]` | `inputs.py` 有 `CsiCameraSource` / `UsbCameraSource` / `VideoFileSource` / `ImageSequenceSource` / `auto_select_source()`；`tests/test_inputs.py` 覆盖 open/read fallback | 新增 `FallbackFrameSource`，读帧失败自动释放当前源并切下一级；修复图片 glob |
| 第 2 层：检测 | `[PASS]` | `detector.py` 定义 P0 四类、COCO 冷启动映射、`YoloDetectorConfig(engine=...)` 后端预留、推理异常返回空 `FrameDetections`；`tests/test_detector.py` 覆盖标签映射和方向推断 | 未重写检测层；保持 TensorRT 仅预留，符合冲刺期策略 |
| 第 3 层：事件 | `[PASS]` | `event_builder.py` 实现 `min_consecutive_frames=2`、`emit_cooldown_frames=5`、TTL 清理、优先级排序、P1 标签过滤；`tests/test_event_builder.py` 覆盖去抖、冷却、P1 隔离、消融开关、`source_frame_ids=[1,2]` | 新增 `enable_debounce` / `enable_cooldown`；事件记录连续帧 ID |
| 第 4 层：语义 | `[PASS]` | `semantics.py` 输出中文短句；`ALL_TEMPLATE_TEXTS` 为 28 条；`tests/test_semantics.py` 覆盖四类提示、五类方向、路障 1-8 米距离模板，并验证 P0 渲染结果全部属于预加载模板 | 收敛动态文案，避免现场触发未缓存 TTS |
| 第 5 层：Pipeline | `[PASS]` | `pipeline.py` 对同帧事件按优先级只播 `events[0]`；`tests/test_pipeline.py` 验证 stairs+ramp 只播 stairs，并记录 detection-to-speak latency | 补 Pipeline 仲裁和延迟测试 |
| 第 6 层：音频 | `[PASS]` | `audio_manager.py` 有 `threading.Lock`、MiniMax / pyttsx3 / print 降级、`preload()`；`audio.py` 的 `build_audio_output("minimax")` 已接入 `AudioManager` | 将 demo 使用的 `minimax` 入口切到 AudioManager，避免单一 MiniMax 后端 |
| 第 7 层：上报 | `[PASS]` | `publisher.py` 只构造结构化事件 JSON，不含原始图像；`HttpPublisher` 使用 `POST` + `Content-Type: application/json`；`tests/test_publisher.py` 验证 payload 和失败缓存 | `SafePublisher` 新增本地缓存、补传、后台发布；USB/视频 demo 启用 `async_publish=True` |
| 第 8 层：性能 | `[PASS]` | `benchmark.py` 输出 JSON/CSV，含 pre / infer / post / total / `end_to_end_ms`；`tests/test_benchmark.py` 覆盖保存字段 | 新增端到端延迟字段 |

---

## 4. 关键证据状态

| 证据项 | 状态 | 当前路径 / 数字 | 说明 |
|---|---|---|---|
| 训练 baseline | `[PARTIAL]` | `docs/evaluation/PERFORMANCE_METRICS.md`：mAP50=0.8264，Recall=0.7684，FPS=108.3，单帧延迟=9.23ms | 这是 RTX 4070 SUPER 环境记录；Jetson 实测仍需现场补测 |
| 数据集规模 | `[PASS]` | `edge_p0/datasets/p0_yolo/images`: train=14909, val=1560, test=868；labels 同数 | 已转换 YOLO 数据存在；`linkable_p0_raw` 仍为空台账入口 |
| EventBuilder 消融 | `[PASS]` | `edge_p0/runs/ablation/event_builder_ablation.json`：4472 帧，无去抖 2608 次，有去抖 545 次，减少 79.1% | 可直接进论文/答辩作为去抖价值证据 |
| 端到端延迟 | `[PARTIAL]` | `edge_p0/test_results/e2e_latency.json` 当前 spoken count=0；代码层已补 `PipelineResult.latency_ms` 和 Benchmark 字段 | 需要用能触发事件的视频/现场输入重跑，目标 <1.5s |
| 固定点位演示 | `[MISSING]` | 尚无本轮生成的视频/点位实测记录 | 需要现场完成 A-D 点位素材和录制 |
| 大模型报告样本 | `[PASS]` | `edge_p0/src/linkable_edge/report_gen.py`、`edge_p0/runs/report_sample/events.json` 和 `edge_p0/runs/report_sample/report.md` | 默认离线模板生成；`--use-llm` 才调用云端 |

---

## 5. 仍未完成但不应由本轮代码强行补的项

| 缺口 | 原因 | 建议下一步 |
|---|---|---|
| Jetson 实测 FPS / 单帧延迟 / 检测到播报延迟 | 需要 Jetson 或现场设备 | 在 Jetson 上运行 `quick_benchmark.py` 和 `measure_e2e_latency.py`，回填 `PERFORMANCE_METRICS.md` |
| 测试集自采占比 >=20% 核验 | 需要来源元数据或人工台账 | 按 `DATA_SOURCE_LEDGER.md` 给 test 集图片补来源字段，再统计 |
| 同一地点/同一视频不跨集合核验 | 需要文件命名、采集批次或 manifest 元数据 | 建立 scene_id / clip_id，再跑泄漏检查 |
| 每类 50 张标注抽检 | 需要人工看框和隐私 | 使用 `visualize_p0_dataset.py` 抽样截图，人工勾选问题 |
| TTS 缓存命中 <5ms 实测 | 需要本地 mp3 缓存和播放器环境 | 先运行 `preload_tts_cache.py`，再测缓存命中启动时间 |
| A-D 点位演示视频 | 需要现场路线、道具、匿名拍摄 | 按 `edge_p0/docs/demo_route_plan.md` 和 `demo_video_shotlist.md` 执行 |

---

## 6. 本轮验证命令

```powershell
$env:PYTHONPATH='D:\000\yds\edge_p0\src'
python -m pytest -q
```

结果：

```text
35 passed in 3.97s
```

```powershell
$env:PYTHONPATH='D:\000\yds\edge_p0\src'
python scripts\generate_report_sample.py
```

结果：

```text
事件数据已保存: edge_p0\runs\report_sample\events.json
报告已保存: edge_p0\runs\report_sample\report.md
默认离线模板生成，不调用 Kimi/Qwen
```

```powershell
python scripts\check_p0_dataset.py --root datasets\linkable_p0_raw
```

结果：

```text
linkable_p0_raw 当前 0 张；该目录不是已转换训练集，不能作为 baseline 数据规模依据。
```

```powershell
Get-ChildItem datasets\p0_yolo\images
Get-ChildItem datasets\p0_yolo\labels
```

结果：

```text
images/train=14909, images/val=1560, images/test=868
labels/train=14909, labels/val=1560, labels/test=868
```

---

## 7. 下一步优先级

1. **先重跑端到端延迟**：用能触发事件的视频或 USB 摄像头，目标是让 `e2e_latency.json` 出现 spoken count 和 p50/p99。
2. **再做 Jetson 数字**：把 FPS、单帧延迟、检测到播报延迟从 PC 数字升级为 Jetson 实测数字。
3. **补数据来源核验**：重点是 test 集自采占比、同场景不跨集合、每类 50 张标注抽检。
4. **固定 B 点演示**：优先保证路面障碍点位稳定，再扩 A/C/D。
5. **最后材料收口**：论文、PPT、展架统一使用“边缘智能无障碍辅助通行原型系统”和“结构化事件支撑治理”口径。

---

## 8. 结论

按根目录最新 `LinkAble_开发文档.md` 核对，代码层的 P0 本地闭环已经补到可验收状态。当前真正未完成的内容主要不是代码，而是现场/数据证据：Jetson 实测、端到端触发数据、A-D 点位视频、自采来源核验和人工标注抽检。

后续不要再扩实时 LLM、Flutter 或导航功能。先把上述证据补齐，LinkAble 的答辩主线才会稳。
