# LinkAble 本地跑通报告

> 日期：2026-06-03
> 目的：先在本地确认 P0 主链路、脚本和证据产出能跑，再上 Jetson / 摄像头 / 现场点位。
> 结论：本地功能闭环已跑通；本机 CPU 性能指标未完全达标；硬件相关项仍需上设备验证。

---

## 1. 三遍 Review 结论

### 第一遍：工程健康检查

| 项目 | 结果 | 说明 |
|---|---:|---|
| 单元测试 | PASS | `39 passed` |
| 格式检查 | PASS | `git diff --check` 无输出 |
| 根目录文档 | PASS | 已新增 `LinkAble_设备选型说明.md`，并加入 `DOCS_INDEX.md` |

第一遍结论：

```text
代码层没有测试失败和空白格式错误，可以继续做功能验证。
```

---

### 第二遍：本地功能脚本验证

| 功能 | 命令 / 证据 | 结果 |
|---|---|---|
| 单图检测 -> 事件 -> 中文提示 -> print 播报 -> 结构化事件 | `python -m linkable_edge.image_demo --model best.pt --source <test image> --audio print --event-conf 0.25 --publish` | PASS |
| 离线周报样本 | `python scripts/generate_report_sample.py` | PASS，输出 `runs/report_sample/events.json` 和 `runs/report_sample/report.md` |
| 端到端延迟短版 | `python scripts/measure_e2e_latency.py --video-limit 2 --max-frames 120` | PASS，47 次播报事件，p99=0.0ms，目标 <1500ms |
| EventBuilder 消融短版 | `python scripts/ablation_event_builder.py --video-limit 2 --max-frames 120` | PASS，播报次数从 189 降到 41，减少 78.3% |
| EventBuilder 消融三视频完整版 | `python scripts/ablation_event_builder.py --video-limit 3` | PASS，播报次数从 1124 降到 241，减少 78.6% |
| 端到端延迟三视频完整版 | `python scripts/measure_e2e_latency.py --video-limit 3` | PASS，226 次播报事件，目标 <1500ms |
| TTS 缓存核对 | `python scripts/check_tts_cache.py` | PARTIAL，28 条模板中 22 条有缓存，6 条缺缓存 |
| pre/infer/post 性能分段 | `python scripts/profile_inference_stages.py --limit 50` | PASS，inference 占 95.8%，瓶颈在推理段 |
| 本地 benchmark | `python scripts/quick_benchmark.py` | PARTIAL，本机 CPU 29.6 FPS，单帧 33.80ms |

第二遍结论：

```text
本地 P0 功能闭环能跑通：能检测、能生成事件、能播报、能记录、能生成报告。
但当前本机 PyTorch 是 CPU 版，CUDA 不可用，因此本地 FPS 只有 29.6，略低于 30 FPS 目标。
```

---

### 第三遍：对照开发文档验收

| 开发文档要求 | 本地状态 | 说明 |
|---|---|---|
| 输入可替代：图片 / 视频 / 摄像头 | PARTIAL | 图片和视频已本地跑通；USB/CSI 摄像头需上设备 |
| YOLO 推理 | PASS | `best.pt` 本地可推理 |
| P0 四类过滤 | PASS | 测试视频中已触发 road_obstacle、blind_road_occupied、stairs、ramp |
| 事件聚合与去抖 | PASS | 消融短版证明播报次数减少 78.3% |
| 模板化中文提示 | PASS | P0 提示固定落在 28 条预加载模板中 |
| 一帧只播最高优先级 | PASS | 单测覆盖 stairs > road_obstacle > blind_road > ramp |
| 本地音频播报 | PASS / PARTIAL | `print` 播报本地通过；真实 mp3 播放需本机播放器或 Jetson 验证 |
| 结构化事件记录 | PASS | 单图 demo 已输出 `/api/v1/detect` 风格 JSON |
| 断网不崩 | PASS | 报告默认离线；上报层有 SafePublisher 缓存；MiniMax 不作为本地必需 |
| 性能指标 FPS >=30 | PARTIAL | 本机 CPU 29.6 FPS；需 Jetson/GPU 重测 |
| 单帧延迟 <100ms | PASS | 本机 CPU 平均 33.80ms |
| 检测到播报 <1.5s | PASS | 本地短版 p99=0.0ms；真实 mp3 播放需再测 |
| A-D 点位演示 | MISSING | 必须现场完成 |
| 自采数据来源 / 标注人工抽检 | MISSING | 需要人工台账和抽检 |

第三遍结论：

```text
开发文档里的“本地可验证代码功能”基本跑通。
开发文档里的“硬件、现场、人工数据核验”还没有完成，不能说全部最终验收完成。
```

---

## 2. 当前可以说已经本地跑通的内容

可以确认：

- 单图输入链路跑通。
- 视频输入链路跑通。
- YOLO `best.pt` 本地推理跑通。
- 检测结果可以转成 P0 事件。
- EventBuilder 去抖 / 冷却可运行。
- 中文模板提示可生成。
- `print` 音频兜底可用。
- 结构化事件 JSON 可输出。
- 离线周报样本可生成。
- 端到端短版脚本可跑。
- 消融短版脚本可跑。
- 三视频 EventBuilder 消融 CSV / PNG 已生成。
- 三视频端到端延迟逐事件 CSV / 分布图已生成。
- pre / infer / post 性能分段 CSV 已生成，推理段占比 95.8%。
- 边界异常测试已补：坏图、空帧、断网缓存、双事件最高优先级、音频锁串行。
- 单元测试全部通过。

---

## 3. 当前不能说已经完全完成的内容

不能说“所有功能最终都完成”，原因如下：

1. **USB / CSI 摄像头没有本地验证。**
   图片和视频输入已跑通，但真实摄像头输入必须上设备或接摄像头验证。

2. **本机性能不是目标设备性能。**
   当前本机 PyTorch 是 CPU 版：

   ```text
   PyTorch: 2.12.0+cpu
   CUDA可用: False
   ```

   本地 benchmark：

   ```text
   FPS: 29.6
   平均延迟: 33.80ms
   ```

   单帧延迟达标，但 FPS 略低于 30。Jetson / GPU 上必须重测。

3. **真实 mp3 播放未作为最终验收。**
   本地已验证 `print` 兜底播报；MiniMax 缓存 mp3 播放需要本地播放器或 Jetson 音频环境验证。
   当前模板缓存核对结果是 28 条模板、22 条已有缓存、6 条缺缓存，严格一一对应尚未满足。

4. **A-D 点位演示还没现场完成。**
   本地视频可以证明链路，但比赛演示仍需要固定点位、道具、拍摄和匿名检查。

5. **数据来源和标注质量仍需人工核验。**
   `datasets/p0_yolo` 已有 train/val/test 数据，但测试集自采占比、同场景不跨集合、每类 50 张标注抽检不是代码能自动完全证明的。

---

## 4. 上设备前建议再跑的本地命令

```powershell
cd D:\000\yds\edge_p0
$env:PYTHONPATH='D:\000\yds\edge_p0\src'
python -m pytest -q
python -m linkable_edge.image_demo --model best.pt --source datasets\p0_yolo\images\test\000000001451.jpg --audio print --event-conf 0.25 --publish
python -m linkable_edge.video_demo --source test_videos\1.mp4 --model best.pt --audio print --event-conf 0.25 --max-frames 120
python scripts\measure_e2e_latency.py --video-limit 2 --max-frames 120
python scripts\ablation_event_builder.py --video-limit 2 --max-frames 120
python scripts\check_tts_cache.py
python scripts\profile_inference_stages.py --limit 50
python scripts\generate_report_sample.py
python scripts\quick_benchmark.py
```

---

## 5. 上设备后必须跑的命令

```powershell
cd D:\000\yds\edge_p0
$env:PYTHONPATH='D:\000\yds\edge_p0\src'
python -m linkable_edge.usb_demo --model best.pt --audio print --event-conf 0.25 --offline
python scripts\quick_benchmark.py
python scripts\measure_e2e_latency.py --video-limit 2 --max-frames 120
```

如果本地 mp3 缓存和播放器就绪，再测：

```powershell
python scripts\preload_tts_cache.py
python -m linkable_edge.video_demo --source test_videos\1.mp4 --model best.pt --audio minimax --event-conf 0.25 --max-frames 120
```

---

## 6. 答辩证据产出

本轮新增证据如下：

| 状态 | 证据项 | 关键结果 | 产出路径 |
|---|---|---|---|
| [x] | EventBuilder 三视频消融 JSON | 1124 -> 241，减少 78.6% | `edge_p0/runs/ablation/event_builder_ablation.json` |
| [x] | EventBuilder 三视频消融 CSV | 列：视频名 / 模式 / 播报次数 / 减少比例 | `edge_p0/runs/ablation/event_builder_ablation.csv` |
| [x] | EventBuilder 消融柱状图 | X 轴视频，Y 轴播报次数，含有/无 EventBuilder 两组柱 | `edge_p0/runs/ablation/event_builder_ablation.png` |
| [x] | 端到端延迟汇总 JSON | 三视频 226 次播报，p99 远低于 1500ms | `edge_p0/test_results/e2e_latency.json` |
| [x] | 端到端延迟逐事件 CSV | 每次播报一行延迟记录 | `edge_p0/test_results/e2e_latency_events.csv` |
| [x] | 端到端延迟分布图 | 延迟分布 PNG | `edge_p0/test_results/e2e_latency_distribution.png` |
| [x] | 坏图 / 空帧降级测试 | 检测层返回空结果，不崩溃 | `edge_p0/tests/test_detector.py` |
| [x] | 断网离线报告 / 本地缓存测试 | 上报失败不崩，事件落本地缓存，报告链路可运行 | `edge_p0/tests/test_offline_resilience.py` |
| [x] | 双事件最高优先级测试 | 同帧 stairs + ramp 只播 stairs | `edge_p0/tests/test_pipeline.py` |
| [x] | 音频锁串行测试 | 两个播报请求被串行执行 | `edge_p0/tests/test_audio_manager.py` |
| [x] | TTS 缓存核对 JSON | 28 条模板、22 条已缓存、6 条缺缓存 | `edge_p0/test_results/tts_cache/tts_cache_check.json` |
| [x] | TTS 缓存核对 CSV | 每条模板对应缓存状态 | `edge_p0/test_results/tts_cache/tts_cache_check.csv` |
| [x] | pre / infer / post 分段 CSV | preprocess 0.84ms，infer 30.16ms，post 0.47ms | `edge_p0/test_results/benchmark_stages/inference_stages.csv` |
| [x] | pre / infer / post 分段汇总 | infer 占 95.8%，瓶颈在推理段 | `edge_p0/test_results/benchmark_stages/inference_stages_summary.json` |

本轮新增 / 补强测试结果：

```text
39 passed in 4.20s
```

EventBuilder 三视频消融结果：

```text
视频 1：214 -> 47，减少 78.0%
视频 2：422 -> 91，减少 78.4%
视频 3：488 -> 103，减少 78.9%
总计：1124 -> 241，减少 78.6%
```

TTS 缓存差异：

```text
模板总数：28
已缓存：22
缺缓存：6
缺失模板：
- 左侧有障碍，请注意避让。
- 右侧有障碍，请注意避让。
- 左侧有台阶，请减速。
- 右侧有台阶，请减速。
- 左侧有坡道。
- 右侧有坡道。
```

性能瓶颈定位：

```text
preprocess：0.84ms，占 2.7%
infer：30.16ms，占 95.8%
postprocess：0.47ms，占 1.5%
结论：瓶颈明确在 YOLO 推理段。
```

---

## 7. 最终判断

当前状态应该这样表述：

> LinkAble 的 P0 本地功能闭环已经跑通：图片/视频输入、YOLO 推理、事件去抖、模板提示、print 播报、结构化记录、离线报告和消融实验都能在本地执行。
> 但本机是 CPU 版 PyTorch，FPS 为 29.6，略低于 30 FPS 目标；USB/CSI 摄像头、真实音频播放、Jetson 性能和 A-D 点位演示还需要上设备和现场验证。

不要说：

```text
所有功能已经最终完成。
所有性能指标都已达标。
已经完成现场演示。
已经完成 Jetson 部署验收。
```

可以说：

```text
本地 P0 闭环已跑通，具备上设备联调条件。
```
