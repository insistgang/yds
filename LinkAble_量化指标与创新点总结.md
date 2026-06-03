# LinkAble 量化指标与创新点总结

> 日期：2026-06-03
> 用途：用于论文、PPT、展架和答辩时统一“已有量化证据”和“主打创新点”的口径。
> 对齐范围：PRD v2.2 / AGENTS.md / `LinkAble_开发文档.md`

---

## 1. 一句话结论

LinkAble 当前最适合主打的创新点是：

```text
面向无障碍治理的数据闭环：
用边缘视觉感知把盲道占用、台阶、坡道和路面障碍转化为可播报、可记录、可统计、可治理的结构化事件数据。
```

推荐答辩主线：

```text
无障碍治理数据闭环
= 边缘实时感知
+ EventBuilder 事件去抖
+ 本地语音提示
+ 结构化事件记录
+ 事后报告增强
```

不要把项目讲成“成熟导盲设备”，也不要主打“实时 AI 智能体”。当前更稳、更有数据支撑的口径是：

> 无人驾驶让车看懂路，LinkAble 让弱势行人听懂路、让管理者看见盲区。

---

## 2. 创新点优先级

| 优先级 | 创新点 | 是否主打 | 答辩定位 | 原因 |
|---:|---|---|---|---|
| 1 | 无障碍治理数据闭环 | 主打 | 核心创新 | 差异化最强，不只是“YOLO + 语音播报”，能服务校园/社区管理方 |
| 2 | 边缘实时感知 | 强支撑 | 架构创新 | 解释为什么使用 Jetson / 边缘 AI 设备：低延迟、断网可用、隐私友好 |
| 3 | EventBuilder 事件去抖 | 强支撑 | 技术创新 | 有消融实验量化证据，重复播报减少 78.6% |
| 4 | 本地语音缓存与降级链 | 工程亮点 | 稳定性保障 | 能回答断网、TTS API 失败、现场音频不稳定等问题 |
| 5 | AI 智能体专项 | 谨慎使用 | P1 增强 | 当前大模型只做事后报告增强，不进入实时安全链路 |

---

## 3. 可量化数据总览

### 3.1 模型与性能指标

| 指标 | 当前数字 | 状态 | 证据路径 | 答辩说明 |
|---|---:|---|---|---|
| GPU baseline FPS | 108.3 FPS | 可引用 | `docs/evaluation/PERFORMANCE_METRICS.md` | RTX 4070 SUPER 环境，不是 Jetson 实测 |
| GPU baseline 平均单帧延迟 | 9.23 ms | 可引用 | `docs/evaluation/PERFORMANCE_METRICS.md` | 单帧推理延迟，目标 <100ms |
| GPU baseline P99 延迟 | 13.36 ms | 可引用 | `docs/evaluation/PERFORMANCE_METRICS.md` | 训练后 YOLO11n 模型 |
| 本机 CPU FPS | 29.6 FPS | 可引用但需限定 | `LinkAble_本地跑通报告.md` | CPU 版 PyTorch，略低于 30 FPS |
| 本机 CPU 平均单帧延迟 | 33.80 ms | 可引用 | `LinkAble_本地跑通报告.md` | 单帧延迟达标，目标 <100ms |
| pre / infer / post 分段 | 0.84 / 30.16 / 0.47 ms | 可引用 | `edge_p0/test_results/benchmark_stages/inference_stages_summary.json` | 证明瓶颈在 infer |
| 推理阶段占比 | 95.8% | 可引用 | `edge_p0/test_results/benchmark_stages/inference_stages_summary.json` | pre 占 2.67%，post 占 1.53% |

答辩口径：

> GPU baseline 达到 108.3 FPS，平均单帧推理延迟 9.23ms；本机 CPU 环境下平均单帧延迟 33.80ms。分段 benchmark 显示推理阶段占总耗时 95.8%，因此当前性能瓶颈主要集中在 YOLO inference 段。Jetson 目标设备实测仍需现场补齐，不能把 RTX 4070 SUPER 数字说成 Jetson 数字。

### 3.2 准确率与训练指标

| 指标 | 当前数字 | 状态 | 证据路径 | 说明 |
|---|---:|---|---|---|
| Precision | 0.8774 | 可引用 | `docs/evaluation/PERFORMANCE_METRICS.md` | 训练 baseline 指标 |
| Recall | 0.7684 | 可引用 | `docs/evaluation/PERFORMANCE_METRICS.md` | 训练 baseline 指标 |
| mAP50 | 0.8264 | 可引用 | `docs/evaluation/PERFORMANCE_METRICS.md` | 模型检测效果核心指标 |
| mAP50-95 | 0.6336 | 可引用 | `docs/evaluation/PERFORMANCE_METRICS.md` | 更严格 IoU 区间指标 |

答辩口径：

> P0 四类检测模型 baseline 的 mAP50 为 0.8264，Precision 为 0.8774，Recall 为 0.7684。当前指标可以证明模型具备 baseline 可用性，但现场点位识别成功率还需要单独统计，不能把 mAP 直接等同于现场演示成功率。

### 3.3 数据集数量

| 指标 | 数量 | 证据路径 | 说明 |
|---|---:|---|---|
| 总图片数 | 17,337 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | YOLO 数据集总规模 |
| train 图片数 | 14,909 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | 训练集 |
| val 图片数 | 1,560 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | 验证集 |
| test 图片数 | 868 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | 测试集 |
| label 文件缺失数 | 0 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | train/val/test 均无缺失 label |
| 总标注框数 | 49,948 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | 四类合计 |
| blind_road_occupied 标注框 | 1,571 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | 盲道占用 |
| stairs 标注框 | 6,007 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | 台阶 |
| ramp 标注框 | 4,717 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | 坡道 |
| road_obstacle 标注框 | 37,653 | `edge_p0/runs/dataset_visualization/p0_yolo/summary.json` | 路面障碍 |

需要诚实说明：

```text
road_obstacle 占比明显偏高，类别不平衡仍是后续优化点。
blind_road_occupied 是最小类，后续自采数据应优先补这一类。
```

### 3.4 EventBuilder 消融实验

| 视频 | 无 EventBuilder 播报次数 | 有 EventBuilder 播报次数 | 减少比例 |
|---|---:|---:|---:|
| 视频 1 | 214 | 47 | 78.0% |
| 视频 2 | 422 | 91 | 78.4% |
| 视频 3 | 488 | 103 | 78.9% |
| 合计 | 1124 | 241 | 78.6% |

证据路径：

- `edge_p0/runs/ablation/event_builder_ablation.json`
- `edge_p0/runs/ablation/event_builder_ablation.csv`
- `edge_p0/runs/ablation/event_builder_ablation.png`

答辩口径：

> 普通目标检测会在连续帧上反复输出检测框，如果直接转语音，会造成高频重复提醒。LinkAble 设计 EventBuilder，通过连续帧确认、去抖、冷却和优先级仲裁，把单帧检测转化为稳定事件。三段视频消融实验中，播报次数从 1124 次降低到 241 次，重复提示减少 78.6%。

### 3.5 端到端延迟

| 指标 | 当前数字 | 证据路径 | 说明 |
|---|---:|---|---|
| 播报事件数 | 226 | `edge_p0/test_results/e2e_latency.json` | 三视频本地延迟测试 |
| 平均端到端延迟 | 0.14 ms | `edge_p0/test_results/e2e_latency.json` | print/cache 链路 |
| 最大端到端延迟 | 16 ms | `edge_p0/test_results/e2e_latency.json` | 低于 1500ms 目标 |
| p50 | 0.0 ms | `edge_p0/test_results/e2e_latency.json` | print 模式下接近 0 |
| p99 | 0.0 ms | `edge_p0/test_results/e2e_latency.json` | print/cache 链路 |
| 目标 | <1500 ms | `LinkAble_开发文档.md` | 检测到播报响应目标 |

必须限定：

```text
当前端到端延迟是本机 print/cache 链路，不等于真实喇叭 mp3 播放延迟。
真实音频设备、Jetson 音频输出和现场环境仍需单独实测。
```

### 3.6 8 视频本地闭环报告

| 指标 | 当前数字 | 证据路径 |
|---|---:|---|
| 视频数 | 8 | `edge_p0/runs/video_report/video_summary.json` |
| 总帧数 | 5,494 | `edge_p0/runs/video_report/video_summary.json` |
| 视频总时长 | 190.89 s | `edge_p0/runs/video_report/video_summary.json` |
| 检测总数 | 8,412 | `edge_p0/runs/video_report/video_summary.json` |
| 结构化事件总数 | 604 | `edge_p0/runs/video_report/video_summary.json` |
| 盲道占用事件 | 327 | `edge_p0/runs/video_report/video_summary.json` |
| 路面障碍事件 | 172 | `edge_p0/runs/video_report/video_summary.json` |
| 台阶事件 | 100 | `edge_p0/runs/video_report/video_summary.json` |
| 坡道事件 | 5 | `edge_p0/runs/video_report/video_summary.json` |
| 每分钟事件数 | 189.84 | `edge_p0/runs/video_report/video_summary.json` |

可视化与中间文件：

- `edge_p0/runs/video_report/video_report.json`
- `edge_p0/runs/video_report/video_summary.json`
- `edge_p0/runs/video_report/video_events.csv`
- `edge_p0/runs/video_report/dashboard.html`

答辩口径：

> 8 段本地测试视频共处理 5494 帧，产生 8412 次检测和 604 条结构化事件。系统不保存原始视频帧，只保留事件类型、置信度、方向、时间和来源帧等结构化信息，可用于统计盲道占用、路障高发和设施问题分布。

### 3.7 TTS 离线缓存与降级链

| 指标 | 当前数字 | 证据路径 | 说明 |
|---|---:|---|---|
| 模板总数 | 28 | `edge_p0/test_results/tts_cache/tts_cache_validation.json` | `ALL_TEMPLATE_TEXTS` |
| 缓存命中数 | 28 | `edge_p0/test_results/tts_cache/tts_cache_validation.json` | 28/28 |
| 离线播报 PASS | 28 | `edge_p0/test_results/tts_cache/tts_cache_validation.json` | 不需要在线 API |
| 在线 fallback 次数 | 0 | `edge_p0/test_results/tts_cache/tts_cache_validation.json` | 断网可用性证据 |
| 默认音色 | presenter_female | `edge_p0/test_results/tts_cache/tts_cache_validation.json` | 与项目规范一致 |
| 默认模型 | speech-2.8-hd | `edge_p0/test_results/tts_cache/tts_cache_validation.json` | MiniMax TTS |

答辩口径：

> 为避免现场网络和 TTS 服务波动影响演示，LinkAble 采用本地模板和缓存 mp3 优先策略。当前 28 条 P0 语音模板全部完成缓存命中，离线播报测试 28/28 PASS，在线 fallback 为 0。

### 3.8 自动化测试与本地稳定性

| 指标 | 当前数字 | 说明 |
|---|---:|---|
| 自动化测试 | 41 passed | 推送前本地执行 `PYTHONPATH=edge_p0/src python -m pytest -q tests` |
| 边界异常测试 | 已覆盖 | 坏图、空帧、断网缓存、双事件最高优先级、音频锁串行 |
| 本地闭环状态 | 已跑通 | 图片/视频输入、YOLO、EventBuilder、模板提示、print 播报、结构化记录、报告生成 |

---

## 4. 当前不能夸大的缺口

| 缺口 | 当前状态 | 答辩处理 |
|---|---|---|
| Jetson 实测 FPS | 尚未形成正式记录 | 说“GPU baseline 与本机 CPU 已测，Jetson 实测下一步补齐” |
| Jetson 单帧推理延迟 | 尚未形成正式记录 | 不把 RTX 4070 SUPER 数字说成 Jetson 数字 |
| 真实 mp3 播放端到端延迟 | print/cache 链路已测，真实音频设备未测 | 说明本地缓存已验证，真实播放器延迟需上设备测试 |
| A-D 点位演示成功次数 | 尚无正式现场统计 | 不说“现场已完成”，说“本地 8 视频闭环完成，现场点位待录制” |
| P0 场景识别成功率 | 有 mAP/Recall，没有现场成功率 | 区分模型指标和系统演示成功率 |
| 实时 AI 智能体 | 当前不是主链路 | 大模型只做事后报告增强，不进入实时安全链路 |

---

## 5. PPT 创新页建议

标题：

```text
创新点：面向无障碍治理的边缘感知事件闭环
```

三条正文：

1. **人行无障碍空间感知迁移**

   借鉴自动驾驶边缘感知思想，将检测对象从机动车道迁移到盲道、台阶、坡道和路面临时障碍，服务校园最后 20 米无障碍通行场景。

2. **从检测框到稳定事件的 EventBuilder**

   通过连续帧确认、去抖、冷却和优先级仲裁，将高频检测结果转化为低信息密度、可播报的事件。三视频消融实验中播报次数从 1124 次降至 241 次，减少 78.6%。

3. **结构化事件驱动的无障碍治理数据**

   系统不保存原始视频，仅沉淀时间、类型、置信度、方向、位置等结构化事件。8 段测试视频产生 604 条事件，可用于统计盲道占用、障碍高发点和设施整改效果。

底部工程保障：

```text
本地模板语音与 TTS 缓存降级链保证断网可用：28/28 模板缓存命中，在线 fallback 为 0。
```

---

## 6. 论文创新点写法

可直接使用以下表述：

```text
本文的主要创新点包括：

1. 提出一种面向人行无障碍空间的边缘感知原型架构，将自动驾驶中的环境感知思想迁移到盲道占用、台阶、坡道和路面障碍等校园最后 20 米场景。

2. 设计 EventBuilder 事件聚合机制，将连续帧目标检测结果转化为稳定、低频、可播报的通行事件。消融实验表明，该机制可将三段测试视频中的播报次数从 1124 次降低至 241 次，重复提示减少 78.6%。

3. 构建结构化无障碍事件记录与离线报告链路，在不保存原始视频帧的前提下沉淀事件类型、时间、方向、置信度等数据，为校园和社区无障碍设施治理提供可量化依据。
```

---

## 7. 推荐答辩话术

### 7.1 30 秒版本

> LinkAble 的核心创新不是再做一个语音提醒设备，而是把自动驾驶式边缘感知迁移到人行无障碍空间。系统在本地识别盲道占用、台阶、坡道和路面障碍，通过 EventBuilder 把连续帧检测结果转成稳定事件，再用本地语音模板播报，并沉淀结构化事件数据。这样既能辅助弱势行人听懂路，也能让管理方看到盲道占用和障碍高发等隐性问题。

### 7.2 量化证据版本

> 当前模型 baseline 的 mAP50 为 0.8264，Precision 为 0.8774，Recall 为 0.7684。GPU 环境下推理达到 108.3 FPS，平均单帧 9.23ms；本机 CPU 环境下平均单帧 33.80ms。EventBuilder 消融实验显示，三段视频中的播报次数从 1124 次降低到 241 次，重复提示减少 78.6%。在 8 段本地测试视频中，系统处理了 5494 帧，产生 8412 次检测和 604 条结构化事件。TTS 侧 28 条模板全部完成缓存命中，断网情况下不需要在线合成。

### 7.3 边界说明版本

> 目前本地 P0 闭环已经跑通，但我们不会把 PC/GPU 数字包装成 Jetson 现场实测。Jetson FPS、真实音频播放延迟和 A-D 点位演示成功率还需要上目标设备和现场路线补齐。实时安全链路也没有依赖大模型，大模型只用于 P1 的事后治理报告增强。

---

## 8. 不建议主打的方向

### 8.1 不主打“成熟导盲设备”

不能说：

- 已经可以替代盲杖或导盲犬。
- 可以保证视障用户完全安全。
- 已经是成熟量产穿戴设备。

建议说：

```text
当前是校园最后 20 米无障碍辅助通行原型系统。
```

### 8.2 不主打“实时 AI 智能体”

不能说：

- 系统支持实时大模型对话。
- 大模型在实时安全链路中做决策。
- 系统已经是完整智能体平台。

建议说：

```text
实时链路走本地 YOLO + EventBuilder + 模板语音；
云端大模型只做 P1 事后报告增强，避免 3-5 秒延迟影响通行提示。
```

### 8.3 不只讲“边缘实时”

只讲边缘实时容易被问：

```text
YOLO 跑在边缘端有什么新？
```

更好的回答：

```text
边缘实时是基础能力，核心价值是把无障碍环境中的隐性问题转化成可治理的结构化事件数据。
```

---

## 9. 最终主打口径

正式材料建议统一写成：

> LinkAble 是一套基于边缘智能的无障碍辅助通行原型系统。它借鉴自动驾驶中的环境感知与边缘推理思想，将感知对象从机动车道迁移到人行无障碍空间，通过 YOLO 识别盲道占用、台阶、坡道和路面障碍；通过 EventBuilder 将连续帧检测转化为稳定事件；通过本地语音模板完成低延迟播报；并沉淀结构化事件数据，为校园和社区无障碍设施治理提供可量化依据。

一句话记忆：

```text
LinkAble 不只提醒一个人绕开障碍，更让管理者知道哪里长期存在障碍。
```
