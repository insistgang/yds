# LinkAble 核心文档导航

> 更新日期：2026-05-24
> 用途：给论文、PPT、展架、报名表、商业计划书和视频脚本做文案入口。

## 先看这 7 个

| 顺序 | 文档 | 位置 | 用途 |
|---:|---|---|---|
| 1 | 项目协作与范围总约束 | `agents.md` | P0/P1/P2 边界、统一口径、禁止夸大表述 |
| 2 | 项目总览 | `README.md` | 一页看懂项目定位、阶段、交付物 |
| 3 | 当前 TODO | `TODO.md` | 现在真正要做什么、哪些已完成 |
| 4 | 西门子杯适配说明 | `SIEMENS_CUP_FREE_EXPLORATION.md` | 自由探索组报名口径、产品化材料清单 |
| 5 | 设备选型说明 | `LinkAble_设备选型说明.md` | 解释为什么使用 Jetson / 边缘 AI 设备 |
| 6 | 边缘端技术 README | `edge_p0/README.md` | P0 技术链路、演示命令、当前工程状态 |
| 7 | PRD 定稿 | `LinkAble_PRD_v2.2.md` | 产品需求、目标用户、场景和功能边界 |

## 写不同材料时看哪里

### 报名表 / 项目简介

优先看：

- `README.md`
- `agents.md`
- `LinkAble_设备选型说明.md`
- `LinkAble_PRD_v2.2.md`
- `SIEMENS_CUP_FREE_EXPLORATION.md`

可直接提炼的关键词：

- 基于边缘智能
- 无障碍辅助通行
- 校园最后 20 米
- AI 边缘感知节点
- 盲道占用、台阶、坡道、路面障碍
- 本地语音播报
- 结构化事件记录
- 无障碍环境治理

### 研电赛论文 / 技术方案

优先看：

- `LinkAble_PRD_v2.2.md`
- `docs/product/LinkAble_PRD_V2.2_深度解读.md`
- `agents.md`
- `LinkAble_设备选型说明.md`
- `edge_p0/README.md`
- `edge_p0/ARCHITECTURE.txt`

写作重点：

- 不把项目写成成熟导盲设备。
- 不把 Jetson 写成炫硬件，而是写成边缘智能感知节点。
- 强调从机动车道感知到人行无障碍空间感知的迁移。
- 强调本地实时链路：检测、事件聚合、模板提示、播报、记录。
- 强调数据治理价值：让隐性无障碍问题变成可量化事件。

### 西门子杯自由探索 / 产品方案

优先看：

- `SIEMENS_CUP_FREE_EXPLORATION.md`
- `README.md`
- `edge_p0/ARCHITECTURE.txt`
- `edge_p0/README.md`

写作重点：

- 把 LinkAble 写成“AI 边缘硬件产品原型”，不是纯软件。
- 补充产品外观、结构设计、BOM 成本、部署方式、商业模式。
- 目标客户写校园、社区、医院、地铁站、园区管理方。

### PPT / 答辩

优先看：

- `README.md`
- `TODO.md`
- `edge_p0/docs/demo_route_plan.md`
- `edge_p0/docs/demo_video_shotlist.md`

建议页面结构：

1. 痛点：无障碍空间的隐性障碍难发现、难治理。
2. 方案：边缘智能感知节点。
3. 技术：YOLO 检测、事件去抖、语音播报、结构化记录。
4. 数据：自采+公共融合集、类别分布、评估指标。
5. 演示：校园最后 20 米 A-D 点位。
6. 价值：弱势行人辅助 + 管理方治理数据。

### 展架 / 海报

优先看：

- `README.md`
- `SIEMENS_CUP_FREE_EXPLORATION.md`
- `edge_p0/docs/demo_video_shotlist.md`
- `edge_p0/docs/demo_route_plan.md`

建议只放这些信息：

- 一句话定位
- 系统流程图
- P0 四类识别场景
- 演示路线 A-D 点位
- 核心指标：FPS、延迟、mAP/Recall（等 baseline 出来后填写）
- 应用价值：辅助通行 + 数据治理

### 演示视频脚本

优先看：

- `edge_p0/docs/demo_video_shotlist.md`
- `edge_p0/docs/demo_route_plan.md`
- `edge_p0/docs/field_capture_checklist.md`

必须出现：

- 输入画面
- 检测框、类别、置信度
- 事件日志
- 语音播报
- Jetson/边缘设备特写
- 匿名处理：无校名、人脸、车牌

### 数据与实验

优先看：

- `edge_p0/configs/p0_data.yaml`
- `docs/evaluation/DATA_SOURCE_LEDGER.md`
- `docs/evaluation/PERFORMANCE_METRICS.md`
- `edge_p0/docs/mangdaojiance_integration.md`
- `edge_p0/docs/field_capture_checklist.md`
- `edge_p0/scripts/visualize_p0_dataset.py`
- `edge_p0/scripts/select_poster_samples.py`

当前要补的文字材料：

- 数据来源台账：自采、公共、购买/第三方、AI 补充。
- train/val/test 划分说明。
- 测试集自采真实场景占比。
- 每类标注抽检结论。
- 类别不平衡处理说明。

## 需要谨慎使用的文档

旧状态、重复快速上手和乱码文档已经删除。写正式材料时只按本索引中的活文档取材。

## 当前统一口径

推荐写法：

> LinkAble 是一套基于边缘智能的无障碍辅助通行原型系统。它借鉴自动驾驶中的环境感知与边缘推理思想，把感知对象从机动车道迁移到人行无障碍空间，通过 YOLO 识别盲道占用、台阶、坡道和路面障碍，提供本地语音提示，并沉淀结构化事件数据支撑校园、社区等场景的无障碍治理。

禁止写法：

- 已经做出了成熟导盲设备
- 可以替代盲杖或导盲犬
- 可以保证视障用户完全安全
- 核心是 App、云平台或志愿者系统
- 支持实时大模型对话
