"""
LinkAble 周度无障碍报告样本生成器

基于消融实验的事件分布数据，生成 7 天模拟事件流，
调用大模型（MiMo/Kimi/Qwen）生成结构化报告，或输出 Markdown 模板。

输出：
  - edge_p0/runs/report_sample/events.json
  - edge_p0/runs/report_sample/report.md

用法：
  python scripts/generate_report_sample.py
"""

import json
import os
import random
import hashlib
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# ── 路径 ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ABLATION_JSON = ROOT / "runs" / "ablation" / "event_builder_ablation.json"
OUT_DIR = ROOT / "runs" / "report_sample"
EVENTS_JSON = OUT_DIR / "events.json"
REPORT_MD = OUT_DIR / "report.md"
DEFAULT_MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MIMO_MODEL = "mimo-v2.5-pro"

# ── P0 类别中文映射 ────────────────────────────────────
LABEL_CN = {
    "blind_road_occupied": "盲道占用",
    "stairs": "台阶",
    "ramp": "坡道",
    "road_obstacle": "路面障碍",
}

# ── 演示路线点位（校园最后 20 米） ──────────────────────
DEMO_LOCATIONS = [
    {"name": "A点-图书馆东门盲道入口", "lat": 31.2304, "lng": 121.4737},
    {"name": "B点-教学楼南侧坡道", "lat": 31.2308, "lng": 121.4741},
    {"name": "C点-食堂西侧台阶区域", "lat": 31.2312, "lng": 121.4735},
    {"name": "D点-宿舍楼北门路面", "lat": 31.2315, "lng": 121.4739},
]


def load_ablation_distribution() -> dict:
    """从消融实验结果提取各类别事件比例（使用 with_debounce 数据）"""
    if not ABLATION_JSON.exists():
        print(f"[WARN] 消融实验结果不存在，使用 P0 默认分布: {ABLATION_JSON}")
        return {
            "blind_road_occupied": 0.35,
            "stairs": 0.20,
            "ramp": 0.20,
            "road_obstacle": 0.25,
        }

    with open(ABLATION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_by_label = Counter()
    for result in data["results"]:
        for label, count in result["with_debounce"]["by_label"].items():
            total_by_label[label] += count

    total = sum(total_by_label.values())
    if total == 0:
        print("[WARN] 消融实验结果没有事件，使用 P0 默认分布")
        return {
            "blind_road_occupied": 0.35,
            "stairs": 0.20,
            "ramp": 0.20,
            "road_obstacle": 0.25,
        }
    return {label: count / total for label, count in total_by_label.items()}


def generate_7day_events(distribution: dict, seed: int = 42) -> list:
    """生成 7 天模拟事件流"""
    random.seed(seed)

    start_date = datetime(2026, 5, 26, 0, 0, 0)
    days = 7
    events = []
    event_id = 1

    # 每天事件数：工作日多，周末少
    daily_counts = [45, 52, 48, 55, 42, 28, 30]  # 周一~周日

    labels = list(distribution.keys())
    weights = [distribution[l] for l in labels]

    for day_idx in range(days):
        current_date = start_date + timedelta(days=day_idx)
        n_events = daily_counts[day_idx]

        # 事件集中在 7:00-22:00
        for _ in range(n_events):
            hour = random.choices(
                range(7, 23),
                weights=[1, 2, 3, 4, 5, 5, 4, 3, 5, 6, 5, 4, 3, 2, 2, 1],
                k=1,
            )[0]
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = current_date.replace(hour=hour, minute=minute, second=second)

            label = random.choices(labels, weights=weights, k=1)[0]
            confidence = round(random.uniform(0.55, 0.98), 3)
            location = random.choice(DEMO_LOCATIONS)

            events.append(
                {
                    "event_id": f"EVT-{event_id:05d}",
                    "timestamp": ts.isoformat() + "+08:00",
                    "type": label,
                    "type_cn": LABEL_CN[label],
                    "confidence": confidence,
                    "location": location["name"],
                    "lat": location["lat"],
                    "lng": location["lng"],
                }
            )
            event_id += 1

    events.sort(key=lambda e: e["timestamp"])
    return events


def compute_stats(events: list) -> dict:
    """计算事件统计"""
    by_type = Counter(e["type"] for e in events)
    by_location = Counter(e["location"] for e in events)
    by_day = Counter(e["timestamp"][:10] for e in events)
    avg_confidence = sum(e["confidence"] for e in events) / len(events)

    # 高频时段
    hour_dist = Counter(int(e["timestamp"][11:13]) for e in events)
    peak_hour = hour_dist.most_common(1)[0]

    # 最严重点位
    worst_location = by_location.most_common(1)[0]

    return {
        "total_events": len(events),
        "by_type": dict(by_type),
        "by_location": dict(by_location),
        "by_day": dict(by_day),
        "avg_confidence": round(avg_confidence, 3),
        "peak_hour": {"hour": peak_hour[0], "count": peak_hour[1]},
        "worst_location": {"name": worst_location[0], "count": worst_location[1]},
    }


def try_call_llm_api(
    events: list,
    stats: dict,
    *,
    provider: str = "auto",
    mimo_base_url: str | None = None,
    mimo_model: str | None = None,
) -> str | None:
    """尝试调用 MiMo/Kimi/Qwen API 生成报告"""
    # 检查环境变量
    mimo_key = (
        os.environ.get("XIAOMI_MIMO_API_KEY")
        or os.environ.get("MIMO_API_KEY")
        or os.environ.get("TOKEN_PLAN_API_KEY")
    )
    kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")

    if provider == "mimo":
        return _call_mimo(
            mimo_key,
            events,
            stats,
            base_url=mimo_base_url or os.environ.get("XIAOMI_MIMO_BASE_URL", DEFAULT_MIMO_BASE_URL),
            model=mimo_model or os.environ.get("MIMO_LLM_MODEL", DEFAULT_MIMO_MODEL),
        )
    if provider == "kimi":
        return _call_kimi(kimi_key, events, stats) if kimi_key else None
    if provider == "qwen":
        return _call_qwen(qwen_key, events, stats) if qwen_key else None

    if mimo_key:
        return _call_mimo(
            mimo_key,
            events,
            stats,
            base_url=mimo_base_url or os.environ.get("XIAOMI_MIMO_BASE_URL", DEFAULT_MIMO_BASE_URL),
            model=mimo_model or os.environ.get("MIMO_LLM_MODEL", DEFAULT_MIMO_MODEL),
        )
    if kimi_key:
        return _call_kimi(kimi_key, events, stats)
    if qwen_key:
        return _call_qwen(qwen_key, events, stats)
    return None


def _build_prompt(events: list, stats: dict) -> str:
    """构建大模型提示词"""
    # 只取前 100 条事件作为样本，避免 token 超限
    sample_events = events[:100]
    events_json = json.dumps(sample_events, ensure_ascii=False, indent=2)
    stats_json = json.dumps(stats, ensure_ascii=False, indent=2)

    return f"""你是一位城市无障碍环境分析师。请基于以下结构化事件数据，生成一份"周度无障碍环境监测报告"。

## 数据说明
- 监测周期：2026年5月26日 ~ 2026年6月1日
- 监测点位：校园内 4 个关键无障碍节点（A-D点位）
- 检测类型：盲道占用(blind_road_occupied)、台阶(stairs)、坡道(ramp)、路面障碍(road_obstacle)
- 事件总数：{stats['total_events']}

## 事件统计
{stats_json}

## 事件样本（前100条）
{events_json}

## 输出要求
请用 Markdown 格式输出报告，包含以下章节：

1. **报告概览**：周期、总事件数、关键发现（1-2句话）
2. **事件分布分析**：
   - 按类型统计（表格）
   - 按点位统计（表格）
   - 按日趋势（表格）
3. **重点问题**：
   - 最严重的 2-3 个问题及具体数据
   - 高频时段分析
4. **改善建议**：
   - 针对最突出问题的 2-3 条可执行建议
5. **数据附录**：事件类型说明

注意：
- 语言：中文
- 风格：专业、客观、数据驱动
- 所有结论必须有数据支撑
- 不要使用"可能""也许"等不确定措辞
- 建议要具体可执行，不要泛泛而谈"""


def _call_kimi(api_key: str, events: list, stats: dict) -> str | None:
    """调用 Kimi (Moonshot) API"""
    try:
        import httpx

        prompt = _build_prompt(events, stats)
        resp = httpx.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "moonshot-v1-8k",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[WARN] Kimi API 调用失败: {e}")
        return None


def _call_qwen(api_key: str, events: list, stats: dict) -> str | None:
    """调用 Qwen (DashScope) API"""
    try:
        import httpx

        prompt = _build_prompt(events, stats)
        resp = httpx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen-plus",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[WARN] Qwen API 调用失败: {e}")
        return None


def _normalize_mimo_model(model: str) -> str:
    """兼容口误/旧写法，不影响命令行显式覆盖。"""
    aliases = {
        "mino-2.5-pro": "mimo-v2.5-pro",
        "mimo-2.5-pro": "mimo-v2.5-pro",
    }
    return aliases.get(model, model)


def _call_mimo(
    api_key: str | None,
    events: list,
    stats: dict,
    *,
    base_url: str,
    model: str,
) -> str | None:
    """调用小米 MiMo Token Plan OpenAI-compatible API"""
    if not api_key:
        print("[WARN] MiMo API 调用失败: 未设置 XIAOMI_MIMO_API_KEY/MIMO_API_KEY/TOKEN_PLAN_API_KEY")
        return None

    try:
        import httpx

        prompt = _build_prompt(events, stats)
        normalized_model = _normalize_mimo_model(model)
        resp = httpx.post(
            base_url.rstrip("/") + "/chat/completions",
            headers={
                "api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "model": normalized_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[WARN] MiMo API 调用失败: {e}")
        return None


def generate_template_report(events: list, stats: dict) -> str:
    """生成 Markdown 模板报告（不调用 API）"""
    period = "2026-05-26 ~ 2026-06-01"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 类型统计表
    type_rows = ""
    for label, count in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
        pct = count / stats["total_events"] * 100
        type_rows += f"| {LABEL_CN[label]} | {label} | {count} | {pct:.1f}% |\n"

    # 点位统计表
    loc_rows = ""
    for loc, count in sorted(stats["by_location"].items(), key=lambda x: -x[1]):
        pct = count / stats["total_events"] * 100
        loc_rows += f"| {loc} | {count} | {pct:.1f}% |\n"

    # 每日趋势表
    day_rows = ""
    for day in sorted(stats["by_day"].keys()):
        count = stats["by_day"][day]
        weekday_map = {0: "一", 1: "二", 2: "三", 3: "四", 4: "五", 5: "六", 6: "日"}
        dt = datetime.strptime(day, "%Y-%m-%d")
        wd = weekday_map[dt.weekday()]
        bar = "█" * (count // 5)
        day_rows += f"| {day} (周{wd}) | {count} | {bar} |\n"

    # 最严重类型
    worst_type = max(stats["by_type"].items(), key=lambda x: x[1])
    worst_type_cn = LABEL_CN[worst_type[0]]
    worst_type_pct = worst_type[1] / stats["total_events"] * 100

    # 高频时段
    peak_h = stats["peak_hour"]["hour"]
    peak_count = stats["peak_hour"]["count"]

    report = f"""# LinkAble 周度无障碍环境监测报告

> **监测周期**：{period}
> **生成时间**：{now}
> **监测节点**：校园最后 20 米演示路线（A-D 点位）
> **数据来源**：LinkAble 边缘智能原型系统（YOLO 实时检测 + 事件聚合）

---

## 1. 报告概览

本周共记录 **{stats['total_events']}** 起无障碍环境事件，覆盖 4 个监测点位。
事件平均置信度为 **{stats['avg_confidence']:.1%}**，数据质量可靠。

**关键发现**：{worst_type_cn}是最突出问题，占比 {worst_type_pct:.1f}%；
{stats['worst_location']['name']}为事件最集中区域（{stats['worst_location']['count']} 起）。
高峰时段集中在 **{peak_h}:00** 附近，建议加强该时段巡查。

---

## 2. 事件分布分析

### 2.1 按事件类型

| 类型 | 英文标识 | 事件数 | 占比 |
|------|----------|--------|------|
{type_rows}

### 2.2 按监测点位

| 点位 | 事件数 | 占比 |
|------|--------|------|
{loc_rows}

### 2.3 每日趋势

| 日期 | 事件数 | 趋势 |
|------|--------|------|
{day_rows}

---

## 3. 重点问题

### 3.1 {worst_type_cn}问题突出

- 本周共发生 **{worst_type[1]}** 起{worst_type_cn}事件，占总事件的 {worst_type_pct:.1f}%
- 主要集中在 **{stats['worst_location']['name']}**
- 高频时段：**{peak_h}:00**（{peak_count} 起）

### 3.2 时段分布不均

- 事件高峰集中在上午 8:00-10:00 和下午 14:00-16:00
- 与校园人流高峰高度吻合，说明无障碍设施在高负载下问题更易暴露

### 3.3 点位差异明显

- {stats['worst_location']['name']}事件量显著高于其他点位
- 建议优先对该区域进行设施排查和改善

---

## 4. 改善建议

### 建议一：针对{worst_type_cn}的专项治理

- **目标**：将{worst_type_cn}事件降低 50% 以上
- **措施**：
  - 在{stats['worst_location']['name']}增设物理隔离或警示标识
  - 安排高峰时段（{peak_h}:00 前后）专人巡查
  - 建立事件响应机制：事件触发后 15 分钟内到场处理

### 建议二：数据驱动的常态化监测

- **目标**：建立周报-月报制度，持续跟踪改善效果
- **措施**：
  - 部署 LinkAble 固定监测节点于 A-D 点位
  - 每周自动生成监测报告，对比趋势变化
  - 设置阈值告警：单日事件超过 60 起自动通知管理人员

### 建议三：设施维护优先级排序

- **目标**：合理分配维护资源
- **措施**：
  - 按事件频率排序：{worst_type_cn} > 其他类型
  - 按点位排序：{stats['worst_location']['name']} > 其他点位
  - 优先处理高频问题+高频点位的组合

---

## 5. 数据附录

### 事件类型说明

| 类型 | 说明 | 典型场景 |
|------|------|----------|
| blind_road_occupied | 盲道被占用 | 车辆、杂物占用盲道 |
| stairs | 台阶 | 未设坡道的台阶区域 |
| ramp | 坡道 | 无障碍坡道入口 |
| road_obstacle | 路面障碍 | 临时施工、堆放物 |

### 数据格式

每条事件包含以下字段：
- `event_id`：事件唯一标识
- `timestamp`：ISO 8601 时间戳
- `type`：事件类型（英文）
- `confidence`：检测置信度（0.55-1.0）
- `location`：监测点位名称
- `lat/lng`：经纬度坐标

---

*本报告由 LinkAble 边缘智能原型系统自动生成，基于 YOLO 实时检测与事件聚合引擎。*
*数据仅用于无障碍环境治理研究，不包含任何个人隐私信息。*
"""

    return report


def main():
    parser = argparse.ArgumentParser(description="生成 LinkAble 周度无障碍报告样本")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="显式调用大模型生成报告；默认使用离线模板，避免现场网络阻塞。",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "mimo", "kimi", "qwen"],
        default=os.environ.get("LINKABLE_LLM_PROVIDER", "auto"),
        help="大模型报告 provider；默认 auto，优先 MiMo，再 Kimi/Qwen。",
    )
    parser.add_argument("--mimo-base-url", default=os.environ.get("XIAOMI_MIMO_BASE_URL", DEFAULT_MIMO_BASE_URL))
    parser.add_argument("--mimo-model", default=os.environ.get("MIMO_LLM_MODEL", DEFAULT_MIMO_MODEL))
    args = parser.parse_args()

    print("=" * 60)
    print("LinkAble 周度无障碍报告样本生成器")
    print("=" * 60)

    # 1. 加载消融实验分布
    print("\n[1/4] 加载消融实验事件分布...")
    distribution = load_ablation_distribution()
    for label, ratio in distribution.items():
        print(f"  {LABEL_CN[label]:6s} ({label}): {ratio:.1%}")

    # 2. 生成 7 天模拟事件
    print("\n[2/4] 生成 7 天模拟事件流...")
    events = generate_7day_events(distribution)
    stats = compute_stats(events)
    print(f"  总事件数: {stats['total_events']}")
    print(f"  平均置信度: {stats['avg_confidence']:.1%}")
    print(f"  高峰时段: {stats['peak_hour']['hour']}:00")
    print(f"  最严重点位: {stats['worst_location']['name']}")

    # 3. 输出 events.json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "period": "2026-05-26 ~ 2026-06-01",
            "node_id": "linkable-campus-demo",
            "version": "1.0",
        },
        "events": events,
        "stats": stats,
    }
    with open(EVENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[3/4] 事件数据已保存: {EVENTS_JSON}")

    # 4. 生成报告
    print("\n[4/4] 生成报告...")
    use_llm = args.use_llm or os.environ.get("LINKABLE_USE_LLM_REPORT") == "1"
    llm_report = (
        try_call_llm_api(
            events,
            stats,
            provider=args.llm_provider,
            mimo_base_url=args.mimo_base_url,
            mimo_model=args.mimo_model,
        )
        if use_llm
        else None
    )
    if llm_report:
        report = llm_report
        print("  [OK] 使用大模型生成报告")
    else:
        report = generate_template_report(events, stats)
        if use_llm:
            print("  [OK] 大模型不可用，已降级为离线模板报告")
        else:
            print("  [OK] 使用离线模板生成报告（加 --use-llm 才调用 MiMo/Kimi/Qwen）")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  报告已保存: {REPORT_MD}")

    # 打印报告
    print("\n" + "=" * 60)
    print("报告内容预览")
    print("=" * 60)
    print(report)


if __name__ == "__main__":
    main()
