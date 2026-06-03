from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Mapping


LABEL_CN = {
    "blind_road_occupied": "盲道占用",
    "stairs": "台阶",
    "ramp": "坡道",
    "road_obstacle": "路面障碍",
}


def summarize_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    event_list = list(events)
    by_type = Counter(str(event.get("type", "")) for event in event_list if event.get("type"))
    by_location = Counter(str(event.get("location", "")) for event in event_list if event.get("location"))
    by_day = Counter(str(event.get("timestamp", ""))[:10] for event in event_list if event.get("timestamp"))
    hour_dist = Counter()
    confidence_sum = 0.0
    confidence_count = 0

    for event in event_list:
        timestamp = str(event.get("timestamp", ""))
        if len(timestamp) >= 13 and timestamp[11:13].isdigit():
            hour_dist[int(timestamp[11:13])] += 1

        confidence = event.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_sum += float(confidence)
            confidence_count += 1

    peak_hour = hour_dist.most_common(1)[0] if hour_dist else (None, 0)
    worst_location = by_location.most_common(1)[0] if by_location else ("未记录", 0)

    return {
        "total_events": len(event_list),
        "by_type": dict(by_type),
        "by_location": dict(by_location),
        "by_day": dict(by_day),
        "avg_confidence": round(confidence_sum / confidence_count, 3) if confidence_count else 0.0,
        "peak_hour": {"hour": peak_hour[0], "count": peak_hour[1]},
        "worst_location": {"name": worst_location[0], "count": worst_location[1]},
    }


def generate_weekly_report(
    events: Iterable[Mapping[str, Any]],
    *,
    period: str,
    node_name: str = "校园最后 20 米演示路线（A-D 点位）",
    generated_at: datetime | None = None,
) -> str:
    stats = summarize_events(events)
    generated = generated_at or datetime.now()
    total_events = int(stats["total_events"])
    avg_confidence = float(stats["avg_confidence"])

    type_rows = _table_rows_by_type(stats["by_type"], total_events)
    location_rows = _table_rows(stats["by_location"], total_events)
    day_rows = _table_rows(stats["by_day"], total_events)

    return f"""# LinkAble 周度无障碍环境监测报告

> **监测周期**：{period}
> **生成时间**：{generated.strftime("%Y-%m-%d %H:%M")}
> **监测节点**：{node_name}
> **数据来源**：LinkAble 边缘智能原型系统结构化事件

---

## 1. 报告概览

本周期共记录 **{total_events}** 起无障碍环境事件，事件平均置信度为 **{avg_confidence:.1%}**。
事件最集中区域为 **{stats["worst_location"]["name"]}**，共 {stats["worst_location"]["count"]} 起。

---

## 2. 事件分布

### 2.1 按事件类型

| 类型 | 英文标识 | 事件数 | 占比 |
|------|----------|--------|------|
{type_rows}

### 2.2 按监测点位

| 点位 | 事件数 | 占比 |
|------|--------|------|
{location_rows}

### 2.3 按日期

| 日期 | 事件数 | 占比 |
|------|--------|------|
{day_rows}

---

## 3. 治理建议

- 优先处理事件最集中点位：{stats["worst_location"]["name"]}。
- 高频时段为 {stats["peak_hour"]["hour"]}:00，建议在该时段加强巡查。
- 保持事件周报制度，用结构化记录跟踪治理前后变化。
"""


def _table_rows_by_type(values: Mapping[str, int], total: int) -> str:
    if not values:
        return "| 无记录 | - | 0 | 0.0% |"

    rows = []
    for label, count in sorted(values.items(), key=lambda item: -item[1]):
        label_cn = LABEL_CN.get(label, label)
        rows.append(f"| {label_cn} | {label} | {count} | {_pct(count, total)} |")
    return "\n".join(rows)


def _table_rows(values: Mapping[str, int], total: int) -> str:
    if not values:
        return "| 无记录 | 0 | 0.0% |"

    rows = []
    for label, count in sorted(values.items(), key=lambda item: -item[1]):
        rows.append(f"| {label} | {count} | {_pct(count, total)} |")
    return "\n".join(rows)


def _pct(count: int, total: int) -> str:
    return f"{(count / total * 100):.1f}%" if total > 0 else "0.0%"
