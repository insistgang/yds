#!/usr/bin/env python3
"""Analyze local LinkAble test videos and build JSON reports plus a static dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from linkable_edge.detector import YoloDetector, YoloDetectorConfig  # noqa: E402
from linkable_edge.event_builder import EventBuilder, EventBuilderConfig  # noqa: E402
from linkable_edge.inputs import VideoFileSource  # noqa: E402
from linkable_edge.models import DetectionEvent  # noqa: E402
from linkable_edge.semantics import render_event_text  # noqa: E402


LABEL_CN = {
    "blind_road_occupied": "盲道占用",
    "stairs": "台阶",
    "ramp": "坡道",
    "road_obstacle": "路面障碍",
}

DEFAULT_OUTPUT_DIR = ROOT / "runs" / "video_report"


@dataclass
class VideoAnalysis:
    video_name: str
    video_path: str
    fps: float
    frame_size: tuple[int, int]
    total_frames: int = 0
    total_detections: int = 0
    total_events: int = 0
    duration_sec: float = 0.0
    detections_by_type: Counter[str] = field(default_factory=Counter)
    events_by_type: Counter[str] = field(default_factory=Counter)
    events_by_direction: Counter[str] = field(default_factory=Counter)
    timeline_by_10s: Counter[str] = field(default_factory=Counter)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "video_name": self.video_name,
            "video_path": self.video_path,
            "fps": round(self.fps, 3),
            "frame_size": list(self.frame_size),
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "total_detections": self.total_detections,
            "total_events": self.total_events,
            "detections_by_type": dict(self.detections_by_type),
            "events_by_type": dict(self.events_by_type),
            "events_by_direction": dict(self.events_by_direction),
            "timeline_by_10s": dict(sorted(self.timeline_by_10s.items(), key=lambda item: int(item[0]))),
            "events_per_minute": round(self.total_events / max(self.duration_sec / 60, 1e-6), 2),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSON report and static dashboard for local test videos.")
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "test_videos")
    parser.add_argument("--video-glob", default="*.mp4")
    parser.add_argument("--model", default=str(ROOT / "best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--event-conf", type=float, default=0.55)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--cooldown-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap per video for smoke runs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_paths = sorted(args.videos_dir.glob(args.video_glob), key=natural_video_sort_key)
    if not video_paths:
        print(f"[ERROR] no videos found: {args.videos_dir / args.video_glob}", file=sys.stderr)
        return 1

    detector = YoloDetector(
        YoloDetectorConfig(
            model_path=args.model,
            image_size=args.imgsz,
            confidence_threshold=args.conf,
            model_type="p0_custom",
        )
    )
    event_config = EventBuilderConfig(
        confidence_threshold=args.event_conf,
        min_consecutive_frames=args.min_frames,
        emit_cooldown_frames=args.cooldown_frames,
    )

    started = time.perf_counter()
    analyses: list[VideoAnalysis] = []
    for index, video_path in enumerate(video_paths, 1):
        print(f"[{index}/{len(video_paths)}] analyzing {video_path.name}")
        analysis = analyze_video(video_path, detector, event_config, max_frames=args.max_frames)
        analyses.append(analysis)
        print(
            f"  frames={analysis.total_frames} detections={analysis.total_detections} "
            f"events={analysis.total_events} duration={analysis.duration_sec:.1f}s"
        )

    elapsed_sec = time.perf_counter() - started
    report = build_report(analyses, args=args, elapsed_sec=elapsed_sec)
    write_outputs(report, args.output_dir)
    print(f"[RESULT] report dir: {args.output_dir}")
    print(f"[RESULT] videos={report['summary']['video_count']} events={report['summary']['total_events']}")
    return 0


def natural_video_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem) if path.stem.isdigit() else 9999, path.name)


def analyze_video(
    video_path: Path,
    detector: YoloDetector,
    event_config: EventBuilderConfig,
    max_frames: int | None = None,
) -> VideoAnalysis:
    source = VideoFileSource(video_path)
    if not source.open():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = source.fps or 30.0
    analysis = VideoAnalysis(
        video_name=video_path.stem,
        video_path=project_relative(video_path),
        fps=fps,
        frame_size=source.frame_size,
    )
    builder = EventBuilder(config=event_config)

    try:
        frame_id = 0
        while max_frames is None or frame_id < max_frames:
            ok, image = source.read()
            if not ok or image is None:
                break
            frame_id += 1
            frame = detector.predict_frame(image, frame_id=frame_id)
            analysis.total_frames = frame_id
            analysis.total_detections += len(frame.detections)

            for detection in frame.detections:
                analysis.detections_by_type[detection.label] += 1

            for event in builder.process_frame(frame):
                add_event(analysis, event, fps=fps)
    finally:
        source.release()

    analysis.duration_sec = analysis.total_frames / max(fps, 1e-6)
    return analysis


def add_event(analysis: VideoAnalysis, event: DetectionEvent, *, fps: float) -> None:
    frame_id = max(event.source_frame_ids[-1] if event.source_frame_ids else 0, 0)
    video_time_sec = frame_id / max(fps, 1e-6)
    timeline_bucket = int(video_time_sec // 10) * 10
    event_type_cn = LABEL_CN.get(event.label, event.label)
    event_id = f"{analysis.video_name}-{event.event_id}"
    payload = event.to_api_dict()
    payload.update(
        {
            "event_id": event_id,
            "video": analysis.video_name,
            "video_path": analysis.video_path,
            "type_cn": event_type_cn,
            "message": render_event_text(event),
            "frame_id": frame_id,
            "video_time_sec": round(video_time_sec, 3),
            "time_bucket_10s": timeline_bucket,
        }
    )

    analysis.events.append(payload)
    analysis.total_events += 1
    analysis.events_by_type[event.label] += 1
    analysis.events_by_direction[event.direction or "front"] += 1
    analysis.timeline_by_10s[str(timeline_bucket)] += 1


def build_report(analyses: list[VideoAnalysis], *, args: argparse.Namespace, elapsed_sec: float) -> dict[str, Any]:
    all_events = [event for analysis in analyses for event in analysis.events]
    type_counts = Counter(event["type"] for event in all_events)
    direction_counts = Counter(event.get("direction") or "front" for event in all_events)
    video_counts = {analysis.video_name: analysis.total_events for analysis in analyses}
    timeline = merge_timeline(analyses)
    total_frames = sum(analysis.total_frames for analysis in analyses)
    total_detections = sum(analysis.total_detections for analysis in analyses)
    total_duration = sum(analysis.duration_sec for analysis in analyses)
    top_video = max(video_counts.items(), key=lambda item: item[1]) if video_counts else ("-", 0)
    top_type = max(type_counts.items(), key=lambda item: item[1]) if type_counts else ("-", 0)

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "local_test_videos",
            "videos_dir": project_relative(args.videos_dir),
            "video_glob": args.video_glob,
            "model": project_relative(Path(args.model)),
            "detector_confidence_threshold": args.conf,
            "event_confidence_threshold": args.event_conf,
            "min_consecutive_frames": args.min_frames,
            "cooldown_frames": args.cooldown_frames,
            "max_frames": args.max_frames,
            "analysis_elapsed_sec": round(elapsed_sec, 2),
            "privacy_note": "No original video frames are stored in this JSON report.",
        },
        "summary": {
            "video_count": len(analyses),
            "total_frames": total_frames,
            "total_duration_sec": round(total_duration, 2),
            "total_detections": total_detections,
            "total_events": len(all_events),
            "events_by_type": dict(type_counts),
            "events_by_type_cn": {LABEL_CN.get(label, label): count for label, count in type_counts.items()},
            "events_by_direction": dict(direction_counts),
            "events_by_video": video_counts,
            "top_video": {"video": top_video[0], "events": top_video[1]},
            "top_event_type": {"type": top_type[0], "type_cn": LABEL_CN.get(top_type[0], top_type[0]), "events": top_type[1]},
            "events_per_minute": round(len(all_events) / max(total_duration / 60, 1e-6), 2),
        },
        "videos": [analysis.to_summary_dict() for analysis in analyses],
        "timeline_by_video_10s": {analysis.video_name: dict(analysis.timeline_by_10s) for analysis in analyses},
        "timeline_total_10s": timeline,
        "events": all_events,
    }


def merge_timeline(analyses: list[VideoAnalysis]) -> dict[str, int]:
    merged: defaultdict[str, int] = defaultdict(int)
    for analysis in analyses:
        for bucket, count in analysis.timeline_by_10s.items():
            merged[bucket] += count
    return dict(sorted(merged.items(), key=lambda item: int(item[0])))


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text_lf(
        output_dir / "video_report.json",
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    write_text_lf(
        output_dir / "video_summary.json",
        json.dumps({"metadata": report["metadata"], "summary": report["summary"], "videos": report["videos"]}, ensure_ascii=False, indent=2),
    )
    write_events_csv(output_dir / "video_events.csv", report["events"])
    write_text_lf(output_dir / "dashboard.html", build_dashboard_html(report))


def write_text_lf(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def write_events_csv(path: Path, events: list[dict[str, Any]]) -> None:
    fieldnames = [
        "event_id",
        "video",
        "frame_id",
        "video_time_sec",
        "type",
        "type_cn",
        "confidence",
        "direction",
        "message",
        "bbox",
        "source_frame_ids",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.get("event_id"),
                    "video": event.get("video"),
                    "frame_id": event.get("frame_id"),
                    "video_time_sec": event.get("video_time_sec"),
                    "type": event.get("type"),
                    "type_cn": event.get("type_cn"),
                    "confidence": f"{float(event.get('confidence', 0.0)):.4f}",
                    "direction": event.get("direction"),
                    "message": event.get("message"),
                    "bbox": json.dumps(event.get("bbox"), ensure_ascii=False),
                    "source_frame_ids": json.dumps(event.get("source_frame_ids"), ensure_ascii=False),
                }
            )


def build_dashboard_html(report: dict[str, Any]) -> str:
    data_json = json.dumps(report, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LinkAble 本地视频数据大屏</title>
  <style>
    :root {{
      --bg: #f5f3ee;
      --ink: #1d2525;
      --muted: #67706e;
      --line: #d8d0c3;
      --panel: #fffdf8;
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-3: #334155;
      --danger: #b91c1c;
      --radius: 8px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: Georgia, "Noto Serif SC", "Songti SC", serif;
    }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    header {{
      padding: 24px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: #eee9df;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 20px;
      align-items: end;
    }}
    h1 {{
      margin: 0;
      font-size: 32px;
      line-height: 1.05;
      letter-spacing: 0;
      font-weight: 700;
    }}
    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
      max-width: 900px;
    }}
    .stamp {{
      text-align: right;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
      white-space: nowrap;
    }}
    main {{
      padding: 24px 32px 36px;
      display: grid;
      gap: 16px;
      grid-template-columns: 1.05fr 0.95fr;
      align-items: start;
    }}
    .kpis {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
    }}
    .kpi, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 1px 0 rgba(29,37,37,0.04);
    }}
    .kpi {{
      padding: 14px 14px 12px;
      min-height: 96px;
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .kpi .value {{
      margin-top: 8px;
      font-size: 30px;
      line-height: 1;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .kpi .hint {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .panel {{
      padding: 16px;
      overflow: hidden;
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      line-height: 1.2;
    }}
    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    canvas {{
      width: 100%;
      height: 280px;
      display: block;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
    }}
    .type-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .dot {{
      width: 9px;
      height: 9px;
      border-radius: 99px;
      background: var(--accent);
      display: inline-block;
    }}
    .events {{
      max-height: 412px;
      overflow: auto;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      margin-top: 10px;
    }}
    @media (max-width: 1180px) {{
      main {{ grid-template-columns: 1fr; }}
      .kpis {{ grid-template-columns: repeat(3, 1fr); }}
      header {{ grid-template-columns: 1fr; }}
      .stamp {{ text-align: left; }}
    }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .kpis, .grid2 {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 25px; }}
      canvas {{ height: 240px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>LinkAble 本地视频数据大屏</h1>
        <div class="subtitle">8 段本地测试视频离线分析：YOLO 检测、EventBuilder 去抖、结构化事件记录。报告不保存原始视频帧，仅保留事件级数据。</div>
      </div>
      <div class="stamp">
        <div id="generatedAt"></div>
        <div id="modelLine"></div>
      </div>
    </header>
    <main>
      <section class="kpis" id="kpis"></section>
      <section class="panel">
        <h2>事件类型分布</h2>
        <canvas id="typeChart" width="900" height="360"></canvas>
      </section>
      <section class="panel">
        <h2>各视频事件量</h2>
        <canvas id="videoChart" width="900" height="360"></canvas>
      </section>
      <section class="panel wide">
        <h2>10 秒时间桶事件热度</h2>
        <canvas id="timelineChart" width="1600" height="340"></canvas>
      </section>
      <section class="panel">
        <h2>视频明细</h2>
        <div id="videoTable"></div>
      </section>
      <section class="panel">
        <h2>最新结构化事件</h2>
        <div class="events" id="eventTable"></div>
      </section>
      <section class="panel wide">
        <h2>治理口径</h2>
        <div class="note">本页用于答辩和本地联调展示。P0 主线仍是本地闭环：图片/视频输入、YOLO 推理、事件聚合去抖、模板中文提示、本地缓存播报、结构化事件记录。云端大模型报告和数据大屏是 P1 增强展示，不阻塞本地播报。</div>
      </section>
    </main>
  </div>
  <script id="report-data" type="application/json">{escape_script_json(data_json)}</script>
  <script>
    const report = JSON.parse(document.getElementById('report-data').textContent);
    const colors = ['#0f766e', '#b45309', '#334155', '#b91c1c', '#2563eb', '#7c3aed', '#15803d', '#be123c'];
    const labelCn = {json.dumps(LABEL_CN, ensure_ascii=False)};

    function fmt(n) {{
      return Number(n || 0).toLocaleString('zh-CN');
    }}

    function pct(count, total) {{
      return total ? (count / total * 100).toFixed(1) + '%' : '0.0%';
    }}

    function init() {{
      document.getElementById('generatedAt').textContent = '生成时间 ' + report.metadata.generated_at.replace('T', ' ').slice(0, 19);
      document.getElementById('modelLine').textContent = '模型 ' + report.metadata.model.split(/[\\\\/]/).pop();
      renderKpis();
      drawBarChart('typeChart', mapCounts(report.summary.events_by_type, labelCn), '事件数');
      drawBarChart('videoChart', Object.entries(report.summary.events_by_video), '事件数');
      drawLineChart('timelineChart', Object.entries(report.timeline_total_10s).map(([k, v]) => [k + 's', v]));
      renderVideoTable();
      renderEventTable();
    }}

    function renderKpis() {{
      const s = report.summary;
      const items = [
        ['视频数', s.video_count, '本地 mp4 文件'],
        ['结构化事件', s.total_events, 'EventBuilder 输出'],
        ['检测目标', s.total_detections, 'YOLO P0 检出'],
        ['总帧数', s.total_frames, '已分析帧'],
        ['事件/分钟', s.events_per_minute, '去抖后密度'],
        ['最高风险视频', s.top_video.video, s.top_video.events + ' 起事件'],
      ];
      document.getElementById('kpis').innerHTML = items.map(([label, value, hint]) => `
        <div class="kpi">
          <div class="label">${{label}}</div>
          <div class="value">${{typeof value === 'number' ? fmt(value) : value}}</div>
          <div class="hint">${{hint}}</div>
        </div>
      `).join('');
    }}

    function mapCounts(counts, names) {{
      return Object.entries(counts).map(([k, v]) => [names[k] || k, v]);
    }}

    function drawBarChart(canvasId, rows, yLabel) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      const pad = {{ left: 58, right: 22, top: 18, bottom: 72 }};
      const max = Math.max(1, ...rows.map(([, v]) => Number(v)));
      drawAxes(ctx, w, h, pad, yLabel);
      const gap = 16;
      const barW = Math.max(18, (w - pad.left - pad.right - gap * (rows.length - 1)) / Math.max(rows.length, 1));
      rows.forEach(([label, value], i) => {{
        const x = pad.left + i * (barW + gap);
        const bh = (h - pad.top - pad.bottom) * Number(value) / max;
        const y = h - pad.bottom - bh;
        ctx.fillStyle = colors[i % colors.length];
        ctx.fillRect(x, y, barW, bh);
        ctx.fillStyle = '#1d2525';
        ctx.font = '18px Georgia, serif';
        ctx.textAlign = 'center';
        ctx.fillText(value, x + barW / 2, y - 8);
        ctx.save();
        ctx.translate(x + barW / 2, h - pad.bottom + 16);
        ctx.rotate(-Math.PI / 5);
        ctx.font = '14px Georgia, serif';
        ctx.fillStyle = '#67706e';
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }});
    }}

    function drawLineChart(canvasId, rows) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      const w = canvas.width, h = canvas.height;
      const pad = {{ left: 58, right: 24, top: 18, bottom: 48 }};
      ctx.clearRect(0, 0, w, h);
      drawAxes(ctx, w, h, pad, '事件数');
      const max = Math.max(1, ...rows.map(([, v]) => Number(v)));
      const innerW = w - pad.left - pad.right;
      const innerH = h - pad.top - pad.bottom;
      const step = innerW / Math.max(rows.length - 1, 1);
      ctx.beginPath();
      rows.forEach(([label, value], i) => {{
        const x = pad.left + i * step;
        const y = h - pad.bottom - innerH * Number(value) / max;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.strokeStyle = '#0f766e';
      ctx.lineWidth = 3;
      ctx.stroke();
      rows.forEach(([label, value], i) => {{
        const x = pad.left + i * step;
        const y = h - pad.bottom - innerH * Number(value) / max;
        ctx.fillStyle = '#b45309';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        if (i % 2 === 0) {{
          ctx.fillStyle = '#67706e';
          ctx.font = '12px Georgia, serif';
          ctx.textAlign = 'center';
          ctx.fillText(label, x, h - 18);
        }}
      }});
    }}

    function drawAxes(ctx, w, h, pad, yLabel) {{
      ctx.strokeStyle = '#d8d0c3';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, h - pad.bottom);
      ctx.lineTo(w - pad.right, h - pad.bottom);
      ctx.stroke();
      ctx.fillStyle = '#67706e';
      ctx.font = '13px Georgia, serif';
      ctx.textAlign = 'left';
      ctx.fillText(yLabel, 12, 26);
    }}

    function renderVideoTable() {{
      const rows = report.videos.map(v => `
        <tr>
          <td>${{v.video_name}}</td>
          <td>${{fmt(v.total_frames)}}</td>
          <td>${{fmt(v.total_detections)}}</td>
          <td>${{fmt(v.total_events)}}</td>
          <td>${{v.events_per_minute}}</td>
          <td>${{topType(v.events_by_type)}}</td>
        </tr>
      `).join('');
      document.getElementById('videoTable').innerHTML = `
        <table>
          <thead><tr><th>视频</th><th>帧数</th><th>检测</th><th>事件</th><th>事件/分钟</th><th>主要类型</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>`;
    }}

    function renderEventTable() {{
      const latest = report.events.slice(-28).reverse();
      const rows = latest.map(e => `
        <tr>
          <td>${{e.video}}</td>
          <td>${{Number(e.video_time_sec).toFixed(1)}}s</td>
          <td><span class="type-pill"><span class="dot"></span>${{e.type_cn}}</span></td>
          <td>${{Number(e.confidence).toFixed(2)}}</td>
          <td>${{e.message}}</td>
        </tr>
      `).join('');
      document.getElementById('eventTable').innerHTML = `
        <table>
          <thead><tr><th>视频</th><th>时间</th><th>类型</th><th>置信度</th><th>提示文本</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>`;
    }}

    function topType(counts) {{
      const entries = Object.entries(counts || {{}});
      if (!entries.length) return '-';
      entries.sort((a, b) => b[1] - a[1]);
      return (labelCn[entries[0][0]] || entries[0][0]) + ' / ' + entries[0][1];
    }}

    init();
  </script>
</body>
</html>
"""


def escape_script_json(value: str) -> str:
    return value.replace("</", "<\\/")


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
