from __future__ import annotations

from .models import DetectionEvent


DIRECTION_MAP = {
    "left": "左侧",
    "right": "右侧",
    "front": "前方",
    "center": "前方",
    "left_front": "左前方",
    "right_front": "右前方",
}

_FRONT_DIRECTIONS = {None, "", "front", "center"}


def _direction_bucket(direction: str | None) -> str:
    if direction == "left":
        return "left"
    if direction == "right":
        return "right"
    if direction == "left_front":
        return "left_front"
    if direction == "right_front":
        return "right_front"
    return "front"


def _rounded_distance_m(distance_m: float) -> int:
    return max(1, min(8, int(round(distance_m))))


def format_distance(distance_m: float | None) -> str:
    if distance_m is None:
        return "前方"
    rounded = _rounded_distance_m(distance_m)
    return f"前方{rounded}米"


def format_direction(direction: str | None) -> str:
    if not direction:
        return ""
    return DIRECTION_MAP.get(direction, direction)


def format_obstacle_location(direction: str | None, distance_m: float | None = None) -> str:
    location = format_direction(direction)
    if distance_m is None:
        return location or "前方"
    if not location or location == "前方":
        return format_distance(distance_m)
    return f"{location}{_rounded_distance_m(distance_m)}米"


def render_event_text(event: DetectionEvent) -> str:
    prefix = format_distance(event.distance_m)
    bucket = _direction_bucket(event.direction)

    if event.label == "blind_road_occupied":
        return P0_TEMPLATE_BY_LABEL_AND_DIRECTION["blind_road_occupied"][bucket]

    if event.label == "stairs":
        return P0_TEMPLATE_BY_LABEL_AND_DIRECTION["stairs"][bucket]

    if event.label == "ramp":
        return P0_TEMPLATE_BY_LABEL_AND_DIRECTION["ramp"][bucket]

    if event.label == "road_obstacle":
        if event.distance_m is not None and event.direction in _FRONT_DIRECTIONS:
            rounded = _rounded_distance_m(event.distance_m)
            return ROAD_OBSTACLE_DISTANCE_TEMPLATES[rounded]
        return P0_TEMPLATE_BY_LABEL_AND_DIRECTION["road_obstacle"][bucket]

    if event.label == "traffic_red":
        return f"{prefix}是红灯，请在路口前等待。"

    if event.label == "crosswalk":
        return f"{prefix}有人行横道，请沿横道通过。"

    return f"{prefix}发现异常目标，请注意安全。"


P0_TEMPLATE_BY_LABEL_AND_DIRECTION = {
    "blind_road_occupied": {
        "front": "前方盲道被占用，请注意绕行。",
        "left_front": "前方盲道被占用，建议向左前方绕行。",
        "right_front": "前方盲道被占用，建议向右前方绕行。",
        "left": "左侧盲道被占用，请注意绕行。",
        "right": "右侧盲道被占用，请注意绕行。",
    },
    "stairs": {
        "front": "前方有台阶，请减速。",
        "left_front": "左前方有台阶，请减速。",
        "right_front": "右前方有台阶，请减速。",
        "left": "左侧有台阶，请减速。",
        "right": "右侧有台阶，请减速。",
    },
    "ramp": {
        "front": "前方有坡道。",
        "left_front": "前方有坡道，位于左前方。",
        "right_front": "前方有坡道，位于右前方。",
        "left": "左侧有坡道。",
        "right": "右侧有坡道。",
    },
    "road_obstacle": {
        "front": "前方有障碍，请注意避让。",
        "left_front": "左前方有障碍，请注意避让。",
        "right_front": "右前方有障碍，请注意避让。",
        "left": "左侧有障碍，请注意避让。",
        "right": "右侧有障碍，请注意避让。",
    },
}

ROAD_OBSTACLE_DISTANCE_TEMPLATES = {
    distance_m: f"前方{distance_m}米有障碍，请注意避让。"
    for distance_m in range(1, 9)
}


# 全部模板文本列表，供启动时预加载 TTS 缓存
ALL_TEMPLATE_TEXTS = [
    # P0 四类核心模板
    "前方盲道被占用，请注意绕行。",
    "前方盲道被占用，建议向左前方绕行。",
    "前方盲道被占用，建议向右前方绕行。",
    "前方有台阶，请减速。",
    "左前方有台阶，请减速。",
    "右前方有台阶，请减速。",
    "前方有坡道。",
    "前方有坡道，位于左前方。",
    "前方有坡道，位于右前方。",
    "前方有障碍，请注意避让。",
    "左前方有障碍，请注意避让。",
    "右前方有障碍，请注意避让。",
    "前方1米有障碍，请注意避让。",
    "前方2米有障碍，请注意避让。",
    "前方3米有障碍，请注意避让。",
    "前方4米有障碍，请注意避让。",
    "前方5米有障碍，请注意避让。",
    "前方6米有障碍，请注意避让。",
    "前方7米有障碍，请注意避让。",
    "前方8米有障碍，请注意避让。",
    "左侧有障碍，请注意避让。",
    "右侧有障碍，请注意避让。",
    "左侧有台阶，请减速。",
    "右侧有台阶，请减速。",
    "左侧有坡道。",
    "右侧有坡道。",
    "左侧盲道被占用，请注意绕行。",
    "右侧盲道被占用，请注意绕行。",
]
