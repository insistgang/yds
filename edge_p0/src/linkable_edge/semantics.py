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


def format_distance(distance_m: float | None) -> str:
    if distance_m is None:
        return "前方"
    rounded = max(1, int(round(distance_m)))
    return f"前方{rounded}米"


def format_direction(direction: str | None) -> str:
    if not direction:
        return ""
    return DIRECTION_MAP.get(direction, direction)


def format_obstacle_location(direction: str | None, distance_m: float | None = None) -> str:
    prefix = format_distance(distance_m)
    location = format_direction(direction)
    if not location:
        return prefix
    return f"{prefix}{location}"


def render_event_text(event: DetectionEvent) -> str:
    prefix = format_distance(event.distance_m)
    direction = format_direction(event.direction)
    is_front = direction == "前方"

    if event.label == "blind_road_occupied":
        if direction and not is_front:
            return f"{prefix}盲道被占用，建议向{direction}绕行。"
        return f"{prefix}盲道被占用，请注意绕行。"

    if event.label == "stairs":
        if direction and not is_front:
            return f"{prefix}{direction}有台阶，请减速。"
        return f"{prefix}有台阶，请减速。"

    if event.label == "ramp":
        if direction and not is_front:
            return f"{prefix}有坡道，位于{direction}。"
        return f"{prefix}有坡道。"

    if event.label == "road_obstacle":
        location = format_obstacle_location(event.direction)
        return f"{location}有障碍，请注意避让。"

    if event.label == "traffic_red":
        return f"{prefix}是红灯，请在路口前等待。"

    if event.label == "crosswalk":
        return f"{prefix}有人行横道，请沿横道通过。"

    return f"{prefix}发现异常目标，请注意安全。"


# 全部模板文本列表，供启动时预加载 TTS 缓存
ALL_TEMPLATE_TEXTS = [
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
    "前方左侧有障碍，请注意避让。",
    "前方右侧有障碍，请注意避让。",
    "前方左侧有台阶，请减速。",
    "前方右侧有台阶，请减速。",
    "前方左侧有坡道。",
    "前方右侧有坡道。",
    "左侧盲道被占用，请注意绕行。",
    "右侧盲道被占用，请注意绕行。",
]
