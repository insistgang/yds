# Public P0 Dataset Staging

This directory is for public dataset metadata and local staging only. It does not replace the campus "last 20 meters" dataset.

## Purpose

Use public datasets only as cold-start or supplemental data for the four P0 labels:

- `road_obstacle`
- `stairs`
- `ramp`
- `blind_road_occupied`

`blind_road_occupied` is relationship-specific and still requires local campus collection. Public tactile paving data can help review blind-road context, but it is not the same as blind-road occupation.

## Structure

```text
datasets/public_p0_sources/
  README.md
  sources.json
  raw/
  converted_yolo/
  _manifests/
```

The `raw/` and `converted_yolo/` folders are ignored by git. Do not commit downloaded public images, converted labels, or derived YOLO datasets.

## Rules

- Do not add P1 labels such as `traffic_light` or `crosswalk`.
- Do not add `person` or `vehicle` as standalone P0 classes.
- Do not claim public proxy data is the final LinkAble P0 model.
- Keep source license, URL, and mapping notes in `sources.json` and docs.
