# LinkAble P0 Raw Dataset

This directory stores reviewed, discrete image samples for the LinkAble P0 dataset. It is for field collection and later annotation only.

## Scope

Allowed positive classes:

- `road_obstacle`
- `stairs`
- `ramp`
- `blind_road_occupied`

Supporting non-training bucket:

- `negative`: background and no-event samples used for false-positive review. It is not a YOLO class.

Rejected bucket:

- `rejected_privacy`: temporary holding area for images that contain clear faces, plates, school identifiers, name tags, or other sensitive content. These samples must be deleted or anonymized before any training use.

## Structure

```text
datasets/linkable_p0_raw/
  README.md
  manifest.csv
  road_obstacle/images/
  stairs/images/
  ramp/images/
  blind_road_occupied/images/
  negative/images/
  rejected_privacy/
  _manifests/
```

No continuous video directory is created by default. The field workflow should collect discrete images only.

## Manifest Fields

`manifest.csv` uses these columns:

- `image_path`
- `label`
- `route_point`
- `session`
- `timestamp`
- `device`
- `width`
- `height`
- `privacy_checked`
- `notes`

`privacy_checked` defaults to `false`. A human review must set privacy status before samples are used for training.

## Privacy

Do not keep clear faces, license plates, school names, name tags, or other sensitive identifiers in training samples. Do not upload raw images or continuous video streams.
