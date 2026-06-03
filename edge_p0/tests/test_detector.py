from __future__ import annotations

import unittest

from linkable_edge.detector import YoloDetector, YoloDetectorConfig, infer_direction_from_bbox, normalize_label


class DetectorUtilityTests(unittest.TestCase):
    def test_keeps_p0_label(self) -> None:
        self.assertEqual(normalize_label("stairs", {}), "stairs")

    def test_maps_coco_label_to_p0_label(self) -> None:
        self.assertEqual(normalize_label("bicycle", {"bicycle": "blind_road_occupied"}), "blind_road_occupied")

    def test_drops_unknown_or_invalid_labels(self) -> None:
        self.assertIsNone(normalize_label("dog", {}))
        self.assertIsNone(normalize_label("dog", {"dog": "pet"}))

    def test_infers_direction_by_horizontal_thirds(self) -> None:
        self.assertEqual(infer_direction_from_bbox((0, 0, 90, 90), 300), "left_front")
        self.assertEqual(infer_direction_from_bbox((105, 0, 195, 90), 300), "front")
        self.assertEqual(infer_direction_from_bbox((210, 0, 300, 90), 300), "right_front")

    def test_predict_frame_returns_empty_on_empty_or_bad_frame(self) -> None:
        class FailingModel:
            def predict(self, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("bad frame")

        detector = YoloDetector.__new__(YoloDetector)
        detector.config = YoloDetectorConfig()
        detector._model = FailingModel()

        frame = detector.predict_frame(None, frame_id=99)

        self.assertEqual(frame.frame_id, 99)
        self.assertEqual(frame.detections, [])

    def test_predict_image_returns_empty_on_bad_image(self) -> None:
        class FailingModel:
            def predict(self, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("bad image")

        detector = YoloDetector.__new__(YoloDetector)
        detector.config = YoloDetectorConfig()
        detector._model = FailingModel()

        frame = detector.predict_image("missing-or-corrupt.jpg", frame_id=7)

        self.assertEqual(frame.frame_id, 7)
        self.assertEqual(frame.detections, [])


if __name__ == "__main__":
    unittest.main()
