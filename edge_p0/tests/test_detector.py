from __future__ import annotations

import unittest

from linkable_edge.detector import infer_direction_from_bbox, normalize_label


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


if __name__ == "__main__":
    unittest.main()
