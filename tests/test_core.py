"""Small checks for the public pipeline and evaluation behavior."""
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foreign_object_detection.evaluation import (
    evaluate_predictions,
    load_voc_annotations,
)
from foreign_object_detection.mgs import GridSlice, iter_corresponding_slices
from foreign_object_detection.pipeline import PipelineConfig, detect_pair
from foreign_object_detection.sam import haarpsi_score, score_slice_candidates


class CoreTests(unittest.TestCase):
    def test_grid_has_expected_number_of_complete_slices(self):
        image = np.zeros((720, 1080, 3), dtype=np.uint8)
        self.assertEqual(len(list(iter_corresponding_slices(image, image, 8))), 64)
        self.assertEqual(len(list(iter_corresponding_slices(image, image, 9))), 81)
        self.assertEqual(len(list(iter_corresponding_slices(image, image, 10))), 100)

    def test_haarpsi_identity_is_higher_than_changed_image(self):
        reference = np.zeros((96, 128), dtype=np.uint8)
        cv2.rectangle(reference, (20, 20), (80, 70), 180, -1)
        changed = reference.copy()
        cv2.rectangle(changed, (88, 10), (116, 85), 255, -1)
        self.assertGreater(haarpsi_score(reference, reference), 0.99)
        self.assertLess(
            haarpsi_score(reference, changed),
            haarpsi_score(reference, reference),
        )

    def test_invalid_black_slice_is_skipped_before_haarpsi(self):
        crop = np.zeros((32, 32, 3), dtype=np.uint8)
        grid_slice = GridSlice(
            bbox_xyxy=(0, 0, 32, 32),
            scale=1,
            template_crop=crop,
            inspection_crop=crop,
        )
        with patch(
            "foreign_object_detection.sam.haarpsi_score",
            side_effect=AssertionError("Invalid black slice should be skipped."),
        ) as score_mock:
            candidates = score_slice_candidates([grid_slice], full_image_score=0.8)
        self.assertEqual(candidates, [])
        score_mock.assert_not_called()

    def test_evaluator_counts_unmatched_prediction_and_ground_truth(self):
        predictions = {"1": {"bbox_xyxy": [0, 0, 10, 10]}}
        ground_truth = {"1": [(0.0, 0.0, 4.0, 10.0)]}
        metrics = evaluate_predictions(predictions, ground_truth, iou_threshold=0.4)
        self.assertEqual(metrics["TP"], 0)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["FN"], 1)

    def test_missing_xml_requires_explicit_negative_image_policy(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotation_path = Path(temporary_directory) / "1.xml"
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = "1.jpg"
            ET.ElementTree(root).write(annotation_path, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no XML annotation"):
                load_voc_annotations(
                    Path(temporary_directory), image_ids={"1", "2"}
                )

            annotations = load_voc_annotations(
                Path(temporary_directory),
                image_ids={"1", "2"},
                missing_is_negative=True,
            )
            self.assertEqual(annotations["1"], [])
            self.assertEqual(annotations["2"], [])

    def test_synthetic_pair_runs_through_pipeline(self):
        rng = np.random.default_rng(3)
        gray = np.zeros((320, 480), dtype=np.uint8)
        for _ in range(140):
            x = int(rng.integers(0, gray.shape[1]))
            y = int(rng.integers(0, gray.shape[0]))
            radius = int(rng.integers(2, 8))
            cv2.circle(gray, (x, y), radius, int(rng.integers(40, 256)), -1)
        for x in range(20, 460, 40):
            cv2.line(gray, (x, 0), (x + 20, 319), 120, 1)
        template = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        transform = np.float32([[1, 0, 3], [0, 1, -2]])
        inspection = cv2.warpAffine(template, transform, (480, 320))
        defect_region = (200, 120, 360, 280)
        cv2.rectangle(
            inspection,
            defect_region[:2],
            defect_region[2:],
            (255, 255, 255),
            -1,
        )

        config = PipelineConfig(
            image_width=480,
            image_height=320,
            grid_scales=(4, 5, 6),
            seed_ransac_from_pair_id=True,
        )
        result = detect_pair(template, inspection, pair_id="1", config=config)
        self.assertEqual(result.pair_id, "1")
        self.assertGreaterEqual(result.good_match_count, 4)
        self.assertGreaterEqual(result.inlier_count, 4)
        self.assertIsNotNone(result.prediction)
        x1, y1, x2, y2 = result.prediction["bbox_xyxy"]
        self.assertEqual(len(result.prediction["bbox_xyxy"]), 4)
        self.assertLessEqual(defect_region[0], (x1 + x2) / 2)
        self.assertLessEqual((x1 + x2) / 2, defect_region[2])
        self.assertLessEqual(defect_region[1], (y1 + y2) / 2)
        self.assertLessEqual((y1 + y2) / 2, defect_region[3])


if __name__ == "__main__":
    unittest.main()
