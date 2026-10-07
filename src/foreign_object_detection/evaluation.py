"""Dataset-level precision, recall, and IoU evaluation."""
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

Box = Tuple[float, float, float, float]


def intersection_over_union(first: Box, second: Box) -> float:
    """Compute IoU for XYXY boxes using continuous coordinates."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def load_voc_annotations(
    directory: Path,
    image_ids=None,
    missing_is_negative: bool = False,
) -> Dict[str, List[Box]]:
    """Load VOC XML files and optionally mark absent IDs as negative images."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("Ground-truth XML directory does not exist.")

    annotations = {}
    xml_files = sorted(directory.glob("*.xml"))
    if not xml_files:
        raise ValueError("No XML annotations were found.")
    for path in xml_files:
        root = ET.parse(str(path)).getroot()
        boxes = []
        for obj in root.findall("./object"):
            box = obj.find("bndbox")
            if box is None:
                raise ValueError("Missing bndbox element in {}.".format(path.name))
            coords = []
            for key in ("xmin", "ymin", "xmax", "ymax"):
                value = box.findtext(key)
                if value is None:
                    raise ValueError(
                        "Missing {} coordinate in {}.".format(key, path.name)
                    )
                coords.append(float(value))
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError("Invalid bounding box in {}.".format(path.name))
            boxes.append(tuple(coords))
        annotations[path.stem] = boxes

    if image_ids is not None:
        expected = set(image_ids)
        extras = sorted(set(annotations) - expected)
        if extras:
            raise ValueError(
                "Annotation IDs do not occur in predictions: {}.".format(extras[:10])
            )
        missing = sorted(expected - set(annotations))
        if missing and not missing_is_negative:
            raise ValueError(
                "Some images have no XML annotation. Pass the explicit "
                "missing-is-negative option only when absent files mean "
                "negative images. Missing IDs: {}.".format(missing[:10])
            )
        for image_id in missing:
            annotations[image_id] = []
    return annotations


def _extract_boxes(prediction) -> List[Box]:
    if prediction is None:
        return []
    if isinstance(prediction, dict):
        prediction = [prediction]
    boxes = []
    for item in prediction:
        box = item.get("bbox_xyxy") if isinstance(item, dict) else item
        if box is None or len(box) != 4:
            raise ValueError("Each prediction must contain an XYXY box.")
        boxes.append(tuple(float(value) for value in box))
    return boxes


def _match_boxes(predictions: List[Box], ground_truth: List[Box], threshold: float):
    """Find a maximum-cardinality one-to-one match using strict IoU thresholding."""
    overlaps = [
        [
            (index, intersection_over_union(prediction, truth))
            for index, truth in enumerate(ground_truth)
            if intersection_over_union(prediction, truth) > threshold
        ]
        for prediction in predictions
    ]
    for row in overlaps:
        row.sort(key=lambda item: item[1], reverse=True)

    truth_to_prediction = {}

    def assign(prediction_index: int, visited: set) -> bool:
        for truth_index, _ in overlaps[prediction_index]:
            if truth_index in visited:
                continue
            visited.add(truth_index)
            previous = truth_to_prediction.get(truth_index)
            if previous is None or assign(previous, visited):
                truth_to_prediction[truth_index] = prediction_index
                return True
        return False

    matched = 0
    for prediction_index in range(len(predictions)):
        if assign(prediction_index, set()):
            matched += 1
    return matched


def evaluate_predictions(
    predictions: Dict[str, object],
    ground_truth: Dict[str, List[Box]],
    iou_threshold: float = 0.4,
) -> dict:
    """Compute paper-style counts and metrics with one-to-one box matching."""
    if not 0.0 <= iou_threshold < 1.0:
        raise ValueError("IoU threshold must be in the range [0, 1).")
    if set(predictions) != set(ground_truth):
        missing_predictions = sorted(set(ground_truth) - set(predictions))
        missing_annotations = sorted(set(predictions) - set(ground_truth))
        raise ValueError(
            "Prediction and annotation IDs differ (missing predictions: {}; "
            "missing annotations: {}).".format(
                missing_predictions[:10], missing_annotations[:10]
            )
        )

    counts = defaultdict(int)
    for image_id, truths in ground_truth.items():
        boxes = _extract_boxes(predictions[image_id])
        true_positives = _match_boxes(boxes, truths, iou_threshold)
        counts["TP"] += true_positives
        counts["FP"] += len(boxes) - true_positives
        counts["FN"] += len(truths) - true_positives
        if not truths and not boxes:
            counts["TN"] += 1

    precision_denominator = counts["TP"] + counts["FP"]
    recall_denominator = counts["TP"] + counts["FN"]
    precision = (
        100.0 * counts["TP"] / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = (
        100.0 * counts["TP"] / recall_denominator if recall_denominator else 0.0
    )
    return {
        "TP": counts["TP"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "TN": counts["TN"],
        "precision_percent": precision,
        "recall_percent": recall,
        "iou_threshold_strictly_greater_than": iou_threshold,
        "evaluated_images": len(ground_truth),
        "ground_truth_boxes": sum(len(boxes) for boxes in ground_truth.values()),
    }