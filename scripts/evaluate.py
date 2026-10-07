"""Command-line evaluation against Pascal VOC-style XML annotations."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foreign_object_detection.evaluation import (
    evaluate_predictions,
    load_voc_annotations,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted XYXY boxes with strict one-to-one IoU matching."
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--ground-truth-xml-dir", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument(
        "--missing-xml-is-negative",
        action="store_true",
        help="Treat prediction IDs without an XML file as images with no ground-truth boxes.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    with args.predictions.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    result_rows = payload.get("results")
    if not isinstance(result_rows, list):
        raise ValueError("Prediction JSON must contain a results list.")

    predictions = {}
    for row in result_rows:
        image_id = str(row["image_id"])
        if image_id in predictions:
            raise ValueError("Duplicate prediction identifier: {}.".format(image_id))
        if row.get("status") != "ok":
            raise ValueError(
                "Cannot evaluate failed inference for image {}.".format(image_id)
            )
        predictions[image_id] = row.get("prediction")

    ground_truth = load_voc_annotations(
        args.ground_truth_xml_dir,
        image_ids=set(predictions),
        missing_is_negative=args.missing_xml_is_negative,
    )
    metrics = evaluate_predictions(
        predictions,
        ground_truth,
        iou_threshold=args.iou_threshold,
    )
    metrics["missing_xml_is_negative"] = args.missing_xml_is_negative
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())