"""Command-line inference for paired template and inspection image folders."""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a JSON object.")
    return config


def _image_files(directory: Path) -> Dict[str, Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("Image directory does not exist.")
    files = {}
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem
        if key in files:
            raise ValueError("Duplicate image identifier: {}.".format(key))
        files[key] = path
    if not files:
        raise ValueError("No supported image files were found.")
    return files


def _read_bgr(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be decoded.")
    return image


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)


def _logger(log_file: Path = None) -> logging.Logger:
    logger = logging.getLogger("foreign_object_detection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _config_from_mapping(values: dict):
    from .pipeline import PipelineConfig

    allowed = {
        "image_width",
        "image_height",
        "grid_scales",
        "ratio_threshold",
        "ransac_reprojection_threshold",
        "descriptor_norm",
        "minimum_good_matches",
        "haarpsi_input_mode",
        "relative_similarity_threshold",
        "max_black_pixel_fraction",
        "seed_ransac_from_pair_id",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError("Unknown configuration keys: {}.".format(sorted(unknown)))
    if "grid_scales" in values:
        values["grid_scales"] = tuple(int(item) for item in values["grid_scales"])
    config = PipelineConfig(**values)
    config.validate()
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Detect local changes between normal templates and inspection images."
    )
    parser.add_argument("--template-dir", required=True, type=Path)
    parser.add_argument("--inspection-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args(argv)

    from .pipeline import detect_pair

    config = _config_from_mapping(_read_config(args.config))
    templates = _image_files(args.template_dir)
    inspections = _image_files(args.inspection_dir)
    if set(templates) != set(inspections):
        raise ValueError(
            "Template and inspection image identifiers must match exactly."
        )

    logger = _logger(args.log_file)
    output = {
        "format_version": 1,
        "method": "IFM-MGS-SAM",
        "coordinate_space": "XYXY pixels after configured resizing",
        "parameters": {
            "image_width": config.image_width,
            "image_height": config.image_height,
            "grid_scales": list(config.grid_scales),
            "descriptor_distance": config.descriptor_norm.upper(),
            "minimum_good_matches": config.minimum_good_matches,
            "ratio_threshold": config.ratio_threshold,
            "ransac_reprojection_threshold": config.ransac_reprojection_threshold,
            "relative_similarity_threshold": config.relative_similarity_threshold,
            "max_black_pixel_fraction": config.max_black_pixel_fraction,
            "seed_ransac_from_pair_id": config.seed_ransac_from_pair_id,
            "haarpsi_input": config.haarpsi_input_mode,
            "haarpsi_C": 30,
            "haarpsi_alpha": 4.2,
        },
        "results": [],
    }

    identifiers = sorted(
        templates,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value.lower()),
    )
    errors = 0
    for image_id in identifiers:
        logger.info("Processing pair %s", image_id)
        try:
            result = detect_pair(
                _read_bgr(templates[image_id]),
                _read_bgr(inspections[image_id]),
                pair_id=image_id,
                config=config,
            )
            output["results"].append(
                {
                    "image_id": image_id,
                    "status": "ok",
                    "prediction": result.prediction,
                    "global_similarity": result.global_similarity,
                    "good_match_count": result.good_match_count,
                    "inlier_count": result.inlier_count,
                }
            )
        except Exception as error:
            errors += 1
            logger.exception("Pair %s failed.", image_id)
            output["results"].append(
                {
                    "image_id": image_id,
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "prediction": None,
                }
            )
    output["summary"] = {
        "total_pairs": len(identifiers),
        "successful_pairs": len(identifiers) - errors,
        "failed_pairs": errors,
    }
    _write_json(args.output_json, output)
    logger.info(
        "Finished: %d successful, %d failed.",
        len(identifiers) - errors,
        errors,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
