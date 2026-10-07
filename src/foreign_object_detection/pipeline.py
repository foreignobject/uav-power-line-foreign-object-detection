"""End-to-end feature-aligned, multi-scale foreign-object localization."""
from dataclasses import dataclass
import zlib
from typing import Optional, Tuple

import cv2
import numpy as np

from .ifm import register_inspection_to_template
from .mgs import collect_multiscale_slices
from .sam import haarpsi_score, score_slice_candidates, select_lowest_similarity


@dataclass(frozen=True)
class PipelineConfig:
    image_width: int = 1080
    image_height: int = 720
    grid_scales: Tuple[int, ...] = (8, 9, 10)
    ratio_threshold: float = 0.55
    ransac_reprojection_threshold: float = 4.0
    descriptor_norm: str = "L2"
    minimum_good_matches: int = 5
    haarpsi_input_mode: str = "three_channel_passthrough"
    relative_similarity_threshold: float = 0.9
    max_black_pixel_fraction: float = 0.03
    seed_ransac_from_pair_id: bool = True

    def validate(self) -> None:
        if self.image_width < 16 or self.image_height < 16:
            raise ValueError("Configured image dimensions must be at least 16 pixels.")
        if not self.grid_scales or any(scale < 1 for scale in self.grid_scales):
            raise ValueError("At least one positive grid scale is required.")
        if len(set(self.grid_scales)) != len(self.grid_scales):
            raise ValueError("Grid scales must not contain duplicates.")
        if not 0.0 < self.ratio_threshold < 1.0:
            raise ValueError("ratio_threshold must be between 0 and 1.")
        if self.ransac_reprojection_threshold <= 0.0:
            raise ValueError("ransac_reprojection_threshold must be positive.")
        if self.descriptor_norm.upper() not in ("L2", "HAMMING"):
            raise ValueError("descriptor_norm must be 'L2' or 'HAMMING'.")
        if self.minimum_good_matches < 4:
            raise ValueError("minimum_good_matches must be at least 4.")
        if self.haarpsi_input_mode not in (
            "grayscale",
            "three_channel_passthrough",
        ):
            raise ValueError(
                "haarpsi_input_mode must be 'grayscale' or "
                "'three_channel_passthrough'."
            )
        if not 0.0 <= self.relative_similarity_threshold <= 1.0:
            raise ValueError("relative_similarity_threshold must be between 0 and 1.")
        if not 0.0 <= self.max_black_pixel_fraction <= 1.0:
            raise ValueError("max_black_pixel_fraction must be between 0 and 1.")


@dataclass(frozen=True)
class DetectionResult:
    pair_id: str
    prediction: Optional[dict]
    global_similarity: float
    good_match_count: int
    inlier_count: int
    image_size: Tuple[int, int]


def _rng_seed(pair_id: str) -> int:
    """Map a pair identifier to a stable, non-negative OpenCV seed."""
    try:
        value = int(pair_id)
    except ValueError:
        value = zlib.crc32(pair_id.encode("utf-8"))
    return int(value & 0x7FFFFFFF)


def _prepare_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image must not be empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel OpenCV BGR image.")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def detect_pair(
    template_bgr: np.ndarray,
    inspection_bgr: np.ndarray,
    pair_id: str = "0",
    config: Optional[PipelineConfig] = None,
) -> DetectionResult:
    """Align one image pair, score multi-scale slices, and return at most one box."""
    config = config or PipelineConfig()
    config.validate()

    template = _prepare_image(template_bgr, config.image_width, config.image_height)
    inspection = _prepare_image(
        inspection_bgr, config.image_width, config.image_height
    )
    seed = _rng_seed(str(pair_id)) if config.seed_ransac_from_pair_id else None
    alignment = register_inspection_to_template(
        template,
        inspection,
        ratio_threshold=config.ratio_threshold,
        ransac_reprojection_threshold=config.ransac_reprojection_threshold,
        descriptor_norm=config.descriptor_norm,
        minimum_good_matches=config.minimum_good_matches,
        rng_seed=seed,
    )
    aligned = alignment.aligned_inspection

    full_score = haarpsi_score(
        template, aligned, input_mode=config.haarpsi_input_mode
    )
    slices = list(collect_multiscale_slices(template, aligned, config.grid_scales))
    candidates = score_slice_candidates(
        slices,
        full_score,
        relative_similarity_threshold=config.relative_similarity_threshold,
        max_black_pixel_fraction=config.max_black_pixel_fraction,
        haarpsi_input_mode=config.haarpsi_input_mode,
    )
    selected = select_lowest_similarity(candidates)
    return DetectionResult(
        pair_id=str(pair_id),
        prediction=selected,
        global_similarity=full_score,
        good_match_count=alignment.good_match_count,
        inlier_count=alignment.inlier_count,
        image_size=(config.image_width, config.image_height),
    )
