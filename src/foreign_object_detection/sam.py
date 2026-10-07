"""HaarPSI scoring and abnormal-slice filtering."""
from typing import Any, Dict

import cv2
import numpy as np

from .third_party.haarPsi import haar_psi


def grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR image to grayscale, preserving grayscale inputs."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a grayscale or three-channel BGR image.")


def _haarpsi_input(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "grayscale":
        return grayscale(image)
    if mode == "three_channel_passthrough":
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 1:
            return image[:, :, 0]
        if image.ndim == 3 and image.shape[2] == 3:
            return image
    raise ValueError(
        "haarpsi_input_mode must be 'grayscale' or 'three_channel_passthrough'."
    )


def haarpsi_score(
    reference: np.ndarray,
    comparison: np.ndarray,
    input_mode: str = "three_channel_passthrough",
) -> float:
    """Return a scalar HaarPSI score using the configured image channel mode."""
    reference_input = _haarpsi_input(reference, input_mode)
    comparison_input = _haarpsi_input(comparison, input_mode)
    if reference_input.shape != comparison_input.shape:
        raise ValueError("HaarPSI inputs must have equal dimensions.")
    if reference_input.size == 0:
        raise ValueError("HaarPSI inputs must not be empty.")
    if reference_input.dtype != np.uint8:
        reference_input = np.clip(reference_input, 0, 255).astype(np.uint8)
    if comparison_input.dtype != np.uint8:
        comparison_input = np.clip(comparison_input, 0, 255).astype(np.uint8)
    return float(haar_psi(reference_input, comparison_input))


def black_pixel_fraction(image: np.ndarray) -> float:
    """Return the fraction of pixels whose three BGR channels are exactly zero."""
    if image.ndim == 2:
        black = image == 0
    elif image.ndim == 3 and image.shape[2] == 1:
        black = image[:, :, 0] == 0
    elif image.ndim == 3 and image.shape[2] == 3:
        black = np.all(image == 0, axis=2)
    else:
        raise ValueError("Expected a grayscale or three-channel BGR image.")
    return float(np.count_nonzero(black)) / float(black.size)


def score_slice_candidates(
    slices,
    full_image_score: float,
    relative_similarity_threshold: float = 0.9,
    max_black_pixel_fraction: float = 0.03,
    haarpsi_input_mode: str = "three_channel_passthrough",
) -> list:
    """Keep local regions that satisfy both paper-described SAM criteria."""
    if not 0.0 <= relative_similarity_threshold <= 1.0:
        raise ValueError("relative_similarity_threshold must be between 0 and 1.")
    if not 0.0 <= max_black_pixel_fraction <= 1.0:
        raise ValueError("max_black_pixel_fraction must be between 0 and 1.")

    candidates = []
    for grid_slice in slices:
        black_fraction = black_pixel_fraction(grid_slice.inspection_crop)
        if black_fraction >= max_black_pixel_fraction:
            continue
        local_score = haarpsi_score(
            grid_slice.template_crop,
            grid_slice.inspection_crop,
            input_mode=haarpsi_input_mode,
        )
        if (
            local_score < relative_similarity_threshold * full_image_score
        ):
            candidates.append(
                {
                    "bbox_xyxy": list(grid_slice.bbox_xyxy),
                    "local_score": local_score,
                    "global_score": full_image_score,
                    "grid_scale": int(grid_slice.scale),
                    "black_pixel_fraction": black_fraction,
                }
            )
    return candidates


def select_lowest_similarity(candidates: list) -> Dict[str, Any]:
    """Return the globally least-similar retained slice, if one exists."""
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            candidate["local_score"],
            candidate["grid_scale"],
            candidate["bbox_xyxy"][1],
            candidate["bbox_xyxy"][0],
        ),
    )
