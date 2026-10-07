"""Complementary, non-overlapping image grids at multiple scales."""
from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class GridSlice:
    bbox_xyxy: Tuple[int, int, int, int]
    scale: int
    template_crop: np.ndarray
    inspection_crop: np.ndarray


def iter_corresponding_slices(
    template: np.ndarray,
    inspection: np.ndarray,
    scale: int,
) -> Iterator[GridSlice]:
    """Yield corresponding complete cells from an n-by-n grid.

    When dimensions are not divisible by n, incomplete right and bottom
    margins are omitted, matching the source implementation.
    """
    if template.shape[:2] != inspection.shape[:2]:
        raise ValueError("Template and inspection images must have equal dimensions.")
    if scale < 1:
        raise ValueError("Grid scale must be at least 1.")

    height, width = template.shape[:2]
    cell_width = width // scale
    cell_height = height // scale
    if cell_width == 0 or cell_height == 0:
        raise ValueError("Grid scale is larger than the image dimensions.")

    for row in range(scale):
        y1 = row * cell_height
        y2 = y1 + cell_height
        if y2 > height:
            continue
        for column in range(scale):
            x1 = column * cell_width
            x2 = x1 + cell_width
            if x2 > width:
                continue
            yield GridSlice(
                bbox_xyxy=(x1, y1, x2, y2),
                scale=scale,
                template_crop=template[y1:y2, x1:x2],
                inspection_crop=inspection[y1:y2, x1:x2],
            )


def collect_multiscale_slices(
    template: np.ndarray,
    inspection: np.ndarray,
    scales: Tuple[int, ...] = (8, 9, 10),
) -> Iterator[GridSlice]:
    """Yield all valid corresponding slices from the requested grid scales."""
    for scale in scales:
        yield from iter_corresponding_slices(template, inspection, scale)