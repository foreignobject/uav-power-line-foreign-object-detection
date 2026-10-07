"""AKAZE feature matching and homography-based image registration."""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


class RegistrationError(RuntimeError):
    """Raised when a reliable image registration cannot be estimated."""


@dataclass(frozen=True)
class Alignment:
    aligned_inspection: np.ndarray
    homography: np.ndarray
    good_match_count: int
    inlier_count: int


def register_inspection_to_template(
    template_bgr: np.ndarray,
    inspection_bgr: np.ndarray,
    ratio_threshold: float = 0.55,
    ransac_reprojection_threshold: float = 4.0,
    descriptor_norm: str = "L2",
    minimum_good_matches: int = 5,
    rng_seed: Optional[int] = None,
) -> Alignment:
    """Warp the inspection image into the template image coordinate system.

    Descriptor distance and the minimum match count are explicit because they
    differ between the audited source behavior and the manuscript description.
    A seed can be supplied to make RANSAC repeatable.
    """
    if template_bgr is None or inspection_bgr is None:
        raise ValueError("Both template and inspection images are required.")
    if template_bgr.ndim != 3 or inspection_bgr.ndim != 3:
        raise ValueError("Expected color images in OpenCV BGR format.")
    if template_bgr.shape[2] != 3 or inspection_bgr.shape[2] != 3:
        raise ValueError("Expected three-channel OpenCV BGR images.")
    if template_bgr.size == 0 or inspection_bgr.size == 0:
        raise ValueError("Input images must not be empty.")
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between 0 and 1.")
    if ransac_reprojection_threshold <= 0:
        raise ValueError("ransac_reprojection_threshold must be positive.")
    norms = {"L2": cv2.NORM_L2, "HAMMING": cv2.NORM_HAMMING}
    norm_name = str(descriptor_norm).upper()
    if norm_name not in norms:
        raise ValueError("descriptor_norm must be 'L2' or 'HAMMING'.")
    if minimum_good_matches < 4:
        raise ValueError("minimum_good_matches must be at least 4.")

    detector = cv2.AKAZE_create()
    template_keypoints, template_descriptors = detector.detectAndCompute(
        template_bgr, None
    )
    inspection_keypoints, inspection_descriptors = detector.detectAndCompute(
        inspection_bgr, None
    )
    if template_descriptors is None or inspection_descriptors is None:
        raise RegistrationError("AKAZE could not find descriptors in both images.")

    matcher = cv2.BFMatcher(norms[norm_name], crossCheck=False)
    pairs = matcher.knnMatch(template_descriptors, inspection_descriptors, k=2)
    good_matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio_threshold * second.distance
    ]
    if len(good_matches) < minimum_good_matches:
        raise RegistrationError(
            "At least {} reliable AKAZE matches are required; found {}.".format(
                minimum_good_matches, len(good_matches)
            )
        )

    template_points = np.float32(
        [template_keypoints[match.queryIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)
    inspection_points = np.float32(
        [inspection_keypoints[match.trainIdx].pt for match in good_matches]
    ).reshape(-1, 1, 2)

    if rng_seed is not None:
        cv2.setRNGSeed(int(rng_seed))
    homography, inlier_mask = cv2.findHomography(
        template_points,
        inspection_points,
        cv2.RANSAC,
        float(ransac_reprojection_threshold),
    )
    if homography is None or inlier_mask is None:
        raise RegistrationError("RANSAC could not estimate a homography.")

    height, width = template_bgr.shape[:2]
    aligned = cv2.warpPerspective(
        inspection_bgr,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    return Alignment(
        aligned_inspection=aligned,
        homography=homography,
        good_match_count=len(good_matches),
        inlier_count=int(np.count_nonzero(inlier_mask)),
    )
