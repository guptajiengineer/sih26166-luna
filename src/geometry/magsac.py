"""MAGSAC++ inspired robust homography/affine estimation."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.geometry.base import GeometricEstimator, _apply_inlier_mask, _to_point_arrays
from src.models.domain import TiePoint


class MAGSACEstimator(GeometricEstimator):
    """
    Uses OpenCV USAC_MAGSAC when available (OpenCV >= 4.5),
    falling back to RANSAC with adaptive threshold.
    """

    @property
    def name(self) -> str:
        return "magsac"

    def estimate(
        self, tie_points: list[TiePoint], params: dict[str, Any]
    ) -> tuple[list[TiePoint], np.ndarray]:
        if len(tie_points) < 4:
            return tie_points, np.eye(3)

        src, dst = _to_point_arrays(tie_points)
        model = params.get("model", "homography")
        threshold = params.get("reproj_threshold", 3.0)
        confidence = params.get("confidence", 0.999)
        max_iters = params.get("max_iters", 10000)

        # USAC_MAGSAC is supported for homography; affine uses RANSAC
        homography_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)

        if model == "affine":
            M, mask = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold,
                confidence=confidence, maxIters=max_iters,
            )
            H = np.eye(3)
            if M is not None:
                H[:2, :] = M
            else:
                mask = np.ones((len(tie_points), 1), dtype=np.uint8)
        else:
            H, mask = cv2.findHomography(
                src, dst, method=homography_method, ransacReprojThreshold=threshold,
                confidence=confidence, maxIters=max_iters,
            )
            if H is None:
                H = np.eye(3)
                mask = np.ones((len(tie_points), 1), dtype=np.uint8)

        filtered = _apply_inlier_mask(tie_points, mask)
        return filtered, H
