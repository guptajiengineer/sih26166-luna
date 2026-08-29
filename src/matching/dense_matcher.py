"""Dense optical-flow based matcher for fine alignment."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.matching.base import Matcher
from src.models.domain import Keypoint, TiePoint


class DenseMatcher(Matcher):
    """Dense Farneback optical flow sampled at keypoint grid."""

    @property
    def name(self) -> str:
        return "dense"

    def match(
        self, ref_kps: list[Keypoint], mov_kps: list[Keypoint], params: dict[str, Any]
    ) -> list[TiePoint]:
        ref_img = params.get("ref_image")
        mov_img = params.get("mov_image")
        if ref_img is None or mov_img is None:
            from src.matching.semi_dense_matcher import SemiDenseMatcher
            return SemiDenseMatcher().match(ref_kps, mov_kps, params)

        ref_gray = ref_img if ref_img.ndim == 2 else np.mean(ref_img, axis=2).astype(np.uint8)
        mov_gray = mov_img if mov_img.ndim == 2 else np.mean(mov_img, axis=2).astype(np.uint8)

        # Resize moving to reference scale if needed
        scale = params.get("scale_mov", 1.0) / params.get("scale_ref", 1.0)
        if abs(scale - 1.0) > 0.01:
            new_w = max(1, int(mov_gray.shape[1] * scale))
            new_h = max(1, int(mov_gray.shape[0] * scale))
            mov_gray = cv2.resize(mov_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        h = min(ref_gray.shape[0], mov_gray.shape[0])
        w = min(ref_gray.shape[1], mov_gray.shape[1])
        ref_gray, mov_gray = ref_gray[:h, :w], mov_gray[:h, :w]

        flow = cv2.calcOpticalFlowFarneback(
            ref_gray.astype(np.float32),
            mov_gray.astype(np.float32),
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )

        step = params.get("grid_step", 16)
        tie_points: list[TiePoint] = []
        for y in range(step // 2, h, step):
            for x in range(step // 2, w, step):
                dx, dy = flow[y, x]
                tie_points.append(
                    TiePoint(ref_x=float(x), ref_y=float(y), mov_x=float(x + dx), mov_y=float(y + dy), confidence=0.8)
                )
        return tie_points
