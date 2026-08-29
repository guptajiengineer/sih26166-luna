"""Contour-based illumination-invariant features."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.features.base import FeatureExtractor, _to_gray
from src.models.domain import Keypoint, LunarImage


class ContourFeatureExtractor(FeatureExtractor):
    """Extracts keypoints at high-curvature contour vertices."""

    @property
    def name(self) -> str:
        return "contour"

    def extract(self, image: LunarImage, params: dict[str, Any]) -> list[Keypoint]:
        gray = _to_gray(image.data)
        gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Adaptive threshold handles varying illumination
        block = params.get("block_size", 51)
        C = params.get("C", 5)
        binary = cv2.adaptiveThreshold(
            gray_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        min_length = params.get("min_contour_length", 30)
        max_kp = params.get("max_keypoints", 1500)
        keypoints: list[Keypoint] = []

        for cnt in contours:
            if cv2.arcLength(cnt, False) < min_length:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=2.0, closed=False)
            for pt in approx:
                x, y = float(pt[0][0]), float(pt[0][1])
                patch = gray_u8[max(0, int(y) - 8): int(y) + 8, max(0, int(x) - 8): int(x) + 8]
                desc = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA).flatten().astype(np.float32)
                keypoints.append(Keypoint(x=x, y=y, descriptor=desc, response=1.0, source=self.name))
                if len(keypoints) >= max_kp:
                    return keypoints
        return keypoints
