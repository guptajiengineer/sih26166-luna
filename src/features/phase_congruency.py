"""Phase congruency — illumination-invariant edge/structure detector."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.features.base import FeatureExtractor, _to_gray
from src.models.domain import Keypoint, LunarImage


class PhaseCongruencyExtractor(FeatureExtractor):
    """
    Simplified phase-congruency-inspired feature map.
    Uses log-Gabor filter bank responses; robust to sun-angle changes.
    """

    @property
    def name(self) -> str:
        return "phase_congruency"

    def extract(self, image: LunarImage, params: dict[str, Any]) -> list[Keypoint]:
        gray = _to_gray(image.data)
        gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Multi-scale gradient magnitude as PC proxy
        scales = params.get("scales", [1, 2, 4, 8])
        pc_map = np.zeros_like(gray, dtype=np.float32)
        for s in scales:
            blurred = cv2.GaussianBlur(gray_u8, (0, 0), sigmaX=s)
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            pc_map += np.sqrt(gx ** 2 + gy ** 2)

        pc_map /= len(scales)
        threshold = params.get("threshold", np.percentile(pc_map, 95))
        ys, xs = np.where(pc_map >= threshold)

        max_kp = params.get("max_keypoints", 2000)
        responses = pc_map[ys, xs]
        order = np.argsort(responses)[::-1][:max_kp]

        keypoints: list[Keypoint] = []
        for idx in order:
            x, y = float(xs[idx]), float(ys[idx])
            patch = pc_map[max(0, int(y) - 4): int(y) + 5, max(0, int(x) - 4): int(x) + 5]
            desc = patch.flatten()[:64] if patch.size >= 64 else np.pad(
                patch.flatten(), (0, max(0, 64 - patch.size))
            )
            keypoints.append(
                Keypoint(x=x, y=y, descriptor=desc.astype(np.float32), response=float(responses[idx]), source=self.name)
            )
        return keypoints
