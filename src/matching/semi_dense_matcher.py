"""Semi-dense matcher for 20x–300x scale gaps."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.matching.base import Matcher, _ratio_test_matches
from src.models.domain import Keypoint, TiePoint


class SemiDenseMatcher(Matcher):
    """
    Semi-dense matching: descriptor NN + scale-aware coordinate rescaling.
    Bridges OHRC (0.25 m) ↔ TMC-2 (5 m) ↔ IIRS (80 m) gaps.
    """

    @property
    def name(self) -> str:
        return "semi_dense"

    def match(
        self, ref_kps: list[Keypoint], mov_kps: list[Keypoint], params: dict[str, Any]
    ) -> list[TiePoint]:
        scale_ref = params.get("scale_ref", 1.0)
        scale_mov = params.get("scale_mov", 1.0)
        ratio = params.get("ratio_threshold", 0.8)

        # Rescale moving keypoints to reference frame
        scaled_mov = [
            Keypoint(
                x=kp.x * (scale_ref / scale_mov),
                y=kp.y * (scale_ref / scale_mov),
                descriptor=kp.descriptor,
                response=kp.response,
                source=kp.source,
            )
            for kp in mov_kps
        ]

        raw = _ratio_test_matches(ref_kps, scaled_mov, ratio=ratio)

        # Sub-pixel refinement via template matching windows (when images provided)
        ref_img = params.get("ref_image")
        mov_img = params.get("mov_image")
        if ref_img is not None and mov_img is not None:
            raw = self._refine_subpixel(raw, ref_img, mov_img, scale_ref, scale_mov)

        return raw

    def _refine_subpixel(
        self,
        tie_points: list[TiePoint],
        ref_img: np.ndarray,
        mov_img: np.ndarray,
        scale_ref: float,
        scale_mov: float,
        win: int = 11,
    ) -> list[TiePoint]:
        ref_gray = ref_img if ref_img.ndim == 2 else np.mean(ref_img, axis=2)
        mov_gray = mov_img if mov_img.ndim == 2 else np.mean(mov_img, axis=2)
        refined: list[TiePoint] = []

        for tp in tie_points:
            rx, ry = int(tp.ref_x), int(tp.ref_y)
            mx = int(tp.mov_x * scale_mov / scale_ref)
            my = int(tp.mov_y * scale_mov / scale_ref)

            if not (win <= rx < ref_gray.shape[1] - win and win <= ry < ref_gray.shape[0] - win):
                refined.append(tp)
                continue
            if not (win <= mx < mov_gray.shape[1] - win and win <= my < mov_gray.shape[0] - win):
                refined.append(tp)
                continue

            template = ref_gray[ry - win: ry + win + 1, rx - win: rx + win + 1].astype(np.float32)
            search = mov_gray[my - win * 2: my + win * 2 + 1, mx - win * 2: mx + win * 2 + 1].astype(np.float32)
            if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
                refined.append(tp)
                continue

            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            dx = max_loc[0] - win * 2 + win
            dy = max_loc[1] - win * 2 + win

            refined.append(
                TiePoint(
                    ref_x=tp.ref_x,
                    ref_y=tp.ref_y,
                    mov_x=(mx + dx) * scale_ref / scale_mov,
                    mov_y=(my + dy) * scale_ref / scale_mov,
                    confidence=tp.confidence,
                )
            )
        return refined
