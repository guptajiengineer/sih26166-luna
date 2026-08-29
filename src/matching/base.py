"""Cross-scale tie-point matching strategies."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import cv2
import numpy as np

from src.models.domain import Keypoint, TiePoint
from src.patterns import Strategy


class Matcher(Strategy[tuple[list[Keypoint], list[Keypoint], dict], list[TiePoint]]):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def execute(self, context: tuple[list[Keypoint], list[Keypoint], dict]) -> list[TiePoint]:
        ref_kps, mov_kps, params = context
        return self.match(ref_kps, mov_kps, params)

    @abstractmethod
    def match(
        self, ref_kps: list[Keypoint], mov_kps: list[Keypoint], params: dict[str, Any]
    ) -> list[TiePoint]:
        ...


def _descriptor_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    min_len = min(len(a), len(b))
    if min_len == 0:
        return float("inf")
    a, b = a[:min_len], b[:min_len]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float(np.linalg.norm(a - b))
    return float(1.0 - np.dot(a, b) / (na * nb))


def _ratio_test_matches(
    ref_kps: list[Keypoint],
    mov_kps: list[Keypoint],
    ratio: float = 0.75,
) -> list[TiePoint]:
    tie_points: list[TiePoint] = []
    for rk in ref_kps:
        if rk.descriptor is None:
            continue
        dists = [
            (_descriptor_distance(rk.descriptor, mk.descriptor), mk)
            for mk in mov_kps
            if mk.descriptor is not None
        ]
        if len(dists) < 2:
            continue
        dists.sort(key=lambda x: x[0])
        best_d, best_mk = dists[0]
        second_d, _ = dists[1]
        if best_d < ratio * second_d:
            confidence = 1.0 - best_d
            tie_points.append(
                TiePoint(ref_x=rk.x, ref_y=rk.y, mov_x=best_mk.x, mov_y=best_mk.y, confidence=confidence)
            )
    return tie_points
