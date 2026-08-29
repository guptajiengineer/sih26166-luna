"""Illumination-invariant feature extraction strategies."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from src.models.domain import Keypoint, LunarImage
from src.patterns import Strategy


class FeatureExtractor(Strategy[tuple[LunarImage, dict], list[Keypoint]]):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def execute(self, context: tuple[LunarImage, dict]) -> list[Keypoint]:
        image, params = context
        return self.extract(image, params)

    @abstractmethod
    def extract(self, image: LunarImage, params: dict[str, Any]) -> list[Keypoint]:
        ...


def _to_gray(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return data.astype(np.float32)
    if data.ndim == 3:
        return np.mean(data, axis=2).astype(np.float32)
    raise ValueError(f"Unsupported image shape: {data.shape}")
