"""Deep learned invariant embeddings (lightweight CNN fallback)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.features.base import FeatureExtractor, _to_gray
from src.models.domain import Keypoint, LunarImage


class DeepEmbeddingExtractor(FeatureExtractor):
    """
    Deep embedding extractor using ORB/SIFT fallback when torch unavailable,
    or torchvision backbone when torch is present.
    """

    @property
    def name(self) -> str:
        return "deep_embedding"

    def extract(self, image: LunarImage, params: dict[str, Any]) -> list[Keypoint]:
        gray = _to_gray(image.data)
        gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as T

            backbone = params.get("_backbone")
            if backbone is None:
                backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
                backbone.classifier = torch.nn.Identity()
                backbone.eval()
                params["_backbone"] = backbone

            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Grid sampling for semi-dense embeddings
            step = params.get("grid_step", 32)
            h, w = gray_u8.shape
            keypoints: list[Keypoint] = []
            with torch.no_grad():
                for y in range(step // 2, h, step):
                    for x in range(step // 2, w, step):
                        patch_size = min(64, h - y, w - x)
                        patch = gray_u8[y: y + patch_size, x: x + patch_size]
                        if patch.size == 0:
                            continue
                        tensor = transform(cv2.cvtColor(
                            cv2.resize(patch, (64, 64)), cv2.COLOR_GRAY2RGB
                        )).unsqueeze(0)
                        emb = backbone(tensor).numpy().flatten()
                        keypoints.append(
                            Keypoint(x=float(x), y=float(y), descriptor=emb, response=1.0, source=self.name)
                        )
            return keypoints[: params.get("max_keypoints", 1000)]

        except ImportError:
            orb = cv2.ORB_create(nfeatures=params.get("max_keypoints", 1000))
            kps, descs = orb.detectAndCompute(gray_u8, None)
            if kps is None:
                return []
            return [
                Keypoint(
                    x=kp.pt[0], y=kp.pt[1],
                    descriptor=descs[i].astype(np.float32) if descs is not None else None,
                    response=kp.response, source=self.name,
                )
                for i, kp in enumerate(kps)
            ]
