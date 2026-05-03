"""Image augmentation matching Evo_depth's ``--use_augmentation`` behaviour.

Original Evo_depth logic (dataset/lerobot_dataset_pretrain_mp.py:191-438):
  Per-image: 50% chance apply all four transforms, 50% just resize.

  aug = [RandomResizedCrop(448, scale=(0.95,1.0)), RandomRotation(±5°),
         ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)]
  basic = [Resize(448)]

This module uses torchvision.transforms.v2 so it works directly on
float32 CHW tensors (lerobot datasets return tensors, not PIL).
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms.functional import InterpolationMode


class EvoDepthImageTransforms(torch.nn.Module):
    """Exact replica of Evo_depth's per-image augmentation pipeline.

    50% chance per image: full aug suite, or plain resize only.
    """

    def __init__(
        self,
        image_size: int = 448,
        aug_probability: float = 0.5,
        crop_scale: tuple[float, float] = (0.95, 1.0),
        rotation_degrees: tuple[float, float] = (-5.0, 5.0),
        brightness: float = 0.3,
        contrast: float = 0.4,
        saturation: float = 0.5,
        hue: float = 0.08,
    ):
        super().__init__()
        self.aug_probability = aug_probability

        self.basic_transform = v2.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        self.aug_transform = v2.Compose([
            v2.RandomResizedCrop(
                image_size,
                scale=crop_scale,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomRotation(
                degrees=rotation_degrees,
                interpolation=InterpolationMode.BILINEAR,
            ),
            v2.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
        ])

    def forward(self, img: Any) -> Any:
        if random.random() < self.aug_probability:
            return self.aug_transform(img)
        return self.basic_transform(img)
