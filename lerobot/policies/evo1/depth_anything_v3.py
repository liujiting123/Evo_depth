"""Thin wrapper around Depth-Anything-V3 used by EVO1's DA3 branch.

The wrapped model is loaded from a local checkpoint directory. The path is
resolved in this order:

1. ``checkpoint_dir`` argument explicitly passed to ``DAV3Module``.
2. The ``EVO1_DA3_CHECKPOINT_DIR`` environment variable.
3. Raises ``FileNotFoundError`` — the caller must provide a checkpoint.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn


def _resolve_checkpoint_dir(checkpoint_dir: str | None) -> str:
    path = checkpoint_dir or os.environ.get("EVO1_DA3_CHECKPOINT_DIR")
    if not path:
        raise FileNotFoundError(
            "DAV3Module requires a Depth-Anything-V3 checkpoint directory. "
            "Pass checkpoint_dir=... or set EVO1_DA3_CHECKPOINT_DIR."
        )
    if not os.path.isdir(path):
        raise FileNotFoundError(f"DA3 checkpoint dir does not exist: {path}")
    return path


class DAV3Module(nn.Module):
    """Extract DPT token features from a batch of images via Depth-Anything-V3.

    Mirrors the original ``Evo_depth`` wrapper: the depth model is kept frozen
    by default (callers handle requires_grad via ``EVO1.set_finetune_flags``)
    and only its ``_get_dpt_embeddings`` path is used — no geometry decoding.
    """

    def __init__(self, checkpoint_dir: str | None = None):
        super().__init__()
        ckpt = _resolve_checkpoint_dir(checkpoint_dir)
        # Lazy import: DA3 pulls a deep dependency tree, only load on demand.
        from lerobot.policies.evo1.depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(ckpt, local_files_only=True)

    def extract_features(self, image_tensor: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        feats = self.model._get_dpt_embeddings(image_tensor)
        if image_mask is None:
            return feats
        selected = []
        for i in range(image_mask.shape[-1]):
            if bool(image_mask[..., i].any()):
                selected.append(feats[:, i, :, :])
        if not selected:
            return feats
        return torch.stack(selected, dim=1)

    def forward(self, image_tensor: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
        return self.extract_features(image_tensor, image_mask)
