"""Common single-pass inference utilities.
(This section is unchanged – pasted here for completeness.)"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import torch
from torch import Tensor
from torch.amp.autocast_mode import autocast


@dataclass
class InferenceOutput:
    site_logits: Tensor  # [B, n_site]
    em_logits: Tensor  # [B, n_em]
    text_logits: Tensor  # [B, T, vocab]
    embedding: Tensor | None  # [B, D] or None
    extra: Dict[str, Any] | None = None


class InferenceRunner:
    """Guarantees **one** forward pass per batch and returns a structured object."""

    def __init__(
        self,
        network: torch.nn.Module,
        device: torch.device | str,
        precision: torch.dtype = torch.float16,
    ):
        self.network = network
        self.device = device
        self.precision = precision

    def __call__(self, img: Tensor, lens: Tensor) -> InferenceOutput:
        self.network.eval()
        with autocast(dtype=self.precision, device_type=str(self.device)) and torch.inference_mode():
            raw = self.network(img, lens)
        return InferenceOutput(
            site_logits=raw["site_logits"],
            em_logits=raw["em_logits"],
            text_logits=raw["text_logits"],
            embedding=raw["embeddings"],
            extra={
                k: v
                for k, v in raw.items()
                if k not in {"site_logits", "em_logits", "text_logits", "embeddings"}
            },
        )
