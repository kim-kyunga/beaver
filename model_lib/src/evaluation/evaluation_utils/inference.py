
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import torch
from torch import Tensor
from torch.amp.autocast_mode import autocast


@dataclass
class InferenceOutput:
    site_logits: Tensor
    em_logits: Tensor
    text_logits: Tensor
    embedding: Tensor | None
    extra: Dict[str, Any] | None = None


class InferenceRunner:

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
