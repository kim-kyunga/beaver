from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

from src.evaluation.evaluation_utils.inference import InferenceOutput


class BaseEvaluator(ABC):
    name: str  # used as key in results dict

    @abstractmethod
    def process_batch(self, batch_out: InferenceOutput, batch_ids: List[str]): ...

    @abstractmethod
    def finalize(self) -> Dict[str, float] | None:
        """Return dict of metrics (may be empty)."""
        ...
