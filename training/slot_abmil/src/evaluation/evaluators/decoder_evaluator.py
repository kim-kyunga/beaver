from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

from torch.nn import functional as F
import torch
from src.evaluation.evaluation_utils.inference import InferenceOutput
from src.evaluation.evaluators.base_evaluator import BaseEvaluator
from src.evaluation.evaluators.REG_evaluator import REG_Evaluator
from src.utils.text_utils import decode_text_pred
from src.utils.text_utils import id_to_tiffid


class TextReportEvaluator(BaseEvaluator):
    """Writes prediction JSON and optionally computes REG_Evaluator score."""

    def __init__(
        self,
        *,
        idx_to_word: Dict[int, str],
        idx_to_site: Dict[int, str],
        idx_to_EM: Dict[int, str],
        pad_idx: int,
        out_path: Path | None,
        reg_evaluator: REG_Evaluator | None,
        ground_truth_json: Path,
        save_json: bool = True,
        disable_TQDM: bool = True
    ):

        self.decode_text_pred = decode_text_pred
        self.idx_to_word = idx_to_word
        self.idx_to_site = idx_to_site
        self.idx_to_em = idx_to_EM
        self.pad_idx = pad_idx
        self.save_json = save_json
        self.out_path = out_path
        self.reg_evaluator = reg_evaluator
        self.gt_json = ground_truth_json
        self.disable_TQDM = disable_TQDM

        # running state
        self._predictions: List[Dict[str, str]] = []

    # ----- API -----
    name = "Decoder Text Report"

    def process_batch(self, batch_out: InferenceOutput, batch_ids: List[str]):
        text_preds =  batch_out.text_logits.cpu() # [B, T, V]
        site_pred = torch.argmax(batch_out.site_logits, dim=1)
        EM_pred = torch.argmax(batch_out.em_logits, dim=1)
        B = text_preds.shape[0]
        for i in range(B):
            site = self.idx_to_site[site_pred[i].item()]
            EM = self.idx_to_em[EM_pred[i].item()]
            text = self.decode_text_pred(text_preds[i], self.idx_to_word, self.pad_idx)

            self._predictions.append(
                {"id": batch_ids[i] + ".tiff", "report": f"{site}, {EM};{text}"}
            )

    def finalize(self):
        if self.save_json and self.out_path is not None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_path, "w") as f:
                json.dump(self._predictions, f, indent=2)

        metrics = {}
        if self.reg_evaluator and self.gt_json is not None:
            assert (
                self.out_path is not None
            ), "Output path must be set for REG evaluation."
            score, part_scores = self.reg_evaluator.evaluate(
                self.gt_json, self._predictions, self.disable_TQDM
            )
            metrics["overall_score"] = score
            metrics["part_score"] = part_scores
        self._predictions = []
        return metrics
