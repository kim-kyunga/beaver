from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import torch
from torch import Tensor
from torch.nn import functional as F
from src.evaluation.evaluation_utils.inference import InferenceOutput
from src.evaluation.evaluators.base_evaluator import BaseEvaluator
from src.evaluation.evaluators.REG_evaluator import REG_Evaluator


class EmbeddingMatchEvaluator(BaseEvaluator):
    """Matches each *test* embedding to the nearest *train* embedding, optionally
    filtered by predicted (site, EM) class. Uses cosine similarity.
    """

    name = "Embedding Matcher"

    def __init__(
        self,
        *,
        train_embeddings: List[Tensor],
        train_ids: List[str],
        train_site_pred: List[torch.Tensor],
        train_em_pred: List[torch.Tensor],
        out_path: Path | None,
        reg_evaluator: REG_Evaluator | None = None,
        ground_truth_json: Path | None = None,
        save_json: bool = True,
        match_site_and_em: bool = True,
        disable_TQDM: bool = True
    ):
        with open(str(ground_truth_json), "r") as f:
            gt_data = json.load(f)
            id_to_report = {item["id"]: item["report"] for item in gt_data}
        self.train_embeddings = F.normalize(train_embeddings.float(), dim=1)
        self.train_ids = train_ids
        self.train_site_pred = train_site_pred
        self.train_em_pred = train_em_pred
        self.id_to_report = id_to_report
        self.out_path = out_path
        self.reg_evaluator = reg_evaluator
        self.gt_json = ground_truth_json
        self.save_json = save_json
        self.match_site_and_em = match_site_and_em
        self.disable_TQDM = disable_TQDM

        # Build (site, em) -> indices mapping once
        self._bin = defaultdict(list)
        for idx, (s, e) in enumerate(zip(self.train_site_pred, self.train_em_pred)):
            self._bin[(int(s), int(e))].append(idx)

        self._predictions = []

        if self.match_site_and_em:
            self.name += " (filtered by site and EM)"

    def process_batch(self, batch_out: InferenceOutput, batch_ids: List[str]):
        emb = F.normalize(batch_out.embedding.float(), dim=1)  # [B, D]
        site_pred = batch_out.site_logits.argmax(-1)  # [B]
        em_pred = batch_out.em_logits.argmax(-1)  # [B]

        for i in range(emb.size(0)):
            if self.match_site_and_em:
                key = (int(site_pred[i]), int(em_pred[i]))
                cand_idx = self._bin[key]
                if len(cand_idx) == 0:
                    cand_idx = list(range(len(self.train_ids)))
                    candidate_train_embeddings = self.train_embeddings
                else:
                    candidate_train_embeddings = self.train_embeddings[cand_idx]
            else:
                candidate_train_embeddings = (
                    self.train_embeddings
                )  # fallback: full bank
                cand_idx = list(range(len(self.train_ids)))
            similarity = F.cosine_similarity(
                emb[i].unsqueeze(0).cpu(), candidate_train_embeddings.cpu()
            )
            try:
                best_idx = cand_idx[int(similarity.argmax().item())]
                best_id = self.train_ids[best_idx] + ".tiff"
                self._predictions.append(
                    {"id": batch_ids[i] + ".tiff", "report": self.id_to_report[best_id]}
                )
            except Exception as e:
                print(f"Embedding shape: {emb[i].unsqueeze(0).cpu().shape}")
                print(f"\nSimilarity: {similarity.shape}")
                print(
                    f"Candidate train embeddings shape: {candidate_train_embeddings.shape}"
                )
                print(f"Candidate indices: {cand_idx}")
                raise e

    def finalize(self):
        if self.save_json and self.out_path is not None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_path, "w") as f:
                json.dump(self._predictions, f, indent=2)

        metrics = {}
        if self.reg_evaluator and self.gt_json is not None:
            score, part_scores = self.reg_evaluator.evaluate(
                self.gt_json, self._predictions, self.disable_TQDM
            )
            metrics["overall_score"] = score
            metrics["part_score"] = part_scores
        return metrics
