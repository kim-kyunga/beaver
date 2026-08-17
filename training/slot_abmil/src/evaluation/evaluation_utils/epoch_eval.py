"""A single helper that encapsulates *all* test‑set evaluation logic so
`train.py` stays concise. Call once per epoch.

Example:
    metrics_fast, metrics_reg = run_epoch_evaluation(
        infer_runner=infer,
        test_loader=test_dataloader,
        loss_fns=(site_loss_fn, extraction_loss_fn, text_loss_fn),
        pad_idx=config.text.pad_idx,
        do_reg_eval=((epoch+1) % config.eval.eval_rate == 0),
        evaluators_factory=lambda: build_reg_evaluators(...)
    )
"""

from typing import Tuple, Callable, Dict, List

from tqdm import tqdm
import torch
from torch import Tensor
from torch.amp.autocast_mode import autocast

# Re‑use utilities already defined in evaluators.py / inference.py
from src.evaluation.evaluation_utils.inference import (
    InferenceRunner,
    InferenceOutput,
)
from src.evaluation.evaluators.base_evaluator import BaseEvaluator
from src.evaluation.evaluation_utils.eval_loop import EvaluationLoop
from src.conf.schema import AppCfg
from src.utils.text_utils import get_num_correct_tokens


def _update_running_acc(
    batch_out: InferenceOutput,
    site_lbl: Tensor,
    em_lbl: Tensor,
    txt_lbl: Tensor,
    pad_idx: int,
    acc: Dict[str, int],
    config: AppCfg,
):
    site_pred = batch_out.site_logits.argmax(-1)
    em_pred = batch_out.em_logits.argmax(-1)
    txt_pred = batch_out.text_logits

    acc["correct_site"] += int((site_pred == site_lbl).sum().item())
    acc["correct_em"] += int((em_pred == em_lbl).sum().item())

    mask = txt_lbl != pad_idx
    acc["correct_text"] += get_num_correct_tokens(txt_pred, txt_lbl, config)
    acc["total_tokens"] += int(mask.sum().item())


def run_epoch_evaluation(
    infer_runner: InferenceRunner,
    test_loader: torch.utils.data.DataLoader,
    pad_idx: int,
    do_reg_eval: bool,
    evaluators_factory: Callable[[], List[BaseEvaluator]] | None,
    config: AppCfg,
):
    """Runs the fast per‑epoch metrics *and* (optionally) the slow REG evaluators.

    Returns
    -------
    fast_metrics : dict  – always (site_acc, em_acc, text_token_acc, losses).
    reg_metrics  : dict or None – returned only when `do_reg_eval` is True.
    """

    running = dict(correct_site=0, correct_em=0, correct_text=0, total_tokens=0)

    # Initialise slow evaluators lazily
    eval_loop = None
    if do_reg_eval and evaluators_factory:
        eval_loop = EvaluationLoop(evaluators_factory())

    for img, lens, site_lbl, em_lbl, txt_lbl, ids, *_ in tqdm(
        test_loader,
        desc="Running evaluation",
        position=1,
        leave=False,
        disable=config.other.disable_TQDM,
    ):
        img, lens, site_lbl, em_lbl, txt_lbl = (
            img.to(infer_runner.device),
            lens.to(infer_runner.device),
            site_lbl.to(infer_runner.device),
            em_lbl.to(infer_runner.device),
            txt_lbl.to(infer_runner.device),
        )
        batch_out = infer_runner(img, lens)

        # --- accuracies ---
        _update_running_acc(
            batch_out, site_lbl, em_lbl, txt_lbl, pad_idx, running, config
        )

        if eval_loop:
            eval_loop.process_batch(batch_out, ids)

    num_items = len(test_loader.dataset)
    fast_metrics = {
        "site_acc": running["correct_site"] / num_items,
        "em_acc": running["correct_em"] / num_items,
        "text_token_acc": running["correct_text"] / num_items,
    }

    reg_metrics = eval_loop.finalize() if eval_loop else None
    return fast_metrics, reg_metrics
