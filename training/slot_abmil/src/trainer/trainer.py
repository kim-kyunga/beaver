"""
Trainer with optional slot supervision.

slot_loss는 organ별/qid별 CrossEntropy의 평균.
slot OFF면(slot_cfg=None) 기존 baseline과 동일.

총 손실:
    loss = site_loss + EM_loss + slot_loss * slot_loss_weight + text_loss

train_epoch는 새 collate 9-tuple을 unpack:
    (img, lens, site, EM, text, ids, sample_organ_keys, slot_labels, slot_mask)
"""
import time
from typing import Union, Dict, Tuple, Optional, List

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp.autocast_mode import autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.amp.grad_scaler import GradScaler

from src.conf.schema import AppCfg
from src.network.model import Network
from src.utils.text_utils import build_decoder_io
from src.trainer.training_utils import get_loss_functions


class Trainer:
    def __init__(
        self,
        network: Network,
        config: AppCfg,
        idx_to_site: Dict[int, str],
        idx_to_extraction: Dict[int, str],
        idx_to_word: Dict[int, str],
    ):
        self.network = network
        self.config = config
        self.opt = AdamW(
            network.parameters(),
            config.training.start_lr,
            weight_decay=config.training.weight_decay,
        )
        # add scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=config.training.epochs, eta_min=1e-6
        )
        self.precision = (
            torch.float16 if config.training.precision == "fp16" else torch.float32
        )
        self.use_amp = self.precision == torch.float16
        assert self.use_amp, f"Amp is not in use, that is bad"

        self.site_loss_fn, self.extraction_loss_fn, self.text_loss_fn = (
            get_loss_functions(
                json_path=config.data.train_json_path,
                idx_to_site=idx_to_site,
                idx_to_extraction=idx_to_extraction,
                idx_to_word=idx_to_word,
                device=config.model.train_device,
                pad_idx=config.text.pad_idx,
            )
        )
        # slot CE: -1 라벨은 무시
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.slot_loss_weight: float = float(
            getattr(config.data, "slot_loss_weight", 1.0)
        )
        self.use_slot: bool = bool(
            getattr(config.data, "use_slot_supervision", False)
        )

        self.scaler: Union[GradScaler, None] = None
        if self.use_amp:
            self.scaler = GradScaler()

    # ─── slot loss 계산 ─────────────────────────────────────────────────────
    def compute_slot_loss(
        self,
        slot_logits: Dict[str, Dict[str, torch.Tensor]],
        slot_labels: Dict[str, torch.Tensor],
        slot_mask: Dict[str, torch.Tensor],
        sample_organ_keys: List[str],
    ) -> Tuple[torch.Tensor, int]:
        """
        slot_logits[organ_key][qid] : [B, n_classes]
        slot_labels[organ_key]      : [B, K_organ], -1 where invalid
        slot_mask[organ_key]        : [B, K_organ], True where valid
        반환: (slot_loss_tensor, n_terms)  — n_terms == 0이면 loss는 0 텐서
        """
        device = next(self.network.parameters()).device
        zero = torch.zeros((), device=device)
        if not slot_logits:
            return zero, 0

        losses = []
        for organ_key, per_q_logits in slot_logits.items():
            if organ_key not in slot_labels or organ_key not in slot_mask:
                continue
            labels_organ = slot_labels[organ_key].to(device)   # [B, K]
            mask_organ = slot_mask[organ_key].to(device)       # [B, K]
            qids = list(per_q_logits.keys())
            for j, qid in enumerate(qids):
                logits = per_q_logits[qid]                     # [B, n_classes]
                if j >= labels_organ.shape[1]:
                    continue
                col_mask = mask_organ[:, j]                    # [B]
                if not torch.any(col_mask):
                    continue
                col_labels = labels_organ[:, j]                # [B]
                # mask True인 row만 loss
                valid_logits = logits[col_mask]
                valid_labels = col_labels[col_mask]
                if valid_labels.numel() == 0:
                    continue
                ce = self.slot_loss_fn(valid_logits, valid_labels)
                losses.append(ce)

        if not losses:
            return zero, 0
        slot_loss = torch.stack(losses).mean()
        return slot_loss, len(losses)

    def train_batch(
        self,
        img: torch.Tensor,
        lens: torch.Tensor,
        site_label: torch.Tensor,
        EM_label: torch.Tensor,
        text_label: torch.Tensor,
        sample_organ_keys: Optional[List[str]] = None,
        slot_labels: Optional[Dict[str, torch.Tensor]] = None,
        slot_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[float, float, float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        img, lens, site_label, EM_label, text_label = (
            img.to(self.config.model.train_device),
            lens.to(self.config.model.train_device),
            site_label.to(self.config.model.train_device),
            EM_label.to(self.config.model.train_device),
            text_label.to(self.config.model.train_device),
        )

        decoder_in, decoder_target = build_decoder_io(text_label)

        with autocast(
            dtype=self.precision,
            device_type=str(self.config.model.train_device),
            enabled=self.use_amp,
        ):
            output = self.network(img, lens=lens, target_seq=decoder_in)
            site_pred, EM_pred, text_pred, embeddings = (
                output["site_logits"],
                output["em_logits"],
                output["text_logits"],
                output["embeddings"],
            )
            if len(site_pred.shape) == 1:
                site_pred = site_pred.unsqueeze(0)
                EM_pred = EM_pred.unsqueeze(0)
                text_pred = text_pred.unsqueeze(0)
            site_loss = self.site_loss_fn(site_pred, site_label)
            EM_loss = self.extraction_loss_fn(EM_pred, EM_label)
            text_loss = self.text_loss_fn(
                text_pred.view(-1, text_pred.shape[-1]), decoder_target.view(-1)
            )

            # slot loss
            slot_loss_tensor = torch.zeros((), device=site_pred.device)
            if self.use_slot and ("slot_logits" in output):
                if slot_labels is not None and slot_mask is not None:
                    slot_loss_tensor, _ = self.compute_slot_loss(
                        slot_logits=output["slot_logits"],
                        slot_labels=slot_labels,
                        slot_mask=slot_mask,
                        sample_organ_keys=sample_organ_keys or [],
                    )

            loss: torch.Tensor = (
                site_loss
                + EM_loss
                + slot_loss_tensor * self.slot_loss_weight
                + text_loss
            )

        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            clip_grad_norm_(
                self.network.parameters(), self.config.training.max_grad_norm
            )
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            raise ValueError(
                "AMP is not enabled, but it should be. Please check your configuration."
            )
        self.opt.zero_grad(set_to_none=True)

        site_loss = site_loss.detach().cpu().numpy().item()
        EM_loss = EM_loss.detach().cpu().numpy().item()
        text_loss = text_loss.detach().cpu().numpy().item()
        slot_loss_val = float(slot_loss_tensor.detach().cpu().numpy().item())
        embeddings = embeddings.detach().cpu()
        site_preds = site_pred.argmax(-1).detach().cpu()
        EM_preds = EM_pred.argmax(-1).detach().cpu()
        return (
            site_loss,
            EM_loss,
            text_loss,
            slot_loss_val,
            embeddings,
            site_preds,
            EM_preds,
        )

    def train_epoch(self, dataloader: DataLoader):
        site_loss_vals = []
        EM_loss_vals = []
        text_loss_vals = []
        slot_loss_vals = []
        embedding_list = []
        id_list = []
        site_pred_list = []
        em_pred_list = []
        self.network.train()

        for batch in tqdm(
            dataloader,
            desc="Training batch",
            leave=False,
            position=1,
            disable=self.config.other.disable_TQDM,
        ):
            # 새 collate는 9-tuple
            (
                img,
                lens,
                site_label,
                EM_label,
                text_label,
                ids,
                sample_organ_keys,
                slot_labels,
                slot_mask,
            ) = batch

            (
                site_loss,
                EM_loss,
                text_loss,
                slot_loss,
                embeddings,
                site_preds,
                EM_preds,
            ) = self.train_batch(
                img,
                lens,
                site_label,
                EM_label,
                text_label,
                sample_organ_keys=sample_organ_keys,
                slot_labels=slot_labels,
                slot_mask=slot_mask,
            )
            site_loss_vals.append(site_loss)
            EM_loss_vals.append(EM_loss)
            text_loss_vals.append(text_loss)
            slot_loss_vals.append(slot_loss)
            embedding_list.append(embeddings)
            id_list.extend(ids)
            site_pred_list.append(site_preds)
            em_pred_list.append(EM_preds)

        embeddings = torch.cat(embedding_list)
        site_pred = torch.cat(site_pred_list)
        em_pred = torch.cat(em_pred_list)

        self.scheduler.step()

        return {
            "site_loss": sum(site_loss_vals) / len(site_loss_vals),
            "extraction_loss": sum(EM_loss_vals) / len(EM_loss_vals),
            "text_loss": sum(text_loss_vals) / len(text_loss_vals),
            "slot_loss": (
                sum(slot_loss_vals) / len(slot_loss_vals)
                if slot_loss_vals
                else 0.0
            ),
            "embeddings": embeddings,
            "ids": id_list,
            "site_preds": site_pred,
            "em_preds": em_pred,
        }
