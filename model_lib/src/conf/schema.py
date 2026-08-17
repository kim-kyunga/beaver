"""
rectum change + slot supervision

DataCfg에 slot supervision용 필드 6개 추가:
  - use_slot_supervision  : slot 학습 ON/OFF (False면 기존 baseline 그대로)
  - slot_cot_dir          : organ별 CSV가 저장된 디렉터리
  - organ_grouping        : "merged" or "separate"
  - slot_skip_first_n     : 앞에서 skip할 question 수 (Q01/Q02 = 2)
  - slot_skip_last_n      : 뒤에서 skip할 question 수 (final diagnosis = 1)
  - slot_loss_weight      : slot loss 가중치
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, cast

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


# ─────────────────────────────────── leaf sections ───────────────────────────
@dataclass
class TrainingCfg:
    epochs: int = cast(int, MISSING)
    bag_size: Optional[int] = cast(Optional[int], MISSING)
    batch_size: int = cast(int, MISSING)
    train_data_part: float = cast(float, MISSING)
    eval_rate: int = cast(int, MISSING)
    precision: str = cast(str, MISSING)
    start_lr: float = cast(float, MISSING)
    seed: int = cast(int, MISSING)
    max_grad_norm: int = cast(int, MISSING)
    weight_decay: float = cast(float, MISSING)


@dataclass
class ModelCfg:
    train_device: str = cast(str, MISSING)
    eval_device: str = cast(str, MISSING)
    num_channels: int = cast(int, MISSING)
    embed_dim: int = cast(int, MISSING)
    ff_dim: int = cast(int, MISSING)
    num_blocks: int = cast(int, MISSING)
    num_layers_transformer: int = cast(int, MISSING)
    num_heads_transformer: int = cast(int, MISSING)
    num_layers_abmil: int = cast(int, MISSING)
    num_attention_heads_abmil: int = cast(int, MISSING)
    trainable_pos_emb: bool = cast(bool, MISSING)
    dropout_abmil: float = cast(float, MISSING)
    slide_feature_encoder_dim: int = cast(int, MISSING)
    model_checkpoint: Optional[str] = cast(Optional[str], MISSING)


@dataclass
class DataCfg:
    data_suffix: str = cast(str, MISSING)
    max_num_datapoints: Optional[int] = cast(Optional[int], MISSING)
    train_json_path: str = cast(str, MISSING)
    train_data_path: str = cast(str, MISSING)
    experiment_path: str = cast(str, MISSING)
    experiment_name: str = cast(str, MISSING)
    site_count_threshold: int = cast(int, MISSING)
    extraction_count_threshold: int = cast(int, MISSING)
    normalize: bool = cast(bool, MISSING)
    mean: list[float] = cast(list[float], MISSING)
    std: list[float] = cast(list[float], MISSING)

    # ─── slot supervision (신규) ───────────────────────────────────────────
    use_slot_supervision: bool = False
    slot_cot_dir: Optional[str] = None
    organ_grouping: str = "merged"   # "merged" or "separate" (Hydra/OmegaConf는 Literal 미지원)
    slot_skip_first_n: int = 2   # Q01(organ), Q02(procedure) — 기존 head와 중복
    slot_skip_last_n: int = 1    # 마지막 question(final diagnosis) — sparse + decoder와 중복
    slot_loss_weight: float = 1.0


@dataclass
class DataloaderCfg:
    num_workers: int = cast(int, MISSING)
    pin_memory: bool = cast(bool, MISSING)
    drop_last: bool = cast(bool, MISSING)
    prefetch_factor: int = cast(int, MISSING)


@dataclass
class TextCfg:
    pad_idx: int = cast(int, MISSING)
    sos_idx: int = cast(int, MISSING)
    eos_idx: int = cast(int, MISSING)
    n_gram: int = cast(int, MISSING)


@dataclass
class OtherCfg:
    dry_run: bool = cast(bool, MISSING)
    verbose: bool = cast(bool, MISSING)
    disable_TQDM: bool = cast(bool, MISSING)


@dataclass
class EvalCfg:
    run_text: bool = cast(bool, MISSING)
    run_embedding: bool = cast(bool, MISSING)
    save_json: bool = cast(bool, MISSING)
    eval_rate: int = cast(int, MISSING)
    test_data_path: Optional[str] = cast(Optional[str], MISSING)


# ─────────────────────────────────── root section ────────────────────────────
@dataclass
class AppCfg:
    training: TrainingCfg = cast(TrainingCfg, MISSING)
    model: ModelCfg = cast(ModelCfg, MISSING)
    data: DataCfg = cast(DataCfg, MISSING)
    dataloader: DataloaderCfg = cast(DataloaderCfg, MISSING)
    text: TextCfg = cast(TextCfg, MISSING)
    eval: EvalCfg = cast(EvalCfg, MISSING)
    other: OtherCfg = cast(OtherCfg, MISSING)


# ───────────────────────── register with Hydra (once) ────────────────────────
cs = ConfigStore.instance()
cs.store(name="app_cfg", node=AppCfg)
