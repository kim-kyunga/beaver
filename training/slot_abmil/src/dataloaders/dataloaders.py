"""
Dataloader with optional slot supervision.

기존 baseline 흐름은 그대로 유지하면서 slot label만 추가로 싣는다.
slot 학습이 꺼져 있으면(use_slot_supervision=False) 기존 baseline과 동일하게 동작.

추가된 데이터:
  - slot_labels : dict {organ_key: LongTensor [B, num_trainable_qids_of_organ]}
  - slot_mask   : dict {organ_key: BoolTensor [B, num_trainable_qids_of_organ]}
  - sample_organ_keys : list[str] of length B  (이 샘플이 어느 organ_key에 속하는지)

slot이 없는 샘플(예: rectum/anus가 separate인데 nipple은 site filter에서 빠진 경우)은
  - mask 전체가 False  → trainer가 자동으로 skip
"""
from torch.utils.data import Dataset, DataLoader
import random
import torch
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
import json
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import h5py
from src.conf.schema import AppCfg
from src.dataloaders.data_utils import (
    get_counts,
    get_site_extraction_method_and_text,
    print_info,
    is_good_datapoint,
    get_vocabulary,
    sample_bag,
    get_sorted_word_map,
)
from torchvision import transforms
from PIL import PngImagePlugin
from src.dataloaders.tokenization import tokenizer


LARGE_ENOUGH_NUMBER = 1000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


# ───────────────────────────────────────────────────────────────────────────
# Organ grouping definitions (must mirror reg2026_complete_pipeline.py)
# ───────────────────────────────────────────────────────────────────────────
ORGAN_GROUPS_MERGED = {
    "breast":          ["Breast", "Nipple"],
    "colon":           ["Colon", "Rectum", "Anus"],
    "urinary_bladder": ["Urinary bladder"],
    "lung":            ["Lung"],
    "prostate":        ["Prostate"],
    "uterine_cervix":  ["Uterine cervix"],
    "stomach":         ["Stomach"],
}

ORGAN_GROUPS_SEPARATE = {
    "breast":          ["Breast"],
    "nipple":          ["Nipple"],
    "colon":           ["Colon"],
    "rectum":          ["Rectum"],
    "anus":            ["Anus"],
    "urinary_bladder": ["Urinary bladder"],
    "lung":            ["Lung"],
    "prostate":        ["Prostate"],
    "uterine_cervix":  ["Uterine cervix"],
    "stomach":         ["Stomach"],
}


def get_organ_groups(mode: str) -> Dict[str, List[str]]:
    if mode == "merged":
        return ORGAN_GROUPS_MERGED
    elif mode == "separate":
        return ORGAN_GROUPS_SEPARATE
    else:
        raise ValueError(f"Unknown organ_grouping: {mode!r}")


def build_site_to_organ_key(organ_groups: Dict[str, List[str]]) -> Dict[str, str]:
    """site name(JSON) → organ_key(CSV file prefix) 역매핑."""
    mapping = {}
    for organ_key, variants in organ_groups.items():
        for v in variants:
            mapping[v] = organ_key
    return mapping


def load_slot_data(
    slot_cot_dir: str,
    organ_groups: Dict[str, List[str]],
    skip_first_n: int,
    skip_last_n: int,
    verbose: bool = True,
) -> Tuple[
    Dict[str, List[str]],          # organ_key → trainable_qids
    Dict[str, Dict[str, int]],      # organ_key → {qid: num_classes}
    Dict[str, Dict[str, Dict[str, int]]],  # organ_key → {id_stem: {qid: code}}
]:
    """
    organ별로 encoded CSV와 codebook을 로드해서:
      - trainable_qids[organ]   : 학습 대상 question id 리스트 (Q01,Q02,마지막 제외)
      - num_classes[organ][qid] : 해당 question의 class 수
      - id2labels[organ][id_stem][qid] : case-id별 인코딩된 정답 (없으면 키 부재)
    """
    slot_dir = Path(slot_cot_dir)
    trainable_qids: Dict[str, List[str]] = {}
    num_classes: Dict[str, Dict[str, int]] = {}
    id2labels: Dict[str, Dict[str, Dict[str, int]]] = {}

    for organ_key in organ_groups.keys():
        enc_path = slot_dir / f"{organ_key}_cot_encoded.csv"
        cb_path = slot_dir / f"{organ_key}_label_codebook.csv"
        qdef_path = slot_dir / f"{organ_key}_question_definition.csv"

        if not (enc_path.exists() and cb_path.exists() and qdef_path.exists()):
            if verbose:
                print(f"[slot]   {organ_key}: CSV missing, skipping organ")
            continue

        qdef = pd.read_csv(qdef_path)
        all_qids = qdef["question_id"].tolist()
        # 안전장치: skip이 너무 크면 빈 리스트가 되므로 max로 보호
        end = max(skip_first_n, len(all_qids) - skip_last_n)
        qids = all_qids[skip_first_n:end]
        if len(qids) == 0:
            if verbose:
                print(f"[slot]   {organ_key}: no trainable qids after skip, skipping")
            continue

        cb = pd.read_csv(cb_path)
        cls = {}
        for qid in qids:
            sub = cb[cb["question_id"] == qid]
            if len(sub) == 0:
                # codebook에 라벨이 전혀 없는 question (min_freq 필터로 다 제거된 경우)
                continue
            cls[qid] = int(sub["code"].max()) + 1
        # codebook에 살아남은 qids만 최종 trainable
        qids = [q for q in qids if q in cls]
        if len(qids) == 0:
            if verbose:
                print(f"[slot]   {organ_key}: no labels in codebook after min_freq filter, skipping")
            continue

        trainable_qids[organ_key] = qids
        num_classes[organ_key] = cls

        enc = pd.read_csv(enc_path)
        # encoded CSV의 id는 보통 *.tiff 또는 .tiff 없는 stem.
        # data_path와 매칭하기 위해 .tiff 제거한 stem으로 보관.
        id2labels[organ_key] = {}
        for _, row in enc.iterrows():
            raw_id = str(row["id"])
            stem = raw_id.replace(".tiff", "")
            per_q = {}
            for qid in qids:
                if qid in enc.columns:
                    val = row[qid]
                    if pd.isna(val):
                        continue
                    code = int(val)
                    if code < 0:
                        continue
                    # 만약 데이터에 codebook 최대값보다 큰 코드가 들어있으면 무시 (방어적)
                    if code >= cls[qid]:
                        continue
                    per_q[qid] = code
            id2labels[organ_key][stem] = per_q

        if verbose:
            print(
                f"[slot]   {organ_key}: trainable_qids={len(qids)}, "
                f"cases={len(id2labels[organ_key])}, "
                f"total_classes={sum(cls.values())}"
            )

    return trainable_qids, num_classes, id2labels


class RegDataset(Dataset):
    def __init__(
        self,
        data_paths: List[Path],
        labels: List[Dict[str, str]],
        site_map: Dict[str, int],
        extraction_map: Dict[str, int],
        augment: bool,
        vocabulary: Dict[str, int],
        max_bag_size: int | None,
        disable_TQDM: bool,
        # ─── slot 관련 (None이면 baseline 동작) ──────────────────────────────
        slot_cfg: Optional[Dict] = None,
    ):
        super().__init__()

        site_map = get_sorted_word_map(site_map)
        extraction_map = get_sorted_word_map(extraction_map)

        self.site_map = site_map
        self.extraction_map = extraction_map
        self.max_bag_size = max_bag_size
        self.datapoints = {}
        self.labels = {}
        self.ids = {}
        self.lens = []
        self.suffix = data_paths[0].suffix
        self.data_paths = data_paths
        for i, (data_path, label) in tqdm(
            enumerate(zip(self.data_paths, labels)),
            desc="Reading datapoints",
            leave=False,
            total=len(data_paths),
            dynamic_ncols=True,
            disable=disable_TQDM,
        ):
            self.datapoints[i] = self.get_datapoint(i)
            self.ids[i] = data_path.stem
            self.labels[i] = label

        self.vocabulary = vocabulary
        self.text_labels = {}
        self.max_seq_len = (
            max(len(tokenizer(label["text"])) for label in self.labels.values())
        ) + 2  # +2 for <SOS> and <EOS>

        for idx, label in self.labels.items():
            text_labels = [self.vocabulary["<SOS>"]]
            words = tokenizer(label["text"])
            for word in words:
                if word not in self.vocabulary:
                    raise ValueError(
                        f"Word '{word}' not found in vocabulary. Available words: {list(self.vocabulary.keys())}"
                    )
                text_labels.append(self.vocabulary[word])
            text_labels.append(self.vocabulary["<EOS>"])
            self.text_labels[idx] = self.pad_sequence(text_labels, self.max_seq_len)

        # ─── slot 데이터 ─────────────────────────────────────────────────────
        self.slot_cfg = slot_cfg  # None or dict
        # slot_cfg = {
        #   "site_to_organ_key": {site_name: organ_key},
        #   "trainable_qids":    {organ_key: [qid, ...]},
        #   "num_classes":       {organ_key: {qid: n}},
        #   "id2labels":         {organ_key: {id_stem: {qid: code}}},
        # }

        self.data_mean = torch.tensor([0.9505, 0.9375, 0.9510]).reshape(3, 1, 1)
        self.data_std = torch.tensor([0.0762, 0.1088, 0.0727]).reshape(3, 1, 1)

        if augment and self.suffix == ".png":
            self.augment = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                ]
            )
        else:
            self.augment = None

    def pad_sequence(self, seq: List[int], max_len: int):
        if len(seq) < max_len:
            seq = seq + [self.vocabulary["<PAD>"]] * (max_len - len(seq))
        assert len(seq) == max_len, f"Sequence length mismatch: {len(seq)} != {max_len}"
        return seq

    def read_image(self, img_path: Path):
        with Image.open(img_path) as img:
            img = (
                torch.from_numpy(np.array(img)).permute(2, 0, 1).to(torch.float32) / 256
            )
        return img

    def read_features(self, h5_file: Path):
        with h5py.File(h5_file, "r") as f:
            embedding = f["features"][:]
            embedding = torch.from_numpy(embedding)
        return embedding

    def get_datapoint(self, idx):
        if idx not in self.datapoints:
            if self.suffix == ".h5":
                self.datapoints[idx] = self.read_features(self.data_paths[idx])
            elif self.suffix == ".png":
                self.datapoints[idx] = self.read_image(self.data_paths[idx])
            else:
                raise ValueError(
                    f"Unsupported file type: {self.data_paths[idx].suffix}"
                )
        return self.datapoints[idx]

    def __len__(self):
        return len(self.data_paths)

    def _get_slot_for_sample(self, index: int) -> Tuple[str, Dict[str, int], Dict[str, bool]]:
        """샘플의 organ_key, qid→code dict, qid→valid bool dict 반환."""
        if self.slot_cfg is None:
            return "", {}, {}
        site_name = self.labels[index]["site"]
        organ_key = self.slot_cfg["site_to_organ_key"].get(site_name, "")
        if organ_key == "" or organ_key not in self.slot_cfg["trainable_qids"]:
            return "", {}, {}
        sample_id = self.ids[index]
        id2labels = self.slot_cfg["id2labels"].get(organ_key, {})
        per_q = id2labels.get(sample_id, {})
        qids = self.slot_cfg["trainable_qids"][organ_key]
        codes = {qid: per_q.get(qid, -1) for qid in qids}
        valid = {qid: (codes[qid] >= 0) for qid in qids}
        return organ_key, codes, valid

    def __getitem__(self, index):
        site = torch.tensor(self.site_map[self.labels[index]["site"]])
        extraction = torch.tensor(
            self.extraction_map[self.labels[index]["extraction_method"]]
        )
        datapoint = self.get_datapoint(index)
        datapoint = self.process_datapoint(datapoint)
        size = 0
        if self.suffix == ".h5":
            if not self.max_bag_size or len(datapoint.shape) == 1:
                size = datapoint.shape[0]
            else:
                datapoint, size = sample_bag(datapoint, self.max_bag_size)

        # slot 정보 (slot OFF면 빈 dict)
        organ_key, codes, valid = self._get_slot_for_sample(index)
        # codes/valid는 dict인 채로 반환 — collate에서 organ별로 묶는다
        return (
            datapoint,
            size,
            site,
            extraction,
            torch.tensor(self.text_labels[index]),
            self.ids[index],
            organ_key,    # str
            codes,        # dict[qid → int]   (slot OFF면 {})
            valid,        # dict[qid → bool]  (slot OFF면 {})
        )

    def process_datapoint(self, datapoint: torch.Tensor):
        if self.suffix == ".png":
            if self.augment is not None:
                datapoint = self.augment(datapoint)
            datapoint = datapoint.sub(self.data_mean).div(self.data_std)
        return datapoint


def make_slot_collate(slot_cfg: Optional[Dict]):
    """
    DataLoader collate_fn.
    기존 7개 텐서는 기본 collate처럼 stack하고, slot은 organ별 dict로 묶는다.

    출력:
      (datapoint, lens, site, extraction, text_labels, ids,
       sample_organ_keys, slot_labels, slot_mask)
        sample_organ_keys : list[str], len=B
        slot_labels       : dict {organ_key: LongTensor [B, K_organ]}
        slot_mask         : dict {organ_key: BoolTensor [B, K_organ]}
                            (해당 organ이 아닌 샘플 row는 mask 전체 False)
    """
    def collate(batch):
        datapoints = torch.stack([b[0] for b in batch], dim=0)
        sizes = torch.tensor([b[1] for b in batch])
        sites = torch.stack([b[2] for b in batch], dim=0)
        extractions = torch.stack([b[3] for b in batch], dim=0)
        texts = torch.stack([b[4] for b in batch], dim=0)
        ids = [b[5] for b in batch]
        sample_organ_keys = [b[6] for b in batch]

        slot_labels: Dict[str, torch.Tensor] = {}
        slot_mask: Dict[str, torch.Tensor] = {}

        if slot_cfg is not None:
            B = len(batch)
            trainable_qids = slot_cfg["trainable_qids"]
            for organ_key, qids in trainable_qids.items():
                K = len(qids)
                labels = torch.full((B, K), -1, dtype=torch.long)
                mask = torch.zeros((B, K), dtype=torch.bool)
                for i, b in enumerate(batch):
                    ok, codes_i, valid_i = b[6], b[7], b[8]
                    if ok != organ_key:
                        continue
                    for j, qid in enumerate(qids):
                        if valid_i.get(qid, False):
                            labels[i, j] = codes_i[qid]
                            mask[i, j] = True
                slot_labels[organ_key] = labels
                slot_mask[organ_key] = mask

        return (
            datapoints,
            sizes,
            sites,
            extractions,
            texts,
            ids,
            sample_organ_keys,
            slot_labels,
            slot_mask,
        )

    return collate


def get_dataloaders(
    config: AppCfg, augment_train: bool
) -> Tuple[DataLoader, DataLoader]:
    if config.training.bag_size is None:
        assert (
            config.training.batch_size == 1
        ), f"Batch size must be 1 if max_bag_size is None, got {config.training.batch_size}"
    with open(config.data.train_json_path, "r") as f:
        label_list = json.load(f)

    label_dict = {}

    site_count, extraction_count = get_counts(label_list)
    word_to_idx = get_vocabulary(label_list, config)
    if config.other.verbose:
        print_info(label_list)
        print(f"Starting with {len(label_list)} elements")
    num_skipped = 0
    reason_skipped = {}
    for elem in label_list:
        if config.data.data_suffix == ".png":
            id = elem["id"].replace(".tiff", "_thumbnail_0000.png")
        elif config.data.data_suffix == ".h5":
            id = elem["id"].replace(".tiff", config.data.data_suffix)
        else:
            raise ValueError(f"Unsupported data suffix: {config.data.data_suffix}")
        site, extraction_method, text = get_site_extraction_method_and_text(
            elem["report"]
        )
        good_datapoint, reason = is_good_datapoint(
            id,
            site,
            extraction_method,
            site_count,
            extraction_count,
            config,
        )
        if not good_datapoint:
            num_skipped += 1
            if reason not in reason_skipped:
                reason_skipped[reason] = 0
            reason_skipped[reason] += 1
            continue

        label_dict[id] = {
            "site": site,
            "extraction_method": extraction_method,
            "text": text,
        }

    if config.other.verbose:
        print(f"Skipped {num_skipped} datapoints")
        for reason, count in reason_skipped.items():
            print(f"\tReason: {reason}, Count: {count}")
        print(f"Running with a total of {len(label_dict)} elements")
    if len(label_dict) == 0:
        raise ValueError(
            "No valid datapoints found. Check your label file and data path."
        )

    datapoints = []
    labels = []
    for datapoint, label in label_dict.items():
        path = Path(config.data.train_data_path) / datapoint
        if not path.exists():
            path = Path(config.data.train_data_path) / Path(datapoint).stem / datapoint
        datapoints.append(path)
        labels.append(label)

    # site/extraction count filtering
    for site in list(site_count.keys()):
        if site_count[site] < config.data.site_count_threshold:
            del site_count[site]
    for extraction in list(extraction_count.keys()):
        if extraction_count[extraction] < config.data.extraction_count_threshold:
            del extraction_count[extraction]
    site_map = {label: i for i, label in enumerate(set(site_count.keys()))}
    extraction_map = {label: i for i, label in enumerate(set(extraction_count.keys()))}

    # ─── slot supervision setup ─────────────────────────────────────────────
    slot_cfg = None
    if getattr(config.data, "use_slot_supervision", False):
        if not config.data.slot_cot_dir:
            raise ValueError("use_slot_supervision=True but slot_cot_dir is empty")
        organ_groups = get_organ_groups(config.data.organ_grouping)
        site_to_organ_key = build_site_to_organ_key(organ_groups)

        if config.other.verbose:
            print(f"\n[slot] use_slot_supervision=True")
            print(f"[slot] organ_grouping = {config.data.organ_grouping}")
            print(f"[slot] slot_cot_dir   = {config.data.slot_cot_dir}")
            print(f"[slot] skip first {config.data.slot_skip_first_n}, last {config.data.slot_skip_last_n}")
            print(f"[slot] loss_weight    = {config.data.slot_loss_weight}")
            print(f"[slot] loading per-organ CSVs ...")

        trainable_qids, num_classes, id2labels = load_slot_data(
            slot_cot_dir=config.data.slot_cot_dir,
            organ_groups=organ_groups,
            skip_first_n=config.data.slot_skip_first_n,
            skip_last_n=config.data.slot_skip_last_n,
            verbose=config.other.verbose,
        )

        # site filter 통과 못한 site에 속하는 organ_key는 살아남아도 그 organ에 해당하는 샘플이 없을 뿐
        # → 실제 학습에는 영향 없음. 명시적으로 제거하지 않아도 됨.

        slot_cfg = {
            "site_to_organ_key": site_to_organ_key,
            "trainable_qids":    trainable_qids,
            "num_classes":       num_classes,
            "id2labels":         id2labels,
        }

        if config.other.verbose:
            total_heads = sum(len(v) for v in trainable_qids.values())
            print(f"[slot] Total organs with slot heads: {len(trainable_qids)}")
            print(f"[slot] Total slot heads across organs: {total_heads}\n")

    train_partition_size = int(len(label_dict) * config.training.train_data_part)

    lst = list(zip(datapoints, labels))
    random.seed(config.training.seed)
    random.shuffle(lst)
    datapoints, labels = zip(*lst)
    datapoints, labels = list(datapoints), list(labels)

    max_num_datapoints = config.data.max_num_datapoints
    if config.training.train_data_part == 1:
        train_datapoints = datapoints[:max_num_datapoints]
        train_labels = labels[:max_num_datapoints]
        test_datapoints = datapoints[:100][:max_num_datapoints]
        test_labels = labels[:100][:max_num_datapoints]
    else:
        train_datapoints = datapoints[:train_partition_size][:max_num_datapoints]
        train_labels = labels[:train_partition_size][:max_num_datapoints]

        test_datapoints = datapoints[train_partition_size:][:max_num_datapoints]
        test_labels = labels[train_partition_size:][:max_num_datapoints]

    train_dataset = RegDataset(
        train_datapoints,
        train_labels,
        site_map,
        extraction_map,
        augment=augment_train,
        vocabulary=word_to_idx,
        max_bag_size=config.training.bag_size,
        disable_TQDM=config.other.disable_TQDM,
        slot_cfg=slot_cfg,
    )

    collate_fn = make_slot_collate(slot_cfg)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.dataloader.num_workers,
        persistent_workers=True,
        shuffle=True,
        drop_last=config.dataloader.drop_last,
        pin_memory=config.dataloader.pin_memory,
        pin_memory_device=(
            str(config.model.train_device) if config.dataloader.pin_memory else ""
        ),
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    test_dataset = RegDataset(
        test_datapoints,
        test_labels,
        site_map,
        extraction_map,
        augment=False,
        vocabulary=word_to_idx,
        max_bag_size=None,
        disable_TQDM=config.other.disable_TQDM,
        slot_cfg=slot_cfg,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config.dataloader.num_workers,
        persistent_workers=True,
        pin_memory=config.dataloader.pin_memory,
        pin_memory_device=(
            str(config.model.train_device) if config.dataloader.pin_memory else ""
        ),
        prefetch_factor=2,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # slot_cfg를 dataset/dataloader 어딘가에 접근할 수 있게 attribute로 추가
    train_dataloader.slot_cfg = slot_cfg
    test_dataloader.slot_cfg = slot_cfg

    return train_dataloader, test_dataloader
