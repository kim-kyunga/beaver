"""
site_classifier + extraction_classifier + (optional) slot_classifiers + decoder

slot supervision이 켜져 있으면 organ별 / question별 분류기를 추가한다.
구조:
    self.slot_classifiers = nn.ModuleDict({
        organ_key: nn.ModuleDict({
            qid: nn.Linear(input_features, n_classes)
        })
    })

forward 출력에 "slot_logits" 가 추가됨:
    slot_logits = {organ_key: {qid: tensor[B, n_classes]}}

slot_num_classes 가 None이면 기존 baseline 동작과 완전히 동일.
"""

import torch
import json
from torch import nn
from .blocks import TransformerDecoder
from .ABMIL import MILModel
from src.conf.schema import AppCfg
from src.dataloaders.data_utils import (
    get_vocabulary,
    get_site_extraction_method_and_text,
)
from src.dataloaders.tokenization import tokenizer
from typing import List, Dict, Optional


class Network(nn.Module):
    def __init__(
        self,
        num_classes_site,
        num_classes_extraction,
        adaptive_pooling_size,
        data_suffix,
        input_feature_shape,
        vocab_size,
        max_seq_len,
        corpus,
        config,
        slot_num_classes: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        super(Network, self).__init__()
        num_channels = config.model.num_channels
        if data_suffix == ".h5":
            # This is if we extracted slide level embeddings, and we don't need ABMIL
            if len(input_feature_shape) == 1:
                self.embedder = nn.Sequential(
                    nn.Linear(
                        input_feature_shape[0], config.model.slide_feature_encoder_dim
                    ),
                    nn.LeakyReLU(),
                    nn.Linear(
                        config.model.slide_feature_encoder_dim,
                        config.model.slide_feature_encoder_dim,
                    ),
                    nn.LeakyReLU(),
                    nn.Linear(
                        config.model.slide_feature_encoder_dim,
                        config.model.num_channels * (adaptive_pooling_size**2),
                    ),
                    nn.LeakyReLU(),
                )
            # If the length is 3, we need ABMIL
            elif len(input_feature_shape) == 2:
                self.embedder = MILModel(
                    input_feature_shape[-1],
                    config.model.dropout_abmil,
                    config.model.num_channels,
                    config.model.num_layers_abmil,
                    config.model.num_attention_heads_abmil,
                    True,
                )
                num_channels = (
                    config.model.num_channels * config.model.num_attention_heads_abmil
                )
        else:
            raise ValueError(f"Unsupported data suffix: {data_suffix}")
        input_features = num_channels * (adaptive_pooling_size**2)
        self.site_classifier = nn.Linear(input_features, num_classes_site)
        self.extraction_classifier = nn.Linear(input_features, num_classes_extraction)

        # ─── slot heads (optional) ────────────────────────────────────────────
        # slot_num_classes: {organ_key: {qid: n_classes}}
        self.slot_classifiers = nn.ModuleDict()
        # qid에 자주 들어가는 '.' 같은 문자는 ModuleDict 키로 못 쓰므로 안전 매핑.
        # forward에서 다시 원래 qid로 되돌려주기 위해 self.slot_qid_map을 둔다.
        self.slot_qid_map: Dict[str, Dict[str, str]] = {}  # organ_key -> {qid: safe_qid}
        if slot_num_classes is not None and len(slot_num_classes) > 0:
            for organ_key, qid_to_n in slot_num_classes.items():
                if len(qid_to_n) == 0:
                    continue
                safe_organ = self._sanitize_key(organ_key)
                inner = nn.ModuleDict()
                self.slot_qid_map[organ_key] = {}
                for qid, n_classes in qid_to_n.items():
                    safe_qid = self._sanitize_key(qid)
                    inner[safe_qid] = nn.Linear(input_features, n_classes)
                    self.slot_qid_map[organ_key][qid] = safe_qid
                self.slot_classifiers[safe_organ] = inner
            # organ_key 도 sanitize 했으니 forward에서 되돌릴 매핑 저장
            self.slot_organ_safe_map: Dict[str, str] = {
                organ_key: self._sanitize_key(organ_key)
                for organ_key in slot_num_classes.keys()
                if len(slot_num_classes[organ_key]) > 0
            }
        else:
            self.slot_organ_safe_map = {}

        self.decoder = TransformerDecoder(
            context_dim=input_features,
            vocab_size=vocab_size,
            corpus=corpus,
            config=config,
            max_seq_len=max_seq_len,
        )

    @staticmethod
    def _sanitize_key(k: str) -> str:
        """nn.ModuleDict 키 규칙(영문/숫자/언더스코어)에 맞도록 변환."""
        return (
            k.replace(".", "_")
             .replace("-", "_")
             .replace(" ", "_")
             .replace("/", "_")
             .replace(":", "_")
        )

    def forward(self, x, lens, target_seq=None, return_embedding=False):
        B = x.shape[0]
        if isinstance(self.embedder, MILModel):
            x = self.embedder(x, lens=lens).squeeze()
        else:
            x = self.embedder(x).squeeze()
        if B == 1:
            x = x.unsqueeze(0)
        x = x.view(x.size(0), -1)

        out = {
            "site_logits": self.site_classifier(x),
            "em_logits": self.extraction_classifier(x),
            "text_logits": self.decoder(x, target_seq),
            "embeddings": x,
        }

        # slot logits: organ_key별 dict
        if len(self.slot_classifiers) > 0:
            slot_logits: Dict[str, Dict[str, torch.Tensor]] = {}
            for organ_key, safe_organ in self.slot_organ_safe_map.items():
                if safe_organ not in self.slot_classifiers:
                    continue
                inner = self.slot_classifiers[safe_organ]
                per_q: Dict[str, torch.Tensor] = {}
                qid_map = self.slot_qid_map.get(organ_key, {})
                for qid, safe_qid in qid_map.items():
                    per_q[qid] = inner[safe_qid](x)
                slot_logits[organ_key] = per_q
            out["slot_logits"] = slot_logits

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# This should preferably not be placed here, but it is a quick fix
def pad_sequence(vocabulary, seq: List[int], max_len: int):
    if len(seq) < max_len:
        seq = seq + [vocabulary["<PAD>"]] * (max_len - len(seq))
    assert len(seq) == max_len, f"Sequence length mismatch: {len(seq)} != {max_len}"
    return seq


def get_corpus(
    report_list: List[str], vocabulary: Dict[str, int], max_seq_len: int
) -> List[List[int]]:
    text_labels = []
    for label in report_list:
        int_labels = [vocabulary["<SOS>"]]
        text = get_site_extraction_method_and_text(label)[2]
        words = tokenizer(text)
        for word in words:
            int_labels.append(vocabulary[word])
        int_labels.append(vocabulary["<EOS>"])
        padded_int_labels = pad_sequence(vocabulary, int_labels, max_seq_len)
        text_labels.append(padded_int_labels)
    return text_labels


def get_max_seq_len(report_list: List[Dict[str, str]]) -> int:
    text = [
        get_site_extraction_method_and_text(label["report"])[2] for label in report_list
    ]
    max_seq_len = max(len(tokenizer(t)) for t in text) + 2
    return max_seq_len


def get_model(
    config: AppCfg,
    idx_to_word: dict[int, str],
    idx_to_site: dict[int, str],
    idx_to_extraction: dict[int, str],
    input_feature_shape: tuple[int, ...],
    slot_num_classes: Optional[Dict[str, Dict[str, int]]] = None,
) -> Network:
    # TODO: This shouldn't take in the dataloader at all
    with open(config.data.train_json_path, "r") as f:
        train_data = json.load(f)
    train_text = [elem["report"] for elem in train_data]
    word_to_idx = get_vocabulary(train_data, config)
    max_seq_len = get_max_seq_len(train_data)

    if config.model.model_checkpoint is not None:
        state_dict = torch.load(config.model.model_checkpoint, map_location="cpu")
        try:
            max_seq_len = state_dict["decoder.pos_embedding.weight"].shape[0]
        except KeyError as e:
            print(
                f"Could not load max_seq_len from checkpoint, using {max_seq_len} as a fallback. Error: {e}"
            )
    corpus = get_corpus(train_text, word_to_idx, max_seq_len)

    network = Network(
        num_classes_site=len(idx_to_site),
        num_classes_extraction=len(idx_to_extraction),
        adaptive_pooling_size=1,
        data_suffix=config.data.data_suffix,
        input_feature_shape=input_feature_shape,
        vocab_size=len(idx_to_word),
        max_seq_len=max_seq_len,
        corpus=corpus,
        config=config,
        slot_num_classes=slot_num_classes,
    )
    print(f"Number of parameters: {network.count_parameters() / 1e6:.2f}M")
    if slot_num_classes is not None and len(slot_num_classes) > 0:
        total_heads = sum(len(v) for v in slot_num_classes.values())
        total_classes = sum(
            sum(qid_to_n.values()) for qid_to_n in slot_num_classes.values()
        )
        print(
            f"[slot] organs={len(slot_num_classes)}, "
            f"slot_heads={total_heads}, total_slot_classes={total_classes}"
        )
    if config.model.model_checkpoint is not None:
        network.load_state_dict(
            torch.load(
                config.model.model_checkpoint, map_location=config.model.train_device
            ),
            strict=False,  # slot head는 checkpoint에 없을 수 있으므로 관대하게
        )
    network.to(config.model.train_device)
    return network
