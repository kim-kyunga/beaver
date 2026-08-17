import json
from pathlib import Path
from functools import lru_cache
from typing import Dict, Tuple, List
from src.dataloaders.tokenization import detokenizer

import torch


def decode_text_pred(
    text_pred: torch.Tensor,
    idx_to_word: Dict[int, str],
    pad_idx: int,
) -> str:
    predicted_ids = text_pred.tolist()  # get last sample in batch
    words = []
    for token_id in predicted_ids:
        if token_id == 2:
            break
        if token_id == pad_idx:
            continue  # skip padding tokens
        words.append(idx_to_word.get(token_id, f"<UNK_{token_id}>"))

    # predicted_text = " ".join(words)
    predicted_text = detokenizer(words)

    return predicted_text


def decode_text_label(
    text_label: torch.Tensor,
    idx_to_word: Dict[int, str],
    pad_idx: int,
) -> str:
    ground_truth_ids = text_label.tolist()  # get last sample in batch
    ground_truth_words = [
        idx_to_word.get(token_id, f"<UNK_{token_id}>")
        for token_id in ground_truth_ids
        if token_id != pad_idx and token_id != 2  # skip padding and EOS tokens
    ]
    ground_truth_text = " ".join(ground_truth_words)

    return ground_truth_text


def decode_text(text_pred, text_label, idx_to_word, pad_idx):
    decoded_text_pred, decoded_text_label = None, None
    if text_pred is not None:
        decoded_text_pred = decode_text_pred(text_pred, idx_to_word, pad_idx)
    if text_label is not None:
        decoded_text_label = decode_text_label(text_label, idx_to_word, pad_idx)

    return decoded_text_pred, decoded_text_label


def get_num_correct_tokens(
    text_pred: torch.Tensor,
    text_label: torch.Tensor,
    config,
) -> int:
    """
    Mean per‑sequence token accuracy, vectorised.

    Matches the logic of the original Python loop when the two
    tensors have different max‑sequence lengths.
    """
    # ------------------------------------------------------------------
    # 0) Align the two tensors along the sequence axis
    # ------------------------------------------------------------------
    seq_len = min(text_pred.size(1), text_label.size(1))
    text_pred = text_pred[:, :seq_len]
    text_label = text_label[:, :seq_len]

    # ------------------------------------------------------------------
    # 1) Tokens strictly *before* the first <EOS> in each prediction
    # ------------------------------------------------------------------
    eos_flags = (text_pred == config.text.eos_idx).int()  # 1 at EOS positions
    after_eos = torch.cumsum(eos_flags, dim=1)  # 0,1,2,… after first EOS
    before_eos = after_eos == 0  # True if j < first‑EOS

    # ------------------------------------------------------------------
    # 2) Valid target positions (ignore <PAD> and <SOS>)
    # ------------------------------------------------------------------
    tgt_valid = (text_label != config.text.pad_idx) & (
        text_label != config.text.sos_idx
    )

    # ------------------------------------------------------------------
    # 3) Positions we actually compare
    # ------------------------------------------------------------------
    compare_mask = before_eos & tgt_valid

    # ------------------------------------------------------------------
    # 4) Correct predictions at those positions
    # ------------------------------------------------------------------
    correct_mask = (text_pred == text_label) & compare_mask

    # ------------------------------------------------------------------
    # 5) Per‑sequence accuracy, then batch mean
    # ------------------------------------------------------------------
    total_per_seq = compare_mask.sum(dim=1)  # [B]
    correct_per_seq = correct_mask.sum(dim=1).float()  # [B]

    acc_per_seq = torch.where(
        total_per_seq > 0,
        correct_per_seq / total_per_seq.float(),
        torch.zeros_like(correct_per_seq),
    )

    return int(acc_per_seq.sum().item())


@lru_cache(maxsize=10000)
def build_decoder_io(batch_seq: torch.Tensor):
    """
    Args
    ----
    batch_seq : LongTensor (B, T)
        Each row: <SOS> ... <EOS> ... <PAD>
    pad_idx   : int
        Index of the <PAD> token (ignored by the loss)

    Returns
    -------
    decoder_in     : LongTensor (B, T-1)
        Feed this to the decoder.
    decoder_target : LongTensor (B, T-1)
        Compare the decoder's logits against this with CrossEntropyLoss.
    """
    # Sanity check
    assert batch_seq.dim() == 2, "expected (B, T)"
    device = batch_seq.device

    # ① remove last token for decoder input
    decoder_in = batch_seq[:, :-1].contiguous()  # (B, T-1)

    # ② remove first token for decoder target
    decoder_target = batch_seq[:, 1:].contiguous()  # (B, T-1)

    # Nothing else needed: <SOS> is now only in decoder_in,
    # <EOS> is only in decoder_target.
    #
    # CrossEntropyLoss(ignore_index=pad_idx) will ignore <PAD> positions.

    return decoder_in.to(device), decoder_target.to(device)


def id_to_tiffid(id: str, data_suffix: str) -> str:
    if data_suffix == ".png":
        id_formatted = id.split("_thumbnail_0000.png")[0] + ".tiff"
    elif data_suffix == ".h5":
        id_formatted = id.replace(".h5", "") + ".tiff"
    else:
        raise ValueError(f"Unsupported data suffix: {data_suffix}")
    return id_formatted


def get_idx_mappings(train_dataloader):
    idx_to_word = {v: k for k, v in train_dataloader.dataset.vocabulary.items()}
    idx_to_site = {v: k for k, v in train_dataloader.dataset.site_map.items()}
    idx_to_extraction = {
        v: k for k, v in train_dataloader.dataset.extraction_map.items()
    }
    return idx_to_word, idx_to_site, idx_to_extraction


def save_idx_mappings(
    idx_to_word: Dict[int, str],
    idx_to_site: Dict[int, str],
    idx_to_extraction: Dict[int, str],
    output_path: Path,
):
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "idx_to_word.json", "w") as f:
        json.dump(idx_to_word, f)
    with open(output_path / "idx_to_site.json", "w") as f:
        json.dump(idx_to_site, f)
    with open(output_path / "idx_to_extraction.json", "w") as f:
        json.dump(idx_to_extraction, f)


def read_idx_mappings(
    input_path: Path,
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:

    with open(input_path / "idx_to_word.json", "r") as f:
        idx_to_word = json.load(f)
        idx_to_word = {int(k): v for k, v in idx_to_word.items()}
    with open(input_path / "idx_to_site.json", "r") as f:
        idx_to_site = json.load(f)
        idx_to_site = {int(k): v for k, v in idx_to_site.items()}
    with open(input_path / "idx_to_extraction.json", "r") as f:
        idx_to_extraction = json.load(f)
        idx_to_extraction = {int(k): v for k, v in idx_to_extraction.items()}
    return idx_to_word, idx_to_site, idx_to_extraction
