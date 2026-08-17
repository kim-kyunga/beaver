import json
import time

from tqdm import tqdm
import torch
from torch import nn

from src.dataloaders.data_utils import get_counts, get_word_counts


def update_running_accuracy(batch_out, site_lbl, em_lbl, txt_lbl, pad_idx, running):
    """Update running dict in-place with correct token / class counts."""
    site_pred = batch_out.site_logits.argmax(-1)
    em_pred = batch_out.em_logits.argmax(-1)
    txt_pred = batch_out.text_logits.argmax(-1)

    running["correct_site"] += (site_pred == site_lbl).sum().item()
    running["correct_em"] += (em_pred == em_lbl).sum().item()

    # --- token-level text accuracy ---
    mask = txt_lbl != pad_idx
    running["correct_text"] += (txt_pred[mask] == txt_lbl[mask]).sum().item()
    running["total_tokens"] += mask.sum().item()


def get_loss_functions(
    json_path,
    idx_to_site,
    idx_to_extraction,
    idx_to_word,
    device,
    pad_idx,
    beta=0.99,  # beta is optional; set None to use 1/freq
) -> tuple[nn.CrossEntropyLoss, nn.CrossEntropyLoss, nn.CrossEntropyLoss]:

    with open(json_path) as f:
        label_list = json.load(f)

    site_counts, em_counts = get_counts(label_list)
    word_counts = get_word_counts(label_list)

    def make_weights(counts_dict, idx_to_label):
        counts_dict["<PAD>"] = torch.inf  # ensure pad token is not counted
        counts_dict["<SOS>"] = torch.inf  # ensure SOS token is not counted
        counts_dict["<EOS>"] = torch.inf  # ensure EOS token is not counted
        if beta is None:  # simple inverse frequency
            weights = [
                1.0 / counts_dict[idx_to_label[i]] for i in range(len(idx_to_label))
            ]
        else:  # “effective number” weighting
            weights = [
                (1.0 - beta) / (1.0 - beta ** counts_dict[idx_to_label[i]])
                for i in range(len(idx_to_label))
            ]
        # normalise so that mean weight = 1
        mean_w = sum(weights) / len(weights)
        return torch.tensor(
            [w / mean_w for w in weights], dtype=torch.float, device=device
        )

    site_loss_fn = nn.CrossEntropyLoss(weight=make_weights(site_counts, idx_to_site))
    extraction_loss_fn = nn.CrossEntropyLoss(
        weight=make_weights(em_counts, idx_to_extraction)
    )

    word_weights = make_weights(word_counts, idx_to_word)
    word_weights[pad_idx] = 0.0  # in case pad_idx != idx_of("<PAD>")
    text_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, weight=word_weights)

    return site_loss_fn, extraction_loss_fn, text_loss_fn

def print_reg_metrics(reg_metrics):
    if reg_metrics is not None:
        for evaluator, scores in reg_metrics.items():
            tqdm.write(f"{evaluator}")
            tqdm.write(f"\tOverall score: {scores['overall_score'] * 100:.2f}%")
            tqdm.write("\tPart scores:")
            for key, value in scores["part_score"].items():
                tqdm.write(f"\t\t{key}: {value * 100:.2f}%")

def print_metrics(
    train_output,
    fast_metrics,
    reg_metrics,
    start_time,
    epoch,
    config,
    training_time,
    embbeding_time,
    evaluation_time,
):
    time_spent = time.time() - start_time
    info_str = f"Epoch {epoch + 1:>3} / {config.training.epochs} "
    info_str += f" | Site loss: {train_output['site_loss']:.3f}"
    info_str += f" | EM loss {train_output['extraction_loss']:.3f}"
    info_str += f" | Slot loss: {train_output.get('slot_loss', 0.0):.3f}"
    info_str += f" | Text loss: {train_output['text_loss']:.3f}"
    info_str += f" | Acc site: {fast_metrics['site_acc']:.3f}"
    info_str += f" | Acc EM: {fast_metrics['em_acc']:.3f}"
    info_str += f" | Acc text: {fast_metrics['text_token_acc']:.3f}"
    info_str += f" | CPS: {train_output['embeddings'].shape[0] / time_spent:.2f}"
    info_str += f" | Training time: {training_time:.2f}s"
    info_str += f" | Evaluation time: {evaluation_time:.2f}s"
    if embbeding_time is not None:
        info_str += f" | Embedding time: {embbeding_time:.2f}s"
    tqdm.write(info_str)

    print_reg_metrics(reg_metrics)
