from typing import List, Dict, Union, Tuple
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.conf.schema import AppCfg
from pathlib import Path
import string
import re
import difflib
from src.dataloaders.tokenization import tokenizer, detokenizer

#추가
def normalize_report(s: str) -> str:
    # 1) 윈도우/유닉스 줄바꿈 통일
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 2) 연속된 공백을 하나로
    s = re.sub(r"[ \t]+", " ", s)
    # 3) 줄 앞뒤 공백 제거
    s = "\n".join(line.strip() for line in s.split("\n"))
    # 4) 전체 앞뒤 공백 제거
    return s.strip()
    
####

def assert_reports_equal(report1: str, report2: str, raise_error=True, verbose: bool = False):
    if report1 != report2:
        diff = difflib.ndiff([report2], [report1])
        # Color the diff: green = added, red = removed, white = unchanged
        colored_lines = []
        for line in diff:
            if line.startswith("-"):
                colored_lines.append(f"{repr(line)}")  # red
            elif line.startswith("+"):
                colored_lines.append(f"{repr(line)}")  # green
            elif line.startswith("?"):
                colored_lines.append(
                    f"\033[93m{line}\033[0m"
                )  # yellow (position markers)
            else:
                colored_lines.append(line)
        if verbose:
            print("Report mismatch:")
            print("\n".join(colored_lines))
            print("\n\n\n")
        if raise_error:
            raise ValueError(f"Report mismatch: {repr(report1)} != {repr(report2)}")
        return True
    return False


def assert_same_report(report: str, verbose: bool = False):
    report_words = tokenizer(report)
    reconstructed_report = detokenizer(report_words)

    # assert_reports_equal(report, reconstructed_report, raise_error=False, verbose=verbose)
    # 여기에서 정규화 후 비교 (추가)
    norm1 = normalize_report(report)
    norm2 = normalize_report(reconstructed_report)

    assert_reports_equal(norm1, norm2, raise_error=False, verbose=verbose)
###
# def tokenizer(text: str) -> List[str]:
#     tokens = []
#     str_builder = []
#     for ch in text:
#         if ch in SPECIAL_SYMBOLS:
#             built_str = "".join(str_builder)
#             if built_str != "":
#                 tokens.append(built_str)
#             str_builder = []
#             tokens.append(ch)
#         else:
#             str_builder.append(ch)

#     if str_builder:
#         built_str = "".join(str_builder)
#         if built_str != "":
#             tokens.append(built_str)
#     return tokens


# def detokenizer(tokens: List[str]) -> str:

#     start = ""
#     s = "".join(tokens)
#     s = f"{start}{s.replace('<SOS>', '')}"
#     return s


def get_vocabulary(label_list: List[Dict[str, str]], config):
    word_counts = {}
    for elem in label_list:
        report = elem["report"]
        if ";" not in report:
            print(f"Skipping report without semicolon: {report}")
            continue
        report = report.split(";", 1)[1]
        # Split the report into words and add them to the set
        report_words = tokenizer(report)
        # print(report_words)
        assert_same_report(report, config.other.verbose)
        for word in report_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1
    words_to_idx = {
        "<PAD>": config.text.pad_idx,
        "<SOS>": config.text.sos_idx,
        "<EOS>": config.text.eos_idx,
    }  # Special tokens
    # NOTE: If any of these three fail, you need to edit the "i + 3" line below
    assert config.text.pad_idx == 0, "PAD index must be 0"
    assert config.text.sos_idx == 1, "SOS index must be 1"
    assert config.text.eos_idx == 2, "EOS index must be 2"
    for i, word in enumerate(
        sorted(list(word_counts.keys()), key=lambda x: word_counts[x], reverse=True)
    ):
        words_to_idx[word] = (
            i + 3
        )  # Start indexing from 3 to leave space for special tokens

    if config.other.verbose:
        for word, count in sorted(
            word_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"Word: {repr(word)}, Count: {count}")
        print(f"Vocabulary size: {len(words_to_idx)}")
    return words_to_idx


def get_counts(label_list: List[Dict[str, str]]):
    site_count = {}
    extraction_count = {}
    for elem in label_list:
        report = elem["report"]
        site, extraction_method, _ = get_site_extraction_method_and_text(report)

        if site not in site_count.keys():
            site_count[site] = 0
        if extraction_method not in extraction_count.keys():
            extraction_count[extraction_method] = 0

        site_count[site] += 1
        extraction_count[extraction_method] += 1

    return site_count, extraction_count


def get_word_counts(label_list: List[Dict[str, str]]):
    word_counts = {}
    for elem in label_list:
        report = elem["report"]
        if ";" not in report:
            print(f"Skipping report without semicolon: {report}")
            continue
        report = report.split(";", 1)[1]
        # Split the report into words and add them to the set
        report_words = tokenizer(report)
        for word in report_words:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

    return word_counts


def get_site_extraction_method_and_text(report: str):
    """
    Extracts the site and extraction method from the report string.
    The report is expected to be in the format "site,extraction_method;other_info".
    """
    site = report.split(",")[0]
    try:
        extraction_method = report.split(",")[1].split(";", 1)[0]
    except IndexError:
        extraction_method = "unknown"

    try:
        text_after_semicolon = report.split(";", 1)[1]
    except IndexError:
        text_after_semicolon = ""

    extraction_method = extraction_method.replace(" olonoscopic", " colonoscopic")
    return (
        site,
        extraction_method.strip(),
        text_after_semicolon,
    )


def print_info(label_list: List[Dict[str, str]]):
    site_to_EM_count = {}
    site_count = {}
    extraction_count = {}

    for elem in label_list:
        report = elem["report"]
        site, extraction_method, text = get_site_extraction_method_and_text(report)

        site_to_EM_count.setdefault(site, {}).setdefault(extraction_method, 0)
        site_to_EM_count[site][extraction_method] += 1

        site_count[site] = site_count.get(site, 0) + 1
        extraction_count[extraction_method] = (
            extraction_count.get(extraction_method, 0) + 1
        )

    print("=== Site and Extraction Method Counts ===")
    for site, EMs in site_to_EM_count.items():
        print(f"Site: {site}")
        for EM, count in EMs.items():
            print(f"  - Extraction Method: {EM}, Count: {count}")
        print()

    # Create the confusion matrix
    site_list = sorted(site_count.keys())
    extraction_list = sorted(extraction_count.keys())
    confusion_matrix = np.zeros((len(site_list), len(extraction_list)))

    for i, site in enumerate(site_list):
        for j, EM in enumerate(extraction_list):
            confusion_matrix[i, j] = site_to_EM_count.get(site, {}).get(EM, 0)

    # Plot with seaborn
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        # xticks=range(len(extraction_list)),
        yticklabels=site_list,
        xticklabels=[e.replace("biopsy", "B") for e in extraction_list],
    )

    ax.set_title("Confusion Matrix of Sites vs Extraction Methods")
    ax.set_xlabel("Extraction Methods")
    ax.set_ylabel("Sites")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("imgs/confusion_matrix.png")
    plt.close()

    print("\n=== Sorted Site Counts ===")
    for site, count in sorted(site_count.items(), key=lambda x: x[1], reverse=True):
        print(f"Site: {site}, Count: {count}")

    print("\n=== Sorted Extraction Method Counts ===")
    for EM, count in sorted(extraction_count.items(), key=lambda x: x[1], reverse=True):
        print(f"Extraction Method: {EM}, Count: {count}")

    print()


def is_good_datapoint(
    datapoint_name: str,
    site: str,
    extraction_method: str,
    site_count: Dict[str, int],
    extraction_count: Dict[str, int],
    config: AppCfg,
) -> Tuple[bool, Union[str, None]]:
    if (
        not (Path(config.data.train_data_path) / datapoint_name).exists()
        and not (
            Path(config.data.train_data_path)
            / Path(datapoint_name).stem
            / datapoint_name
        ).exists()
    ):
        return False, "Datapoint does not exist"

    if site_count[site] < config.data.site_count_threshold:
        return False, "Site count too low"
    if extraction_count[extraction_method] < config.data.extraction_count_threshold:
        return False, "Extraction method count too low"

    if not datapoint_name.endswith(".png") and not datapoint_name.endswith(".h5"):
        return False, "Datapoint name does not end with .png or .h5"

    return True, None


def zero_pad_tensor(tensor, bag_size, device):
    sizes = []
    sizes.append(bag_size - tensor.shape[0])
    if len(tensor.shape) > 1:
        sizes.append(tensor.shape[1])
    return torch.cat(
        (
            tensor,
            torch.zeros(sizes, device=device),
        )
    )


def sample_bag(features: torch.Tensor, bag_size: int) -> Tuple[torch.Tensor, int]:
    device = torch.device("cpu")

    num_features = features.shape[0]
    bag_idxs = bag_idxs = torch.randperm(num_features, device=device)[:bag_size]
    features = features[bag_idxs]
    zero_padded_bag = zero_pad_tensor(features, bag_size, device)
    return zero_padded_bag, min(bag_size, len(bag_idxs))


def get_sorted_word_map(dictionary):
    # 1) idx → word
    new_dict = {}
    words = sorted(list(dictionary.keys()))
    for i, word in enumerate(words):
        new_dict[word] = i

    return new_dict


