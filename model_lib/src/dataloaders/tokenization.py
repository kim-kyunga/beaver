from typing import List, Dict, Union, Tuple
import string


def tokenizer(text: str):
    return text.strip().split()


def detokenizer(tokens: List[str]):
    return f"\n {' '.join(tokens)}".replace("<SOS>", "")
