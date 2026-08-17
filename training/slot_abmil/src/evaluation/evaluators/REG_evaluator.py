from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List, Tuple, Dict
import json
from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Possible set union.*",
    category=FutureWarning,
)


# class EmbeddingEvaluator:
#     def __init__(self, model_name, device):
#         assert model_name in [
#             "dmis-lab/biobert-v1.1",
#             "aaditya/Llama3-OpenBioLLM-8B",
#             "openai-community/gpt2",
#             "meta-llama/Meta-Llama-3.1-8B-Instruct",
#             "NeuML/pubmedbert-base-embeddings",
#         ], "Not Possible model_name."
#         assert model_name == "aaditya/Llama3-OpenBioLLM-8B"
#         self.device = device
#         self.model_name = model_name
#         self._get_embedding_model()

#     def _get_embedding_model(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModel.from_pretrained(self.model_name)
#         self.model.to(self.device)

#     # @torch.no_grad()
#     # def get_embedding(self, text):
#     #     inputs = self.tokenizer(
#     #         text, return_tensors="pt", padding=False, truncation=True, max_length=512
#     #     )
#     #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
#     #     inputs["input_ids"] = inputs["input_ids"].long()
#     #     outputs = self.model(**inputs)
#     #     return outputs.last_hidden_state.mean(dim=1).cpu().squeeze().numpy()

#     #수정
#     @torch.no_grad()
#     def get_embedding(self, text):
#         inputs = self.tokenizer(
#             text, return_tensors="pt", padding=False, truncation=True, max_length=512
#         )
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         inputs["input_ids"] = inputs["input_ids"].long()
#         outputs = self.model(**inputs)

#         emb = outputs.last_hidden_state.mean(dim=1)
#         emb = emb.float()          # bfloat16 → float32로 변환
#         return emb.cpu().squeeze().numpy()
    
# ###
#     def get_score(self, ref_text, hyp_text, scale=0.5):
#         ref_embedding = self.get_embedding(ref_text)
#         hyp_embedding = self.get_embedding(hyp_text)
#         score = cosine_similarity([ref_embedding], [hyp_embedding])[0][0]
#         if (scale != 0) and (score > scale):
#             score = (score - scale) / (1 - scale)
#         return score
#수정본

class EmbeddingEvaluator:
    def __init__(self, model_name, device):
        assert model_name in [
            "dmis-lab/biobert-v1.1",
            "aaditya/Llama3-OpenBioLLM-8B",
            "openai-community/gpt2",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "NeuML/pubmedbert-base-embeddings",
        ], "Not Possible model_name."
        assert model_name == "aaditya/Llama3-OpenBioLLM-8B"
        self.device = device
        self.model_name = model_name
        self._get_embedding_model()

    def _get_embedding_model(self):
        # 이미 캐시된 경로 또는 앞으로 사용할 캐시 경로
        cache_dir = "/root/.cache/huggingface"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            use_safetensors=False,   # safetensors 자동 변환 끄기
        )
        self.model.to(self.device)

    @torch.no_grad()
    def get_embedding(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["input_ids"] = inputs["input_ids"].long()
        outputs = self.model(**inputs)

        emb = outputs.last_hidden_state.mean(dim=1)
        emb = emb.float()  # bfloat16 → float32로 변환
        return emb.cpu().squeeze().numpy()

    def get_score(self, ref_text, hyp_text, scale=0.5):
        ref_embedding = self.get_embedding(ref_text)
        hyp_embedding = self.get_embedding(hyp_text)
        score = cosine_similarity([ref_embedding], [hyp_embedding])[0][0]
        if (scale != 0) and (score > scale):
            score = (score - scale) / (1 - scale)
        return score

class KeywordEvaluator:
    def __init__(self, model_name="en_core_sci_lg"):
        import spacy
        import scispacy
        from spacy.tokens import Doc

        self.nlp = spacy.load(model_name)

    @torch.no_grad()
    def get_keywords(self, text: str, min_length: int = 3) -> List[str]:
        doc = self.nlp(text)
        keywords = []
        for ent in doc.ents:
            if len(ent.text) >= min_length:
                keywords.append(ent.text.lower())
        return list(set(keywords))

    @staticmethod
    def get_jaccard(list1: List[str], list2: List[str]) -> float:
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    def get_score(self, ref_text: str, hyp_text: str, min_length: int = 3) -> float:
        ref_keywords = self.get_keywords(ref_text, min_length)
        hyp_keywords = self.get_keywords(hyp_text, min_length)

        score = self.get_jaccard(ref_keywords, hyp_keywords)

        return score


class REG_Evaluator:
    def __init__(self, embedding_model, device, spacy_model="en_core_sci_lg"):
        self.embedding_eval = EmbeddingEvaluator(embedding_model, device)

        self.key_eval = KeywordEvaluator(spacy_model)

    @staticmethod
    def get_bleu4(ref_text, hyp_text):
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()

        ref_fourgrams = [
            " ".join(ref_words[i : i + 4]) for i in range(len(ref_words) - 3)
        ]
        hyp_fourgrams = [
            " ".join(hyp_words[i : i + 4]) for i in range(len(hyp_words) - 3)
        ]

        count = 0
        total = 0

        for fourgram in hyp_fourgrams:
            count += min(hyp_fourgrams.count(fourgram), ref_fourgrams.count(fourgram))
            total += 1

        if total == 0:
            return 0.0

        precision = count / total
        return precision

    @staticmethod
    def get_rouge(ref_text, hyp_text):

        def lcs(X, Y):
            m = len(X)
            n = len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])
            return L[m][n]

        ref_tokens = ref_text.lower().split()
        hyp_tokens = hyp_text.lower().split()

        lcs_length = lcs(ref_tokens, hyp_tokens)

        ref_length = len(ref_tokens)
        hyp_length = len(hyp_tokens)

        precision = lcs_length / hyp_length if hyp_length > 0 else 0
        recall = lcs_length / ref_length if ref_length > 0 else 0

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return f1_score

    def evaluate_text(self, ref_text, hyp_text) -> Tuple[float, Dict[str, float]]:
        emb_score = self.embedding_eval.get_score(ref_text, hyp_text)
        key_score = self.key_eval.get_score(ref_text, hyp_text)
        bleu_score = self.get_bleu4(ref_text, hyp_text)
        rouge_score = self.get_rouge(ref_text, hyp_text)

        ranking_score = (
            0.15 * (rouge_score + bleu_score) + 0.4 * key_score + 0.3 * emb_score
        )
        info = {
            "rouge": rouge_score,
            "bleu": bleu_score,
            "key": key_score,
            "emb": emb_score,
            "ranking": ranking_score,
        }
        return ranking_score, info

    def evaluate_dummy(self, eval_lists):
        """list of tuples(pairs)"""
        score = 0
        pbar = tqdm(eval_lists, total=len(eval_lists))
        for i, (ref_text, hyp_text) in enumerate(pbar):
            score += self.evaluate_text(ref_text, hyp_text)[0]
        score /= len(eval_lists)
        return score

    def evaluate(
        self,
        ground_truth: str | List,
        predictions: str | List,
        disable_TQDM: bool = True,
    ) -> Tuple[float, Dict[str, float]]:

        if isinstance(ground_truth, str) or isinstance(ground_truth, Path):
            with open(ground_truth, "r") as f:
                gt_data = json.load(f)

        elif isinstance(ground_truth, list):
            gt_data = ground_truth

        if isinstance(predictions, str) or isinstance(predictions, Path):
            with open(predictions, "r") as f:
                pred_data = json.load(f)

        elif isinstance(predictions, list):
            pred_data = predictions

        gt_dict = {item["id"]: item["report"] for item in gt_data}
        eval_pairs = []
        for item in pred_data:
            pred_id = item["id"]
            pred_report = item["report"]

            gt_report = gt_dict.get(pred_id, "")
            if not gt_report:
                tqdm.write(f"Warning: No ground truth for ID {pred_id}. Skipping...")
                continue
            eval_pairs.append((gt_report, pred_report))
        if not eval_pairs:
            raise ValueError(
                "No valid evaluation pairs found. Check your ground truth and predictions."
            )
        score = 0
        part_scores = {"rouge": [], "bleu": [], "key": [], "emb": []}
        for ref_text, hyp_text in tqdm(
            eval_pairs,
            desc="Evaluating",
            total=len(eval_pairs),
            leave=False,
            position=1,
            disable=disable_TQDM,
        ):
            ranking_score, info = self.evaluate_text(ref_text, hyp_text)
            score += ranking_score
            for key in part_scores:
                part_scores[key].append(info[key])

        score /= len(eval_pairs)
        for key in part_scores:
            part_scores[key] = np.mean(part_scores[key])

        # tqdm.write(f" Overall Score: {score:.4f}")
        # tqdm.write("Part Scores:")
        # for key, value in part_scores.items():
        #    tqdm.write(f"\t{key}: {value*100:.2f}%")

        return score, part_scores


if __name__ == "__main__":
    evaluator = REG_Evaluator(
        embedding_model="aaditya/Llama3-OpenBioLLM-8B",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth_json",
        type=str,
        required=True,
        help="Path to ground truth JSON file",
    )
    parser.add_argument(
        "--predictions_json",
        type=str,
        required=True,
        help="Path to predictions JSON file",
    )
    args = parser.parse_args()

    ground_truth_json = args.ground_truth_json
    predictions_json = args.predictions_json
    score, part_scores = evaluator.evaluate(
        ground_truth_json, predictions_json, disable_TQDM=False
    )
    print(f"Final Score: {score:.4f}")
    print("Part Scores:", part_scores)
