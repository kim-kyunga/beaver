import json
from pathlib import Path

import hydra
import h5py
import torch

from src.dataloaders.dataloaders import get_dataloaders
from src.utils.general_utils import process_config, get_input_feature_shape
from src.network.model import get_model
from src.evaluation.evaluators.decoder_evaluator import TextReportEvaluator
from src.evaluation.evaluators.embedding_evaluator import EmbeddingMatchEvaluator
from src.evaluation.evaluation_utils.embedding_extractor import EmbeddingExtractor
from src.evaluation.evaluation_utils.inference import InferenceRunner
from src.utils.text_utils import read_idx_mappings
from src.evaluation.evaluation_utils.epoch_eval import run_epoch_evaluation
from src.trainer.training_utils import print_reg_metrics
from src.dataloaders.dataloaders import RegDataset
from torch.utils.data import DataLoader
from src.conf.schema import AppCfg
from src.dataloaders.data_utils import get_vocabulary


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def evaluate(config: AppCfg) -> None:

    config = process_config(config, is_eval=True)
    assert (
        config.model.model_checkpoint is not None
    ), "Model checkpoint must be specified to extract embeddings"
    assert (
        config.eval.run_text or config.eval.run_embedding
    ), "Needs to evaluate at least one of them"

    if config.eval.run_embedding:
        train_dataloader, _ = get_dataloaders(config, augment_train=False)

    idx_to_word, idx_to_site, idx_to_extraction = read_idx_mappings(
        Path(config.data.experiment_path) / config.data.experiment_name
    )
    assert (
        config.eval.test_data_path is not None
    ), "Test data path must be specified for evaluation"
    assert (
        config.training.train_data_part >= 1
    ), "Train data part must be >= 1 for evaluation"
    test_datapoints = list(Path(config.eval.test_data_path).rglob("*.h5"))[
        : config.data.max_num_datapoints
    ]
    with open(config.data.train_json_path, "r") as f:
        train_data = json.load(f)
    word_to_idx = get_vocabulary(train_data, config)
    test_dataset = RegDataset(
        test_datapoints,
        [
            {
                # These are not used, and are thus just placeholders
                "site": "lung",
                "extraction_method": "biopsy",
                "text": "type type type ",
            }
            for _ in range(len(test_datapoints))
        ],
        {"lung": 5},
        {"biopsy": 5},
        augment=False,
        vocabulary=word_to_idx,
        max_bag_size=None,
        disable_TQDM=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=16,
        persistent_workers=True,
        pin_memory=False,
        prefetch_factor=2,
        drop_last=False,  # Oops
    )

    input_feature_shape = get_input_feature_shape(config)

    network = get_model(
        config,
        idx_to_word,
        idx_to_site,
        idx_to_extraction,
        input_feature_shape,
    ).to(config.model.train_device)

    infer = InferenceRunner(
        network,
        device=config.model.train_device,
        precision=(
            torch.float16 if config.training.precision == "fp16" else torch.float32
        ),
    )

    evaluator = None
    embedding_extractor = None
    if config.eval.run_embedding:
        embedding_extractor = EmbeddingExtractor(
            network,
            config,
        )

    if config.eval.run_text:
        text_eval = TextReportEvaluator(
            idx_to_word=idx_to_word,
            idx_to_site=idx_to_site,
            idx_to_EM=idx_to_extraction,
            pad_idx=config.text.pad_idx,
            out_path=Path(config.data.experiment_path)
            / config.data.experiment_name
            / "decoder_predictions_external.json",
            reg_evaluator=evaluator,
            ground_truth_json=Path(config.data.train_json_path),
            save_json=True if not config.other.dry_run else False,
            disable_TQDM=config.other.disable_TQDM,
        )

    if config.eval.run_embedding and embedding_extractor is not None:
        embedding_output = embedding_extractor.get_embeddings(train_dataloader)
        embed_eval_filter = EmbeddingMatchEvaluator(
            train_embeddings=embedding_output["embeddings"],
            train_ids=embedding_output["ids"],
            train_site_pred=embedding_output["site_preds"],
            train_em_pred=embedding_output["em_preds"],
            out_path=Path(config.data.experiment_path)
            / config.data.experiment_name
            / "embedding_predictions_external_matched.json",
            reg_evaluator=evaluator,
            ground_truth_json=Path(config.data.train_json_path),
            save_json=True if not config.other.dry_run else False,
            match_site_and_em=True,
            disable_TQDM=config.other.disable_TQDM,
        )
        embed_eval_no_filter = EmbeddingMatchEvaluator(
            train_embeddings=embedding_output["embeddings"],
            train_ids=embedding_output["ids"],
            train_site_pred=embedding_output["site_preds"],
            train_em_pred=embedding_output["em_preds"],
            out_path=Path(config.data.experiment_path)
            / config.data.experiment_name
            / "embedding_predictions_external.json",
            reg_evaluator=evaluator,
            ground_truth_json=Path(config.data.train_json_path),
            save_json=True if not config.other.dry_run else False,
            match_site_and_em=False,
            disable_TQDM=config.other.disable_TQDM,
        )

    evaluators = []
    if config.eval.run_text:
        evaluators.append(text_eval)
    if config.eval.run_embedding:
        evaluators.append(embed_eval_filter)
        evaluators.append(embed_eval_no_filter)
    eval_func = lambda: evaluators

    _, reg = run_epoch_evaluation(
        infer_runner=infer,
        test_loader=test_dataloader,
        pad_idx=config.text.pad_idx,
        do_reg_eval=True,
        evaluators_factory=eval_func,
        config=config,
    )

    print_reg_metrics(reg)


if __name__ == "__main__":
    evaluate()
