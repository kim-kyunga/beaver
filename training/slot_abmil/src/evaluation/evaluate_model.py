from pathlib import Path

import hydra
import h5py
import torch

from src.network.model import get_model
from src.dataloaders.dataloaders import get_dataloaders
from src.utils.general_utils import process_config, get_input_feature_shape
from src.conf.schema import AppCfg
from src.evaluation.evaluation_utils.embedding_extractor import EmbeddingExtractor
from src.network.model import get_model
from src.dataloaders.dataloaders import get_dataloaders
from src.utils.general_utils import process_config, get_input_feature_shape
from src.conf.schema import AppCfg
from src.evaluation.evaluators.REG_evaluator import REG_Evaluator
from src.evaluation.evaluators.decoder_evaluator import TextReportEvaluator
from src.evaluation.evaluators.embedding_evaluator import EmbeddingMatchEvaluator
from src.evaluation.evaluation_utils.embedding_extractor import EmbeddingExtractor
from src.evaluation.evaluation_utils.inference import InferenceRunner
from src.utils.text_utils import read_idx_mappings, get_idx_mappings, save_idx_mappings
from src.evaluation.evaluation_utils.epoch_eval import run_epoch_evaluation
from src.trainer.training_utils import print_reg_metrics


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def evaluate(config: AppCfg) -> None:

    config = process_config(config, is_eval=True)
    assert (
        config.model.model_checkpoint is not None
    ), "Model checkpoint must be specified to extract embeddings"
    assert (
        config.eval.run_text or config.eval.run_embedding
    ), "Needs to evaluate at least one of them"

    train_dataloader, test_dataloader = get_dataloaders(config, augment_train=True)

    try:
        idx_to_word, idx_to_site, idx_to_extraction = read_idx_mappings(
            Path(config.data.experiment_path) / config.data.experiment_name
        )
    except Exception as e:
        print("Couldn't load mappings, creating instead")
        idx_to_word, idx_to_site, idx_to_extraction = get_idx_mappings(train_dataloader)

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
    if config.eval.run_embedding or config.eval.run_text:
        evaluator = REG_Evaluator(
            "aaditya/Llama3-OpenBioLLM-8B", device=config.model.eval_device
        )
    embedding_extractor = None
    if config.eval.run_embedding:
        embedding_extractor = EmbeddingExtractor(
            network,
            config,
        )

    if config.eval.run_text and evaluator is not None:
        text_eval = TextReportEvaluator(
            idx_to_word=idx_to_word,
            idx_to_site=idx_to_site,
            idx_to_EM=idx_to_extraction,
            pad_idx=config.text.pad_idx,
            out_path=Path(config.data.experiment_path)
            / config.data.experiment_name
            / "decoder_predictions.json",
            reg_evaluator=evaluator,
            ground_truth_json=Path(config.data.train_json_path),
            save_json=True if not config.other.dry_run else False,
            disable_TQDM=config.other.disable_TQDM,
        )

    if config.eval.run_embedding and embedding_extractor is not None:
        print("Getting text embeddings")
        embedding_output = embedding_extractor.get_embeddings(train_dataloader)
        print("Finished getting text embeddings")
        embed_eval_filter = EmbeddingMatchEvaluator(
            train_embeddings=embedding_output["embeddings"],
            train_ids=embedding_output["ids"],
            train_site_pred=embedding_output["site_preds"],
            train_em_pred=embedding_output["em_preds"],
            out_path=Path(config.data.experiment_path)
            / config.data.experiment_name
            / "embedding_predictions.json",
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
            / "embedding_predictions.json",
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
