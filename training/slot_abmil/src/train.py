import time
from pathlib import Path
import os

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])


import hydra  #config managemnet
from tqdm import tqdm
import torch


from src.network.model import Network, get_model
from src.dataloaders.dataloaders import get_dataloaders
from src.utils.text_utils import get_idx_mappings, save_idx_mappings
from src.utils.general_utils import process_config, get_input_feature_shape
from src.conf.schema import AppCfg
from src.evaluation.evaluation_utils.inference import InferenceRunner
from src.evaluation.evaluators.REG_evaluator import REG_Evaluator
from src.evaluation.evaluators.decoder_evaluator import TextReportEvaluator
from src.evaluation.evaluators.embedding_evaluator import EmbeddingMatchEvaluator
from src.trainer.trainer import Trainer
from src.evaluation.evaluation_utils.embedding_extractor import EmbeddingExtractor
from src.evaluation.evaluation_utils.epoch_eval import run_epoch_evaluation
from src.trainer.training_utils import print_metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(config: AppCfg) -> None:
    print("DEBUG train_json_path:", config.data.train_json_path)
    print("DEBUG train_data_path:", config.data.train_data_path)

    config = process_config(config)

    train_dataloader, test_dataloader = get_dataloaders(config, augment_train=True)

    input_feature_shape = get_input_feature_shape(config)

    idx_to_word, idx_to_site, idx_to_extraction = get_idx_mappings(train_dataloader)
    save_idx_mappings(
        idx_to_word,
        idx_to_site,
        idx_to_extraction,
        Path(config.data.experiment_path) / config.data.experiment_name,
    )

    # ─── slot supervision: model에 slot head 전달 ──────────────────────
    slot_num_classes = None
    if getattr(config.data, "use_slot_supervision", False):
        slot_cfg = getattr(train_dataloader, "slot_cfg", None)
        if slot_cfg is not None and slot_cfg.get("num_classes"):
            slot_num_classes = slot_cfg["num_classes"]
            n_organs = len(slot_num_classes)
            n_heads = sum(len(v) for v in slot_num_classes.values())
            print(f"[slot] passing to model: {n_organs} organs, {n_heads} heads")
        else:
            print("[slot] WARNING: use_slot_supervision=True but slot_cfg is missing")

    network = get_model(
        config,
        idx_to_word,
        idx_to_site,
        idx_to_extraction,
        input_feature_shape,
        slot_num_classes=slot_num_classes,
    ).to(config.model.train_device)

    trainer = Trainer(
        network,
        config,
        idx_to_site=idx_to_site,
        idx_to_extraction=idx_to_extraction,
        idx_to_word=idx_to_word,
    )

    infer = InferenceRunner(
        network,
        device=config.model.train_device,
        precision=(
            torch.float16 if config.training.precision == "fp16" else torch.float32
        ),
    )
    evaluator, text_eval, embed_eval_filter, embed_eval_no_filter = (
        None,
        None,
        None,
        None,
    )
    #추가
    print("DEBUG: eval flags -> run_text =", config.eval.run_text,
          ", run_embedding =", config.eval.run_embedding)
    #추가 
    if config.eval.run_embedding or config.eval.run_text:
        print("DEBUG: Creating REG_Evaluator (this will load aaditya/Llama3-OpenBioLLM-8B)...")
        evaluator = REG_Evaluator(
            "aaditya/Llama3-OpenBioLLM-8B", device=config.model.eval_device
        )
        print("DEBUG: REG_Evaluator created.")
    
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

    for i in tqdm(
        range(config.training.epochs),
        "Epochs",
        leave=False,
        position=0,
        disable=config.other.disable_TQDM,
    ):
        start_time = time.time()
        train_output = trainer.train_epoch(train_dataloader)
        training_time = time.time() - start_time
        do_reg_eval = (i + 1) % config.eval.eval_rate == 0
        if (
            config.eval.run_embedding
            and embedding_extractor is not None
            and do_reg_eval
        ):
            embedding_start_time = time.time()
            embedding_output = embedding_extractor.get_embeddings(train_dataloader)
            embbeding_time = time.time() - embedding_start_time

            embed_eval_filter = EmbeddingMatchEvaluator(
                train_embeddings=embedding_output["embeddings"],
                train_ids=embedding_output["ids"],
                train_site_pred=embedding_output["site_preds"],
                train_em_pred=embedding_output["em_preds"],
                out_path=Path(config.data.experiment_path)
                / config.data.experiment_name
                / "embedding_predictions_match_site_em.json",
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
        else:
            embbeding_time = None

        if (config.eval.run_text or config.eval.run_embedding) and do_reg_eval:
            evaluators = []
            if config.eval.run_text:
                evaluators.append(text_eval)
            if config.eval.run_embedding:
                evaluators.append(embed_eval_filter)
                evaluators.append(embed_eval_no_filter)
            eval_func = lambda: evaluators
        else:
            eval_func = None
        eval_start_time = time.time()
        if do_reg_eval:
            fast, reg = run_epoch_evaluation(
                infer_runner=infer,
                test_loader=test_dataloader,
                pad_idx=config.text.pad_idx,
                do_reg_eval=do_reg_eval,
                evaluators_factory=eval_func,
                config=config,
            )
            evaluation_time = time.time() - eval_start_time
            print_metrics(
                train_output,
                fast,
                reg,
                start_time,
                i,
                config,
                training_time,
                embbeding_time,
                evaluation_time,
            )
            print()
        else:
            # print("Finished Epoch", i + 1)
            pass
        if not config.other.dry_run:
            torch.save(
                network.state_dict(),
                f"{Path(config.data.experiment_path)/config.data.experiment_name}/checkpoints/network_epoch_{i+1:03d}.pth",
            )

    del train_dataloader
    del test_dataloader


if __name__ == "__main__":
    train()
