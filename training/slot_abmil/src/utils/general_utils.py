from pathlib import Path
from omegaconf import OmegaConf
import torch
import h5py
from src.conf.schema import AppCfg


def process_config(config: AppCfg, is_eval: bool = False) -> AppCfg:
    if not config.other.dry_run and config.data.experiment_name == "":
        raise ValueError(
            "Experiment name must be specified. Please set 'experiment_name'."
        )
    output = Path(config.data.experiment_path) / config.data.experiment_name
    if config.other.dry_run == False:
        output.mkdir(exist_ok=True, parents=True)
        if (
            not is_eval
            and not config.other.dry_run
            and len(list(output.rglob("*.pth"))) > 0
        ):
            raise ValueError(
                f"Checkpoints directory {output} is not empty. Please remove the files before training."
            )

        (output / "checkpoints").mkdir(exist_ok=True, parents=True)
        # Save config to the checkpoint path
        with open(output / "config.yaml", "w") as f:
            OmegaConf.save(config, f)

    assert config.training.precision in [
        "fp16",
        "fp32",
    ], f"Invalid precision {config.training.precision}. Must be 'fp16' or 'fp32'."

    if config.eval.run_embedding:
        assert (
            not config.dataloader.drop_last
        ), "Embedding evaluation requires drop_last=False in the dataloader, otherwise we won't get all embeddings."

    if config.model.model_checkpoint is not None:
        assert Path(
            config.model.model_checkpoint
        ).exists(), f"Model checkpoint {config.model.model_checkpoint} does not exist."

    return config


def get_input_feature_shape(config):
    if (
        Path(config.data.train_data_path).is_dir()
        and len(list(Path(config.data.train_data_path).glob("*.h5"))) > 0
    ):
        data_path = Path(config.data.train_data_path)
    elif (
        Path(config.eval.test_data_path).is_dir()
        and len(list(Path(config.eval.test_data_path).glob("*.h5"))) > 0
    ):
        data_path = Path(config.eval.test_data_path)
    else:
        raise ValueError(
            "Either train_data_path or test_data_path must be a valid directory"
        )
    if data_path.is_dir():
        data_path = list(data_path.glob("*.h5"))[0]
        if not data_path.exists():
            raise FileNotFoundError(f"Couldn't find data at {data_path}")
        with h5py.File(data_path, "r") as f:
            input_feature_shape = f["features"].shape
        return input_feature_shape
    elif data_path.is_file():
        raise NotImplementedError("Support for input files isn't implemented yet")
    else:
        raise ValueError(f"Invalid data path: {data_path}")


def print_info(train_dataloader):
    print(f"Number of sites: {len(train_dataloader.dataset.site_map.keys())}")
    for site in train_dataloader.dataset.site_map.keys():
        print(f"\t{site}: {train_dataloader.dataset.site_map[site]}")
    print(
        f"Number of extraction methods:",
        len(train_dataloader.dataset.extraction_map.keys()),
    )
    for extraction in train_dataloader.dataset.extraction_map.keys():
        print(f"\t{extraction}: {train_dataloader.dataset.extraction_map[extraction]}")
