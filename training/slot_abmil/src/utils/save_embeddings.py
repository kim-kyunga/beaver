from pathlib import Path

import hydra
import h5py

from src.network.model import Network, get_model
from src.dataloaders.dataloaders import get_dataloaders
from src.utils.general_utils import process_config, get_input_feature_shape
from src.utils.text_utils import read_idx_mappings, get_idx_mappings, save_idx_mappings

from src.conf.schema import AppCfg
from src.evaluation.evaluation_utils.embedding_extractor import EmbeddingExtractor


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(config: AppCfg) -> None:

    config = process_config(config)
    assert (
        config.model.model_checkpoint is not None
    ), "Model checkpoint must be specified to extract embeddings"

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

    embedding_extractor = EmbeddingExtractor(
        network,
        config,
    )

    for dataloader, name in zip([train_dataloader, test_dataloader], ["train", "test"]):
        embedding_output = embedding_extractor.get_embeddings(dataloader)
        embedding = embedding_output["embeddings"]
        ids = embedding_output["ids"]
        output_dir = (
            Path(config.data.experiment_path)
            / config.data.experiment_name
            / "embeddings"
            / name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        for embedding, id in zip(embedding, ids):
            embedding = embedding.numpy()
            with h5py.File(output_dir / f"{id}.h5", "w") as f:
                f["embedding"] = embedding
        print(
            f"Extracted embeddings for {len(ids)} samples and saved to {Path(config.data.experiment_path) / 'embeddings' / name}"
        )


if __name__ == "__main__":
    train()
