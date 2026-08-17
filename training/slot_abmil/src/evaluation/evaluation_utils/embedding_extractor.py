from typing import Tuple, Dict, List

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast

from src.conf.schema import AppCfg
from src.network.model import Network


class EmbeddingExtractor:
    def __init__(
        self,
        network: Network,
        config: AppCfg,
    ):
        self.network = network
        self.config = config

        self.precision = (
            torch.float16 if config.training.precision == "fp16" else torch.float32
        )
        self.use_amp = self.precision == torch.float16

    def extract_batch(
        self,
        img: torch.Tensor,
        lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, lens = (
            img.to(self.config.model.train_device),
            lens.to(self.config.model.train_device),
        )
        with autocast(
            dtype=self.precision,
            device_type=str(self.config.model.train_device),
            enabled=self.use_amp,
        ) and torch.inference_mode():
            output = self.network(img, lens=lens)
            site_pred, EM_pred, embeddings = (
                output["site_logits"],
                output["em_logits"],
                output["embeddings"],
            )
            if len(site_pred.shape) == 1:
                site_pred = site_pred.unsqueeze(0)
                EM_pred = EM_pred.unsqueeze(0)
                embeddings = embeddings.unsqueeze(0)

        embeddings = embeddings.detach().cpu()
        site_preds = site_pred.argmax(-1).detach().cpu()
        EM_preds = EM_pred.argmax(-1).detach().cpu()
        return embeddings, site_preds, EM_preds

    def get_embeddings(self, dataloader: DataLoader):
        embedding_list = []
        site_preds_list = []
        EM_preds_list = []
        id_list = []
        self.network.eval()
        for img, lens, _, _, _, ids, *_ in tqdm(
            dataloader, desc="Training batch", leave=False, position=1, disable=self.config.other.disable_TQDM
        ):
            embeddings, site_preds, EM_preds = self.extract_batch(img, lens)
            embedding_list.append(embeddings)
            site_preds_list.append(site_preds)
            EM_preds_list.append(EM_preds)
            id_list.extend(ids)

        embeddings = torch.cat(embedding_list, dim=0)
        site_preds = torch.cat(site_preds_list, dim=0)
        EM_preds = torch.cat(EM_preds_list, dim=0)

        return {
            "embeddings": embeddings,
            "ids": id_list,
            "site_preds": site_preds,
            "em_preds": EM_preds,
        }
