from typing import Optional
import torch
import torch.nn as nn


class MILModel(nn.Module):
    def __init__(
        self,
        n_feats: int,
        dropout_rate: float,
        embedding_size: int,
        encoder_layers: int,
        num_attention_heads: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads

        self.encoder = nn.Sequential(
            nn.Linear(n_feats, embedding_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embedding_size) if use_batchnorm else nn.Identity(),
            *[
                nn.Sequential(
                    nn.Linear(embedding_size, embedding_size),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(embedding_size) if use_batchnorm else nn.Identity(),
                )
                for _ in range(encoder_layers - 1)
            ],
        )

        self.attention_heads = nn.ModuleList(
            [Attention(embedding_size) for _ in range(num_attention_heads)]
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.initialize_weights()

    def forward(
        self,
        bags: torch.Tensor,
        lens: torch.Tensor,
    ):
        B, N, F = bags.shape
        D = self.encoder[0].out_features

        embeddings = self.encoder(bags.reshape(-1, F))
        embeddings = embeddings.reshape(B, N, D)

        per_head_sums = []

        for head in self.attention_heads:
            scores = self._masked_attention_scores(head, embeddings, lens)

            weighted_sum = (scores * embeddings).sum(dim=1)
            per_head_sums.append(weighted_sum)

        per_head_sums = torch.stack(per_head_sums, dim=1)

        concat_embeds = per_head_sums.reshape(B, -1)
        concat_embeds = self.dropout(concat_embeds)
        assert concat_embeds.shape[1] == self.embedding_size * self.num_attention_heads
        return concat_embeds

    def _masked_attention_scores(
        self,
        head: nn.Module,
        embeddings: torch.Tensor,
        lens: torch.Tensor
    ) -> torch.Tensor:
        B, N, _ = embeddings.shape
        attention_scores = head(embeddings)

        idx = torch.arange(N, device=attention_scores.device).repeat(B, 1)
        attention_mask = (idx < lens.unsqueeze(-1)).unsqueeze(-1)

        mask_value = -1e4 if attention_scores.dtype == torch.float16 else -1e8
        masked_attention = torch.where(
            attention_mask,
            attention_scores,
            torch.full_like(attention_scores, mask_value),
        )
        return torch.softmax(masked_attention, dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def device(self):
        return self.encoder[0].bias.device

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def Attention(n_in: int, n_latent: Optional[int] = None) -> nn.Module:
    n_latent = n_latent or (n_in + 1) // 2
    return nn.Sequential(
        nn.Linear(n_in, n_latent),
        nn.Tanh(),
        nn.Linear(n_latent, 1),
    )
