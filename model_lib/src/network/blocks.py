from tqdm import tqdm
import torch
import math
from torch import nn
from collections import defaultdict
from src.conf.schema import AppCfg

def sinusoidal_positions(L, d_model, device, base=10000.0):
    pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    div = torch.exp(-math.log(base) * i / d_model)
    pe = torch.zeros(L, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        context_dim,
        vocab_size,
        corpus,
        config: AppCfg,
        max_seq_len=20,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = config.model.embed_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = config.text.pad_idx
        self.sos_idx = config.text.sos_idx
        self.eos_idx = config.text.eos_idx
        assert (
            self.embed_dim % 2 == 0
        ), "Need embed dim to be even, due to positional embeddings"

        self.token_embedding = nn.Embedding(
            vocab_size, self.embed_dim, padding_idx=self.pad_idx
        )
        self.pos_embedding = None
        if config.model.trainable_pos_emb:
            self.pos_embedding = nn.Embedding(max_seq_len, self.embed_dim)

        self.context_proj = nn.Linear(context_dim, self.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=config.model.num_heads_transformer,
            dim_feedforward=config.model.ff_dim,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.model.num_layers_transformer
        )

        self.batch_first = bool(getattr(decoder_layer, "batch_first", True))

        self.output_proj = nn.Linear(self.embed_dim, vocab_size)

        self.n_gram_table = self.build_ngram_table(corpus, config.text.n_gram)
        self.n = config.text.n_gram
        self.corpus = corpus

    def update_n_gram(self, n_gram):
        self.n_gram_table = self.build_ngram_table(self.corpus, n_gram)
        self.n = n_gram

    def build_ngram_table(self, tokenised_corpus, n):
        if n is None or n < 1:
            return None
        cont = defaultdict(set)
        for sent in tokenised_corpus:
            for i in range(1, n):
                prefix = tuple(sent[0:i])
                nxt = sent[i]
                cont[prefix].add(nxt)
            for i in range(len(sent) - n + 1):
                prefix = tuple(sent[i : i + n - 1])
                nxt = sent[i + n - 1]
                cont[prefix].add(nxt)


        return cont

    def _apply_ngram_mask_batch(self, logits, decoder_in):
        if self.n_gram_table is None:
            return logits

        B, T, _ = logits.shape
        masked = logits.view(B, T, -1)

        for b in range(B):
            for t in range(T):
                prefix = tuple(
                    decoder_in[b, max(0, t + 1 - self.n + 1) : t + 1].tolist()
                )
                masked[b, t] = self.apply_ngram_mask(masked[b, t], prefix)
        return masked

    def apply_ngram_mask(self, logits, prefix):
        allowed = self.n_gram_table.get(prefix, None)
        if allowed is None or len(allowed) == 0:
            if allowed is not None:
                assert (
                    prefix[-1] == self.sos_idx
                ), f"Prefix: {prefix} should not be possible to generate..."
            return logits

        if prefix[-1] == self.eos_idx or prefix[-1] == self.pad_idx:
            allowed = [self.pad_idx]
        mask = torch.full_like(logits, float("-inf"))
        mask[list(allowed)] = 0.0
        logits = logits + mask
        return logits

    def get_pos_emb(self, token_emb):
        L = token_emb.size(1)
        if self.pos_embedding is None:
            pos = sinusoidal_positions(L, self.embed_dim, token_emb.device)
            tgt = token_emb + pos.unsqueeze(0)
        else:
            pos_ids = torch.arange(L, device=token_emb.device).unsqueeze(0)
            tgt = token_emb + self.pos_embedding(pos_ids)

        return tgt

    def forward_with_target_seq(self, context, target_seq):

        B, T = target_seq.shape
        device = target_seq.device

        sos_col = torch.full((B, 1), self.sos_idx, device=device, dtype=torch.long)

        decoder_in = torch.cat([sos_col, target_seq[:, :-1]], dim=1)

        tok_emb = self.token_embedding(decoder_in)
        tgt = self.get_pos_emb(tok_emb)

        tgt_key_padding_mask = decoder_in.eq(self.pad_idx)
        tgt_mask = torch.triu(torch.ones((T, T), device=device), 1).bool()

        memory = self.context_proj(context).unsqueeze(1)
        out = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_proj(out)

        return logits

    def forward_auto_regressive(self, context):
        B = context.size(0)

        device = context.device
        memory = self.context_proj(context).unsqueeze(1)

        generated = torch.full((B, 1), self.sos_idx, device=device, dtype=torch.long)

        for t in range(self.max_seq_len):
            token_emb = self.token_embedding(generated)
            tgt = self.get_pos_emb(token_emb)

            tgt_mask = torch.triu(
                torch.full((t + 1, t + 1), float("-inf"), device=device), diagonal=1
            )

            out = self.transformer_decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
            )
            logits = self.output_proj(out[:, -1, :])
            if self.n_gram_table is not None:
                logit_list = []
                for j in range(B):
                    prefix = tuple(
                        generated[j, max(0, generated.size(1) - self.n + 1) :].tolist()
                    )
                    logit_list.append(self.apply_ngram_mask(logits[j], prefix))
                logits = torch.stack(logit_list, dim=0)

            next_token = torch.argmax(logits, dim=-1)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        return generated

    def forward(self, context, target_seq=None):
        if target_seq is not None:
            return self.forward_with_target_seq(context, target_seq)
        else:
            return self.forward_auto_regressive(context)
