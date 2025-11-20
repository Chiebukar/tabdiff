# tabdiffusion/model.py
"""
Transformer-based TabDiffusion generator.
Includes:
- conditional token builder
- transformer backbone
- numeric & categorical decoders
- training loss using cosine noise schedule (DDPM-style simplified)
- sampling (simple deterministic reverse steps using predicted embeddings)
"""

from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import cosine_beta_schedule, set_seed

class TabDiffusionGenerator(nn.Module):
    """
    Transformer denoiser that works in token/embedding space for mixed numeric/categorical tabular data.
    """

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: List[int],
        cond_columns: Dict[str, Dict],
        token_dim: int = 192,
        time_embed_dim: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        transformer_ff: int = 512,
        timesteps: int = 100,
        uncond_prob: float = 0.1,
        seed: int = 42
    ):
        super().__init__()
        set_seed(seed)
        self.num_num = num_numeric
        self.cat_cardinalities = cat_cardinalities
        self.cond_specs = cond_columns
        self.cond_columns = list(cond_columns.keys())
        self.token_dim = token_dim
        self.time_embed_dim = time_embed_dim
        self.uncond_prob = uncond_prob
        self.timesteps = timesteps

        # embeddings for numeric features (project scalar -> token) and categoricals (embedding)
        self.num_proj = nn.Linear(1, token_dim)
        self.cat_embeds = nn.ModuleList([nn.Embedding(card, token_dim) for card in cat_cardinalities])

        # condition embeddings
        self.cond_embeds = nn.ModuleDict()
        for col, spec in cond_columns.items():
            typ = spec["type"]
            if typ == "cat":
                card = spec.get("cardinality")
                if card is None:
                    raise ValueError("cardinality required for categorical cond")
                self.cond_embeds[col] = nn.Embedding(card, token_dim)
            elif typ == "binary":
                self.cond_embeds[col] = nn.Embedding(2, token_dim)
            elif typ == "num":
                self.cond_embeds[col] = nn.Linear(1, token_dim)
            else:
                raise ValueError("unknown cond type")

        # time embedding projection
        self.time_proj = nn.Linear(time_embed_dim, token_dim)

        # cond combine projector
        cond_in = token_dim * (1 + len(self.cond_embeds))
        self.cond_proj = nn.Linear(cond_in, token_dim)

        # transformer backbone
        enc_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=transformer_heads, dim_feedforward=transformer_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # output projector and decoders
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.num_decoders = nn.ModuleList([nn.Linear(token_dim, 2) for _ in range(num_numeric)])
        self.cat_decoders = nn.ModuleList([nn.Linear(token_dim, card) for card in cat_cardinalities])

        # noise schedule (cosine)
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod, dtype=torch.float32))
        # precompute sqrt terms
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(torch.tensor(alphas_cumprod, dtype=torch.float32)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - torch.tensor(alphas_cumprod, dtype=torch.float32)))

    def _make_time_embedding(self, t: torch.Tensor):
        """
        Map t in [0,1] to Fourier features then to token_dim.
        t: [B]
        """
        device = t.device
        half = self.time_embed_dim // 2
        freqs = torch.linspace(1.0, 10.0, half, device=device)
        angles = t[:, None] * freqs[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.time_embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(t.shape[0], 1, device=device)], dim=-1)
        return emb  # [B, time_embed_dim]

    def _build_tokens_from_inputs(self, x_num, x_cat, cond_token):
        toks = [cond_token.unsqueeze(1)]  # [B,1,D]
        if x_num is not None and x_num.numel() > 0:
            num_tokens = []
            for i in range(self.num_num):
                feat = x_num[:, i:i+1]  # [B,1]
                num_tokens.append(self.num_proj(feat.unsqueeze(-1)))
            toks.append(torch.cat(num_tokens, dim=1))
        if x_cat is not None and x_cat.numel() > 0:
            cat_tokens = []
            for i, emb in enumerate(self.cat_embeds):
                cat_tokens.append(emb(x_cat[:, i]))
            toks.append(torch.stack(cat_tokens, dim=1))
        return torch.cat(toks, dim=1)  # [B, 1 + num_num + num_cat, D]

    def _cond_token(self, t: torch.Tensor, cond_batch: dict, do_cfg_dropout=False):
        B = t.size(0)
        time_emb = self._make_time_embedding(t)
        t_proj = self.time_proj(time_emb)
        cond_embs = [t_proj]
        for col, layer in self.cond_embeds.items():
            val = cond_batch.get(col, None)
            if val is None:
                if isinstance(layer, nn.Linear):
                    v = torch.zeros(B, device=t.device)
                    cond_embs.append(layer(v.unsqueeze(-1).float()))
                else:
                    idx = torch.zeros(B, dtype=torch.long, device=t.device)
                    cond_embs.append(layer(idx))
            else:
                v = val.to(t.device)
                if isinstance(layer, nn.Linear):
                    cond_embs.append(layer(v.unsqueeze(-1).float()))
                else:
                    cond_embs.append(layer(v.long()))
        cond_full = torch.cat(cond_embs, dim=-1)
        cond_proj = self.cond_proj(cond_full)
        if do_cfg_dropout and torch.rand(1).item() < self.uncond_prob:
            cond_proj = torch.zeros_like(cond_proj)
        return cond_proj

    def forward(self, x_num, x_cat, cond_batch, t):
        """
        Standard forward (denoiser) returning embeddings shaped [B, seq, D]
        t: float tensor [B] in [0,1]
        """
        cond_token = self._cond_token(t, cond_batch, do_cfg_dropout=False)
        tokens = self._build_tokens_from_inputs(x_num, x_cat, cond_token)
        h = self.transformer(tokens)
        return self.out_proj(h)

    def training_loss(self, x_num, x_cat, cond_batch):
        """
        DDPM-style simplified loss in embedding-space: model predicts noise on noisy tokens.
        We compute tokens_clean (embedding representation), add noise q(x_t|x_0), predict noise with model,
        and minimize L2 between predicted noise and true noise.
        """
        B = x_num.size(0)
        device = x_num.device
        # random timestep t in {0..T-1}
        timesteps = self.timesteps
        t_int = torch.randint(0, timesteps, (B,), device=device)
        t = t_int.float() / (timesteps - 1)

        # build clean tokens
        cond_token = self._cond_token(t, cond_batch, do_cfg_dropout=True)
        tokens_clean = self._build_tokens_from_inputs(x_num, x_cat, cond_token)  # [B, seq, D]

        # sample noise
        noise = torch.randn_like(tokens_clean)

        # compute scaling factors from precomputed buffers (index by t_int)
        sqrt_ac = self.sqrt_alphas_cumprod[t_int].view(-1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t_int].view(-1, 1, 1)

        tokens_noisy = sqrt_ac * tokens_clean + sqrt_om * noise

        # predict noise
        pred = self.transformer(tokens_noisy)
        pred = self.out_proj(pred)

        loss = F.mse_loss(pred, tokens_clean)  # equivalent to predicting tokens directly
        return loss

    @torch.no_grad()
    def sample(self, num_samples: int, cond_batch: dict, steps: int = 50, cfg_scale: float = 1.5):
        """
        Generate num_samples rows conditioned on cond_batch (each entry shape (num_samples,)).
        Returns decoded numeric tensor [B, num_num] and categorical indices [B, num_cat].
        The reverse process is a simple iterative denoising in embedding space using the transformer.
        """
        device = next(self.parameters()).device
        B = num_samples
        seq_len = 1 + self.num_num + len(self.cat_cardinalities)
        toks = torch.randn(B, seq_len, self.token_dim, device=device)

        for step in reversed(range(steps)):
            t = torch.full((B,), step / max(1, steps - 1), device=device)
            cond_proj = self._cond_token(t, cond_batch, do_cfg_dropout=False)
            tokens_in = toks.clone()
            tokens_in[:, 0, :] = cond_proj
            toks_cond = self.out_proj(self.transformer(tokens_in))
            if cfg_scale != 1.0:
                null_cond_proj = self._cond_token(t, cond_batch, do_cfg_dropout=True)
                tokens_null = toks.clone()
                tokens_null[:, 0, :] = null_cond_proj
                toks_null = self.out_proj(self.transformer(tokens_null))
                toks_cond = toks_null + cfg_scale * (toks_cond - toks_null)
            toks = toks_cond

        # decode
        num_outs = [dec(toks[:, 1 + i]) for i, dec in enumerate(self.num_decoders)]
        x_num_gen = torch.cat(num_outs, dim=1) if len(num_outs) > 0 else torch.zeros(B, 0, device=device)
        cat_outs = []
        offset = 1 + self.num_num
        for i, dec in enumerate(self.cat_decoders):
            logits = dec(toks[:, offset + i])           # [B, card]
            probs = torch.softmax(logits / 1.0, dim=-1) # temperature=1.0 (can be parameterized)
            idx = torch.multinomial(probs, 1).squeeze(1)
            cat_outs.append(idx)
        x_cat_gen = torch.stack(cat_outs, dim=1) if len(cat_outs) > 0 else torch.zeros(B, 0, dtype=torch.long, device=device)
        return x_num_gen, x_cat_gen
