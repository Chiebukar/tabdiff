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

from typing import List, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_time_embed(t: torch.Tensor, dim: int, max_freq: float = 10.0) -> torch.Tensor:
    """Fourier/time embedding for scalar timesteps t ∈ [0,1]."""
    device = t.device
    half = dim // 2
    freqs = torch.linspace(1.0, max_freq, half, device=device)
    angles = t[:, None] * freqs[None, :] * 2 * math.pi
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=device)], dim=-1)
    return emb


class TabDiffusionGenerator(nn.Module):
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
        uncond_prob: float = 0.1
    ):
        """
        Args:
            num_numeric: number of numeric feature columns
            cat_cardinalities: list of cardinalities for categorical features (order must match data)
            cond_columns: dict mapping cond_col -> {"type":"cat"/"num"/"binary", "cardinality":int (if cat)}
        """
        super().__init__()
        self.num_num = num_numeric
        self.cat_cardinalities = cat_cardinalities
        self.cond_specs = cond_columns
        self.cond_columns = list(cond_columns.keys())
        self.token_dim = token_dim
        self.time_embed_dim = time_embed_dim
        self.uncond_prob = uncond_prob

        # input feature projections / embeddings
        self.num_proj = nn.Linear(1, token_dim)
        self.cat_embeds = nn.ModuleList([nn.Embedding(card, token_dim) for card in cat_cardinalities])

        # conditioning embeddings (time + condition features)
        self.cond_embeds = nn.ModuleDict()
        for col, spec in cond_columns.items():
            typ = spec.get("type")
            if typ == "cat":
                card = spec.get("cardinality")
                if card is None:
                    raise ValueError(f"categorical condition {col} requires 'cardinality' in cond_columns")
                self.cond_embeds[col] = nn.Embedding(card, token_dim)
            elif typ == "num":
                self.cond_embeds[col] = nn.Linear(1, token_dim)
            elif typ == "binary":
                self.cond_embeds[col] = nn.Embedding(2, token_dim)
            else:
                raise ValueError(f"Unknown conditional type {typ} for {col}")

        # learned null token for classifier-free guidance
        self.null_cond = nn.Parameter(torch.zeros(token_dim))

        # time projection
        self.time_proj = nn.Linear(time_embed_dim, token_dim)

        # project combined conds -> single token
        cond_in_dim = token_dim * (1 + len(self.cond_embeds))
        self.cond_proj = nn.Linear(cond_in_dim, token_dim)

        # transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff,
            batch_first=True
        )
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # final output projection (embedding space)
        self.out_proj = nn.Linear(token_dim, token_dim)

        # decoders:
        # numeric: output (mu, logvar) per numeric feature
        self.num_decoders = nn.ModuleList([nn.Linear(token_dim, 2) for _ in range(num_numeric)])
        # categorical: output logits over each cardinality
        self.cat_decoders = nn.ModuleList([nn.Linear(token_dim, card) for card in cat_cardinalities])

    # ---- helpers to build token sequences ----
    def _build_tokens_from_inputs(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor], cond_token: torch.Tensor):
        """
        Build token sequence: [cond_token, num_tokens..., cat_tokens...]
        x_num: [B, num_num]
        x_cat: [B, num_cat] (long)
        cond_token: [B, D]
        """
        toks = [cond_token.unsqueeze(1)]  # [B,1,D]
        if x_num is not None and x_num.numel() > 0:
            num_tokens = []
            for i in range(self.num_num):
                feat = x_num[:, i:i+1]            # [B,1]
                num_tokens.append(self.num_proj(feat.unsqueeze(-1)))  # [B,1,D]
            toks.append(torch.cat(num_tokens, dim=1))  # [B, num_num, D]
        if x_cat is not None and x_cat.numel() > 0:
            cat_tokens = []
            for i, emb in enumerate(self.cat_embeds):
                cat_tokens.append(emb(x_cat[:, i]))
            toks.append(torch.stack(cat_tokens, dim=1))  # [B, num_cat, D]
        return torch.cat(toks, dim=1)  # [B, 1 + num_num + num_cat, D]

    def _cond_token(self, t: torch.Tensor, cond_batch: Dict[str, torch.Tensor], do_cfg_dropout: bool = False):
        """
        Build a combined conditioning vector from time and cond columns.
        cond_batch may contain None for some keys -> will be filled with zeros/learned null.
        """
        B = t.size(0)
        device = t.device
        time_emb = fourier_time_embed(t, self.time_embed_dim)  # [B, time_embed_dim]
        t_proj = self.time_proj(time_emb)  # [B, D]

        cond_embs = [t_proj]
        for col, emb_layer in self.cond_embeds.items():
            val = cond_batch.get(col, None)
            if val is None:
                # default zero vector (embedding will map zeros) OR learned null depending on dropout below
                if isinstance(emb_layer, nn.Linear):
                    v = torch.zeros(B, device=device)
                    cond_embs.append(emb_layer(v.unsqueeze(-1).float()))
                else:
                    idx = torch.zeros(B, dtype=torch.long, device=device)
                    cond_embs.append(emb_layer(idx))
            else:
                v = val.to(device)
                if isinstance(emb_layer, nn.Linear):
                    cond_embs.append(emb_layer(v.unsqueeze(-1).float()))
                else:
                    cond_embs.append(emb_layer(v.long()))

        cond_full = torch.cat(cond_embs, dim=-1)  # [B, token_dim * (1+num_conds)]
        cond_proj = self.cond_proj(cond_full)     # [B, D]

        # classifier-free guidance dropout: replace cond token by learned null token sometimes
        if do_cfg_dropout and torch.rand(1).item() < self.uncond_prob:
            cond_proj = self.null_cond.unsqueeze(0).expand(B, -1)

        return cond_proj

    # ---- forward for training (returns embeddings) ----
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, cond_batch: Dict[str, torch.Tensor], t: torch.Tensor):
        cond_token = self._cond_token(t, cond_batch, do_cfg_dropout=False)
        tokens = self._build_tokens_from_inputs(x_num, x_cat, cond_token)
        h = self.model(tokens)
        return self.out_proj(h)

    # ---- training loss ----
    def training_loss(self, x_num: torch.Tensor, x_cat: torch.Tensor, cond_batch: Dict[str, torch.Tensor]):
        """
        Compute reconstruction loss: numeric NLL + categorical CE.
        x_num: [B, num_num] (float)
        x_cat: [B, num_cat] (long)
        cond_batch: dict of conditioning tensors (B,)
        """
        B = x_num.size(0)
        device = x_num.device
        t = torch.rand(B, device=device)

        cond_token = self._cond_token(t, cond_batch, do_cfg_dropout=True)
        tokens_clean = self._build_tokens_from_inputs(x_num, x_cat, cond_token)  # [B, seq, D]

        noise = torch.randn_like(tokens_clean)
        tokens_noisy = tokens_clean + noise

        pred = self.model(tokens_noisy)
        pred = self.out_proj(pred)

        # numeric predictions -> (mu, logvar) per numeric feature
        num_mus = []
        num_logvars = []
        for i, dec in enumerate(self.num_decoders):
            out = dec(pred[:, 1 + i])  # [B, 2]
            num_mus.append(out[:, :1])
            num_logvars.append(out[:, 1:])

        mu = torch.cat(num_mus, dim=1)
        logvar = torch.cat(num_logvars, dim=1).clamp(min=-10.0, max=10.0)

        # gaussian NLL
        nll = 0.5 * (((x_num - mu) ** 2) / (logvar.exp() + 1e-6) + logvar + math.log(2 * math.pi))
        num_loss = nll.mean()

        # categorical predictions -> cross entropy per categorical column
        cat_losses = []
        offset = 1 + self.num_num
        for i, dec in enumerate(self.cat_decoders):
            logits = dec(pred[:, offset + i])
            cat_losses.append(F.cross_entropy(logits, x_cat[:, i]))
        cat_loss = torch.stack(cat_losses).mean() if len(cat_losses) > 0 else torch.tensor(0.0, device=device)

        return num_loss + cat_loss

    # ---- sampling ----
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        cond_batch: Dict[str, torch.Tensor],
        steps: int = 50,
        cfg_scale: float = 1.5,
        sampling_kwargs: Optional[Dict] = None
    ):
        """
        Generate synthetic samples.

        Args:
            num_samples: number of rows to generate (B)
            cond_batch: dict mapping cond_col -> tensor of shape (B,) (if None for some cols, handled)
            steps: iterative denoising steps (embedding-space)
            cfg_scale: classifier-free guidance scale
            sampling_kwargs: {
                "temperature": float (default 1.0),
                "top_k": Optional[int],
                "deterministic_num": bool (default False)
            }

        Returns:
            x_num_gen: torch.FloatTensor [B, num_num]
            x_cat_gen: torch.LongTensor [B, num_cat]
        """
        sampling_kwargs = sampling_kwargs or {}
        temperature = float(sampling_kwargs.get("temperature", 1.0))
        top_k = sampling_kwargs.get("top_k", None)
        deterministic_num = bool(sampling_kwargs.get("deterministic_num", False))

        device = next(self.parameters()).device
        B = num_samples
        seq_len = 1 + self.num_num + len(self.cat_cardinalities)

        toks = torch.randn(B, seq_len, self.token_dim, device=device)

        for step in reversed(range(steps)):
            t = torch.full((B,), step / max(1, steps), device=device)
            # cond token (normal and null for CFG)
            cond_proj = self._cond_token(t, cond_batch, do_cfg_dropout=False)
            tokens_in = toks.clone()
            tokens_in[:, 0, :] = cond_proj
            toks_cond = self.out_proj(self.model(tokens_in))

            if cfg_scale != 1.0:
                null_cond_proj = self._cond_token(t, cond_batch, do_cfg_dropout=True)
                tokens_null = toks.clone()
                tokens_null[:, 0, :] = null_cond_proj
                toks_null = self.out_proj(self.model(tokens_null))
                toks_cond = toks_null + cfg_scale * (toks_cond - toks_null)

            toks = toks_cond

        # decode numerics (mu, logvar) and sample
        num_vals = []
        for i, dec in enumerate(self.num_decoders):
            out = dec(toks[:, 1 + i])  # [B, 2]
            mu = out[:, 0]
            logvar = out[:, 1].clamp(min=-10.0, max=10.0)
            if deterministic_num:
                sampled = mu
            else:
                std = (0.5 * logvar).exp()
                sampled = mu + std * torch.randn_like(mu)
            num_vals.append(sampled.unsqueeze(1))
        x_num_gen = torch.cat(num_vals, dim=1) if len(num_vals) > 0 else torch.zeros(B, 0, device=device)

        # decode categories stochastically
        cat_outs = []
        offset = 1 + self.num_num
        for i, dec in enumerate(self.cat_decoders):
            logits = dec(toks[:, offset + i])  # [B, card]
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                probs_topk = torch.softmax(topk_vals / temperature, dim=-1)
                sampled_topk = torch.multinomial(probs_topk, num_samples=1).squeeze(1)
                idx = topk_idx.gather(1, sampled_topk.unsqueeze(-1)).squeeze(1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            cat_outs.append(idx)
        x_cat_gen = torch.stack(cat_outs, dim=1) if len(cat_outs) > 0 else torch.zeros(B, 0, dtype=torch.long, device=device)

        return x_num_gen, x_cat_gen