# tabdiffusion/sampler.py
"""
Helper sampling utilities used by the high-level TabDiffusion wrapper.

Provides:
- functions to build cond_batch for sampling (empirical marginals / overrides)
- decode generated tensors to pandas DataFrame with original column names
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import pandas as pd


def build_cond_batch_from_overrides(
    cond_specs: Dict[str, Dict],
    num_samples: int,
    cond_overrides: Optional[Dict] = None,
    empirical_cond_probs: Optional[Dict[str, pd.Series]] = None,
    label_encoders: Optional[Dict[str, object]] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Build cond_batch dict mapping cond_name -> torch.tensor(shape=(num_samples,)) to feed model.sample().
    If an override is provided for a column, use that; otherwise sample from empirical_cond_probs (if provided)
    or fall back to zeros.
    """
    cond_overrides = cond_overrides or {}
    empirical_cond_probs = empirical_cond_probs or {}
    device = torch.device(device) if isinstance(device, str) else device

    batch = {}
    for col, spec in cond_specs.items():
        typ = spec.get("type")
        if col in cond_overrides:
            val = cond_overrides[col]
            # categorical string -> index via label_encoders if possible
            if typ == "cat":
                if isinstance(val, str) and label_encoders and col in label_encoders:
                    idx = int(label_encoders[col].transform([val])[0])
                    batch[col] = torch.full((num_samples,), idx, dtype=torch.long, device=device)
                elif isinstance(val, (list, np.ndarray)):
                    arr = np.array(val)
                    if arr.shape[0] == num_samples:
                        batch[col] = torch.tensor(arr.astype(int), dtype=torch.long, device=device)
                    else:
                        # broadcast scalar
                        batch[col] = torch.full((num_samples,), int(arr.ravel()[0]), dtype=torch.long, device=device)
                else:
                    batch[col] = torch.full((num_samples,), int(val), dtype=torch.long, device=device)
            elif typ == "binary":
                batch[col] = torch.full((num_samples,), int(val), dtype=torch.long, device=device)
            else:  # numeric
                if isinstance(val, (list, np.ndarray)):
                    arr = np.array(val).astype(float)
                    if arr.shape[0] == num_samples:
                        batch[col] = torch.tensor(arr, dtype=torch.float32, device=device)
                    else:
                        batch[col] = torch.full((num_samples,), float(arr.ravel()[0]), dtype=torch.float32, device=device)
                else:
                    batch[col] = torch.full((num_samples,), float(val), dtype=torch.float32, device=device)
        else:
            # sample from empirical distribution if available
            if typ in ("cat", "binary"):
                probs = empirical_cond_probs.get(col, None)
                if probs is not None:
                    choices = probs.index.to_numpy()
                    p = probs.values.astype(float)
                    idxs = np.random.choice(choices, size=num_samples, p=p)
                    batch[col] = torch.tensor(idxs.astype(int), dtype=torch.long, device=device)
                else:
                    batch[col] = torch.zeros(num_samples, dtype=torch.long, device=device)
            else:
                stats = empirical_cond_probs.get(col, None)
                if stats is not None and isinstance(stats, dict) and "mean" in stats:
                    mu = float(stats["mean"])
                    sigma = float(stats.get("std", 1.0))
                    vals = np.random.normal(loc=mu, scale=sigma, size=num_samples)
                    batch[col] = torch.tensor(vals, dtype=torch.float32, device=device)
                else:
                    batch[col] = torch.zeros(num_samples, dtype=torch.float32, device=device)
    return batch


def decode_generated_to_df(
    x_num_gen: np.ndarray,
    x_cat_gen: np.ndarray,
    num_col_names: List[str],
    cat_col_names: List[str],
    scaler=None,
    label_encoders: Optional[Dict[str, object]] = None
) -> pd.DataFrame:
    """
    Decode generated numeric & categorical arrays into pandas DataFrame.
    - x_num_gen: [B, num_num] floats (in scaled space if scaler provided)
    - x_cat_gen: [B, num_cat] int indices
    - scaler: fitted scaler with .inverse_transform
    - label_encoders: dict col->LabelEncoder for inverse mapping
    """
    if scaler is not None and x_num_gen.size > 0:
        try:
            x_num_orig = scaler.inverse_transform(x_num_gen)
        except Exception:
            x_num_orig = x_num_gen
    else:
        x_num_orig = x_num_gen

    df = pd.DataFrame(x_num_orig, columns=num_col_names) if x_num_orig.size > 0 else pd.DataFrame()

    # decode categorical indices back to labels (if label_encoders provided)
    for i, col in enumerate(cat_col_names):
        if x_cat_gen.size == 0 or x_cat_gen.shape[1] <= i:
            df[col] = pd.NA
            continue
        idxs = x_cat_gen[:, i].astype(int)
        if label_encoders and col in label_encoders:
            try:
                df[col] = label_encoders[col].inverse_transform(idxs)
            except Exception:
                df[col] = idxs
        else:
            df[col] = idxs
    return df
