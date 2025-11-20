# tabdiffusion/core.py
"""
High-level TabDiffusion wrapper.

Usage example:
    td = TabDiffusion(df, target='isFraud', conditionals=['ProductCD', 'DeviceType'])
    td.fit(epochs=10, batch_size=256)
    samples = td.sample(num_samples=1000, labels_to_sample=[1,0], proportions=[0.5,0.5])
"""

import os
import math
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from .model import TabDiffusionGenerator
from .data import TabularDataset
from .sampler import build_cond_batch_from_overrides, decode_generated_to_df
from .utils import sanitize_categoricals 


class TabDiffusion:
    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        conditionals: Optional[List[str]] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        High level API that prepares data, fits TabDiffusionGenerator, and samples.
        - df: raw pandas DataFrame (will be copied internally)
        - target: name of the target column (optional)
        - conditionals: list of column names to use as conditioning (categorical or numeric). If None, default to all categorical columns.
        """
        self.df_raw = sanitize_categoricals(df.copy())
        self.target = target
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # determine numeric and categorical columns automatically
        self.num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if target is not None and target in self.num_cols:
            self.num_cols = [c for c in self.num_cols if c != target]
        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # choose conditionals default -> all categorical cols if not provided
        if conditionals is None:
            self.cond_cols = [c for c in self.cat_cols]
        else:
            self.cond_cols = conditionals

        # placeholders for fitted transformers / encoders
        self.scaler = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.cond_empirical_probs: Dict[str, pd.Series] = {}  # for categorical cond sampling
        self.cond_empirical_stats: Dict[str, Dict] = {}        # for numeric cond sampling (mean/std)
        self.model: Optional[TabDiffusionGenerator] = None
        self.train_history = {"train_loss": [], "val_loss": []}
        self.checkpoint_path = None

    def _prepare(self):
        """Fit encoders and scaler, label-encode categoricals in a copy of dataframe."""
        df = self.df_raw.copy()

        # fill missing
        df[self.num_cols] = df[self.num_cols].fillna(0.0)
        df[self.cat_cols] = df[self.cat_cols].fillna("NA")

        # fit label encoders for categorical columns
        for c in self.cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str).values)
            self.label_encoders[c] = le

        # fit scaler for numerics (Robust to outliers)
        if len(self.num_cols) > 0:
            self.scaler = RobustScaler()
            self.scaler.fit(df[self.num_cols].values)

        # compute empirical conditional distributions
        for col in self.cond_cols:
            if col in self.cat_cols:
                counts = df[col].value_counts(normalize=True)
                # index contains integer class labels (already encoded)
                self.cond_empirical_probs[col] = counts
            elif col in self.num_cols:
                self.cond_empirical_stats[col] = {"mean": float(df[col].mean()), "std": float(df[col].std())}

        self.df_prepared = df

    def _make_dataset_loaders(self, batch_size: int = 128, val_split: float = 0.2, balance: bool = True, random_state: int = 42):
        df = self.df_prepared.reset_index(drop=True)
        train_df, val_df = train_test_split(df, test_size=val_split, random_state=random_state, stratify=(df[self.target] if self.target in df.columns else None))

        train_ds = TabularDataset(train_df, self.num_cols, self.cat_cols, label_col=self.target, scaler=self.scaler)
        val_ds = TabularDataset(val_df, self.num_cols, self.cat_cols, label_col=self.target, scaler=self.scaler)

        if balance and (self.target is not None) and (self.target in df.columns):
            labels = train_df[self.target]
            class_counts = labels.value_counts().to_dict()
            weights = labels.map(lambda x: 1.0 / class_counts[x]).values
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def fit(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-4,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        transformer_ff: int = 512,
        token_dim: int = 192,
        uncond_prob: float = 0.1,
        patience: int = 10,
        checkpoint_path: str = "tabdiff_best.pt",
        balance: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Fit the TabDiffusion model on the prepared dataframe.
        Returns: train_losses, val_losses (lists)
        """
        self._prepare()
        train_loader, val_loader = self._make_dataset_loaders(batch_size=batch_size, val_split=0.2, balance=balance)

        # prepare model
        cat_cardinalities = [len(self.label_encoders[c].classes_) for c in self.cat_cols]
        # cond_columns format: {name: {"type": "cat"/"num"/"binary", "cardinality": int (if cat)}}
        cond_specs = {}
        for col in self.cond_cols:
            if col in self.cat_cols:
                cond_specs[col] = {"type": "cat", "cardinality": len(self.label_encoders[col].classes_)}
            elif col in self.num_cols:
                cond_specs[col] = {"type": "num"}

        device = torch.device(self.device)
        self.model = TabDiffusionGenerator(
            num_numeric=len(self.num_cols),
            cat_cardinalities=cat_cardinalities,
            cond_columns=cond_specs,
            token_dim=token_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ff=transformer_ff,
            uncond_prob=uncond_prob
        ).to(device)

        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-6)

        best_val = float("inf")
        patience_counter = 0
        self.checkpoint_path = checkpoint_path

        self.train_history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            # training
            self.model.train()
            train_losses = []
            for x_num, x_cat, y in train_loader:
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                # build cond_batch from batch values (for conditionals that are categorical numeric in cat_cols order)
                cond_batch = {}
                for col in self.cond_cols:
                    if col in self.cat_cols:
                        idx = self.cat_cols.index(col)
                        cond_batch[col] = x_cat[:, idx].to(device)
                    elif col in self.num_cols:
                        idx = self.num_cols.index(col)
                        cond_batch[col] = x_num[:, idx].to(device)

                loss = self.model.training_loss(x_num, x_cat, cond_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_losses.append(loss.item())

            avg_train = float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0
            self.train_history["train_loss"].append(avg_train)

            # validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for x_num, x_cat, y in val_loader:
                    x_num = x_num.to(device)
                    x_cat = x_cat.to(device)
                    cond_batch = {}
                    for col in self.cond_cols:
                        if col in self.cat_cols:
                            idx = self.cat_cols.index(col)
                            cond_batch[col] = x_cat[:, idx].to(device)
                        elif col in self.num_cols:
                            idx = self.num_cols.index(col)
                            cond_batch[col] = x_num[:, idx].to(device)
                    vloss = self.model.training_loss(x_num, x_cat, cond_batch)
                    val_losses.append(vloss.item())
            avg_val = float(np.mean(val_losses)) if len(val_losses) > 0 else 0.0
            self.train_history["val_loss"].append(avg_val)

            if self.verbose:
                print(f"[Epoch {epoch}/{epochs}] train_loss={avg_train:.6f} val_loss={avg_val:.6f}")

            scheduler.step()

            # checkpoint
            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                if self.verbose:
                    print("Saved new best checkpoint:", checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if self.verbose:
                    print("Early stopping triggered.")
                break

        # load best model automatically
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=device))
            if self.verbose:
                print("Loaded best model from", self.checkpoint_path)
        return self.train_history["train_loss"], self.train_history["val_loss"]

    def sample(
        self,
        num_samples: int = 100,
        labels_to_sample: Optional[List] = None,
        proportions: Optional[List[float]] = None,
        cond_overrides: Optional[Dict] = None,
        steps: int = 50,
        cfg_scale: float = 1.5,
        sampling_kwargs: Optional[Dict] = None,
        device: Optional[str] = None
    ) -> pd.DataFrame:
        """
        High-level sample function that returns a pandas DataFrame with original column names (decoded).
        - labels_to_sample: list of labels for target to sample (if target provided)
        - proportions: list of proportions corresponding to labels_to_sample (sums to 1.0)
        - cond_overrides: dict of additional conditioning overrides
        """
        if self.model is None:
            raise RuntimeError("Model not trained - call fit() first.")

        device = device or self.device
        dev = torch.device(device)

        sampling_kwargs = sampling_kwargs or {}
        cond_overrides = cond_overrides or {}

        # build per-label counts
        if self.target is not None and labels_to_sample is not None:
            if proportions is None:
                # even split
                proportions = [1.0 / len(labels_to_sample)] * len(labels_to_sample)
            counts = [int(round(p * num_samples)) for p in proportions]
            # adjust to sum
            diff = num_samples - sum(counts)
            if diff != 0:
                counts[0] += diff
            parts = []
            for label_val, cnt in zip(labels_to_sample, counts):
                # build cond_batch using overrides + setting target to label_val
                local_overrides = dict(cond_overrides)
                local_overrides[self.target] = int(label_val)
                cond_batch = build_cond_batch_from_overrides(
                    cond_specs=self.model.cond_specs,
                    num_samples=cnt,
                    cond_overrides=local_overrides,
                    empirical_cond_probs=self.cond_empirical_probs,
                    label_encoders=self.label_encoders,
                    device=device
                )
                # move cond tensors to device
                cond_batch = {k: v.to(dev) for k, v in cond_batch.items()}
                x_num_gen, x_cat_gen = self.model.sample(cnt, cond_batch, steps=steps, cfg_scale=cfg_scale, sampling_kwargs=sampling_kwargs)
                x_num_np = x_num_gen.detach().cpu().numpy() if x_num_gen.numel() > 0 else np.zeros((cnt, 0))
                x_cat_np = x_cat_gen.detach().cpu().numpy() if x_cat_gen.numel() > 0 else np.zeros((cnt, 0), dtype=int)
                df_part = decode_generated_to_df(x_num_np, x_cat_np, self.num_cols, self.cat_cols, scaler=self.scaler, label_encoders=self.label_encoders)
                # add target column
                if self.target is not None:
                    df_part[self.target] = int(label_val)
                parts.append(df_part)
            out_df = pd.concat(parts, axis=0, ignore_index=True)
        else:
            # unconditional or cond_overrides only
            cond_batch = build_cond_batch_from_overrides(
                cond_specs=self.model.cond_specs,
                num_samples=num_samples,
                cond_overrides=cond_overrides,
                empirical_cond_probs=self.cond_empirical_probs,
                label_encoders=self.label_encoders,
                device=device
            )
            cond_batch = {k: v.to(dev) for k, v in cond_batch.items()}
            x_num_gen, x_cat_gen = self.model.sample(num_samples, cond_batch, steps=steps, cfg_scale=cfg_scale, sampling_kwargs=sampling_kwargs)
            x_num_np = x_num_gen.detach().cpu().numpy() if x_num_gen.numel() > 0 else np.zeros((num_samples, 0))
            x_cat_np = x_cat_gen.detach().cpu().numpy() if x_cat_gen.numel() > 0 else np.zeros((num_samples, 0), dtype=int)
            out_df = decode_generated_to_df(x_num_np, x_cat_np, self.num_cols, self.cat_cols, scaler=self.scaler, label_encoders=self.label_encoders)

        # Final sanity: clip numeric outputs to training percentiles to avoid extreme outliers
        for col in self.num_cols:
            if col in out_df.columns and col in self.df_prepared.columns:
                lo, hi = self.df_prepared[col].quantile([0.001, 0.999]).values
                out_df[col] = out_df[col].clip(lo, hi)

        return out_df
