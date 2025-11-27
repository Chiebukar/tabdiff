# tabdiffusion/core.py
"""
High-level TabDiffusion wrapper.

Usage example:
    td = TabDiffusion(df, target='isFraud', conditionals=['ProductCD', 'DeviceType'])
    td.fit(epochs=10, batch_size=256)
    samples = td.sample(num_samples=1000, labels_to_sample=[1,0], proportions=[0.5,0.5])
"""

import os
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
        - conditionals: list of column names to use as conditioning (categorical or numeric). 
                        If None, default uses all categorical columns.
        """
        self.df_raw = sanitize_categoricals(df.copy())
        self.target = target
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # detect numeric vs categorical
        self.num_cols = df.select_dtypes(include=["int64", "float64", "float32"]).columns.tolist()
        if target in self.num_cols:
            self.num_cols.remove(target)

        self.cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # conditionals
        self.cond_cols = conditionals if conditionals else list(self.cat_cols)

        # containers
        self.scaler = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.cond_empirical_probs = {}   # categorical distributions
        self.cond_empirical_stats = {}   # numeric mean/std

        self.model = None
        self.checkpoint_path = None

        # <-- UPDATED: tracked losses -->
        self.train_losses = []
        self.val_losses = []
        self.history = {}
        self.best_val_loss = None
    # ---------------------------------------------------------------


    def _prepare(self):
        """Fit encoders + scaler and produce encoded training frame."""
        df = self.df_raw.copy()

        df[self.num_cols] = df[self.num_cols].fillna(0.0)
        df[self.cat_cols] = df[self.cat_cols].fillna("NA")

        for c in self.cat_cols:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            self.label_encoders[c] = le

        if len(self.num_cols) > 0:
            self.scaler = RobustScaler()
            self.scaler.fit(df[self.num_cols])

        # compute empirical conditional priors
        for col in self.cond_cols:
            if col in self.cat_cols:
                self.cond_empirical_probs[col] = df[col].value_counts(normalize=True)
            elif col in self.num_cols:
                self.cond_empirical_stats[col] = {"mean": df[col].mean(), "std": df[col].std()}

        self.df_prepared = df


    def _make_dataset_loaders(self, batch_size=128, val_split=0.2, balance=True, seed=42):
        df = self.df_prepared.reset_index(drop=True)

        y = df[self.target] if self.target in df.columns else None
        train_df, val_df = train_test_split(df, test_size=val_split,
                        stratify=y if y is not None else None, random_state=seed)

        train_ds = TabularDataset(train_df, self.num_cols, self.cat_cols, label_col=self.target, scaler=self.scaler)
        val_ds   = TabularDataset(val_df,   self.num_cols, self.cat_cols, label_col=self.target, scaler=self.scaler)

        if balance and self.target in df.columns:
            labels = train_df[self.target]
            weights = 1 / labels.value_counts()[labels].values
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        return train_loader, DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    def fit(
        self,
        epochs=50,
        batch_size=256,
        lr=1e-4,
        transformer_layers=4,
        transformer_heads=4,
        transformer_ff=512,
        token_dim=192,
        uncond_prob=0.1,
        patience=10,
        checkpoint_path="tabdiff_best.pt",
        balance=True
    ):
        self._prepare()
        train_loader, val_loader = self._make_dataset_loaders(batch_size, balance=balance)

        # model setup
        cat_cardinalities = [len(self.label_encoders[c].classes_) for c in self.cat_cols]

        cond_specs = {}
        for c in self.cond_cols:
            if c in self.cat_cols:
                cond_specs[c] = {"type": "cat", "cardinality": len(self.label_encoders[c].classes_)}
            else:
                cond_specs[c] = {"type": "num"}

        dev = torch.device(self.device)
        self.model = TabDiffusionGenerator(
            num_numeric=len(self.num_cols),
            cat_cardinalities=cat_cardinalities,
            cond_columns=cond_specs,
            token_dim=token_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ff=transformer_ff,
            uncond_prob=uncond_prob
        ).to(dev)

        opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

        best_val = np.inf
        patience_ctr = 0
        self.checkpoint_path = checkpoint_path

        self.train_losses, self.val_losses = [], []  # <-- RESET cleanly

        # ------------------ Training Loop ------------------
        for ep in range(1, epochs + 1):

            self.model.train()
            tr_loss = []
            for x_num, x_cat, y in train_loader:
                x_num, x_cat = x_num.to(dev), x_cat.to(dev)

                cond = {}
                for col in self.cond_cols:
                    cond[col] = x_cat[:, self.cat_cols.index(col)].to(dev) if col in self.cat_cols \
                                else x_num[:, self.num_cols.index(col)].to(dev)

                loss = self.model.training_loss(x_num, x_cat, cond)
                opt.zero_grad(); loss.backward(); opt.step()
                tr_loss.append(loss.item())

            avg_train = np.mean(tr_loss)
            self.train_losses.append(avg_train)

            # ------------------ Validation ------------------
            self.model.eval()
            vloss = []
            with torch.no_grad():
                for x_num, x_cat, y in val_loader:
                    x_num, x_cat = x_num.to(dev), x_cat.to(dev)

                    cond = {}
                    for col in self.cond_cols:
                        cond[col] = x_cat[:, self.cat_cols.index(col)].to(dev) if col in self.cat_cols \
                                    else x_num[:, self.num_cols.index(col)].to(dev)

                    vloss.append(self.model.training_loss(x_num, x_cat, cond).item())

            avg_val = np.mean(vloss)
            self.val_losses.append(avg_val)

            if self.verbose:
                print(f"[{ep}/{epochs}] Train={avg_train:.6f} | Val={avg_val:.6f}")

            sched.step()

            # Best checkpoint
            if avg_val < best_val:
                best_val = avg_val
                patience_ctr = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                if self.verbose: print("↳ Saved new best checkpoint.")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print("Early Stopping.")
                    break

        self.best_val_loss = best_val
        self.history = {"train_loss": self.train_losses, "val_loss": self.val_losses}

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
        print("✓ Loaded Best Model")

        return self.train_losses, self.val_losses


    def sample(
        self,
        num_samples=100,
        labels_to_sample=None,
        proportions=None,
        cond_overrides=None,
        steps=50,
        cfg_scale=1.5,
        sampling_kwargs=None,
        device=None
    ) -> pd.DataFrame:

        if self.model is None:
            raise RuntimeError("Model not trained — call fit() first.")

        device = torch.device(device or self.device)

        cond_overrides = cond_overrides or {}
        sampling_kwargs = sampling_kwargs or {}

        # With label-based sampling
        if self.target and labels_to_sample:
            proportions = proportions or [1/len(labels_to_sample)]*len(labels_to_sample)
            counts = [round(p*num_samples) for p in proportions]
            diff = num_samples - sum(counts); counts[0]+=diff

            dfs=[]
            for lbl, cnt in zip(labels_to_sample, counts):
                local = dict(cond_overrides); local[self.target]=int(lbl)

                cond = build_cond_batch_from_overrides(
                    self.model.cond_specs, cnt, local,
                    self.cond_empirical_probs, self.label_encoders, device
                )

                xN,xC = self.model.sample(cnt,cond,steps,cfg_scale,sampling_kwargs)
                dfs.append(
                    decode_generated_to_df(xN.cpu().numpy(), xC.cpu().numpy(),
                                           self.num_cols,self.cat_cols,
                                           scaler=self.scaler,label_encoders=self.label_encoders)
                )
            df = pd.concat(dfs).reset_index(drop=True)

        else: # unconditional
            cond = build_cond_batch_from_overrides(
                self.model.cond_specs,num_samples,cond_overrides,
                self.cond_empirical_probs,self.label_encoders,device
            )
            xN,xC = self.model.sample(num_samples,cond,steps,cfg_scale,sampling_kwargs)

            df = decode_generated_to_df(
                xN.cpu().numpy(), xC.cpu().numpy(),
                self.num_cols,self.cat_cols,
                scaler=self.scaler,label_encoders=self.label_encoders
            )

        # limit extreme outliers for numeric stability
        for col in self.num_cols:
            lo,hi = self.df_prepared[col].quantile([0.001,0.999])
            df[col]=df[col].clamp(lo,hi)

        return df

