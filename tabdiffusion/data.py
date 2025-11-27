# tabdiffusion/data.py
"""
Data handling for TabDiffusion:
 - Detect numeric & categorical data
 - Apply scaling safely (no sklearn warning)
 - Store encoded + scaled values efficiently
 - Return tensors ready for the diffusion model
"""

from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    Dataset returned to the DataLoader

    Returns:
        x_num → float32 tensor  [num_features]
        x_cat → long tensor     [cat_features]
        y     → long tensor     (classification target)
    """

    def __init__(
        self,
        df,
        num_cols: List[str],
        cat_cols: List[str],
        label_col: Optional[str] = None,
        scaler=None
    ):
        self.df = df.reset_index(drop=True)

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.scaler = scaler

        # Pre–extract & convert all numeric values (10x faster than per row transform)
        if len(num_cols) > 0:
            X = df[num_cols].to_numpy(dtype=float)

            # 🔥 Safe scaling — never produces sklearn warnings
            if scaler is not None:
                X = scaler.transform(X)

            self.X_num = torch.tensor(X, dtype=torch.float32)
        else:
            self.X_num = torch.zeros((len(df), 0), dtype=torch.float32)

        # Store categorical tensors once — **no repeated extraction**
        if len(cat_cols) > 0:
            self.X_cat = torch.tensor(df[cat_cols].astype(int).values, dtype=torch.long)
        else:
            self.X_cat = torch.zeros((len(df), 0), dtype=torch.long)

        # Store label tensor (optional)
        if label_col is not None and label_col in df.columns:
            self.y = torch.tensor(df[label_col].astype(int).values, dtype=torch.long)
        else:
            self.y = torch.zeros(len(df), dtype=torch.long)

    # ----------------------------------------------------------

    def __len__(self):
        return len(self.df)

    # ----------------------------------------------------------

    def __getitem__(self, idx):
        # Return tensors already prepared — **no per-item scaling anymore**
        return (
            self.X_num[idx],    # float32
            self.X_cat[idx],    # long
            self.y[idx]         # long
        )
