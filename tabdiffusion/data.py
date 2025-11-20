# tabdiffusion/data.py
"""
Data handling: detect types, fit encoders/scalers, provide DataLoader-friendly dataset,
and inverse transform utils.
"""


from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    Lightweight dataset that returns (x_num, x_cat, y, misc...) for integration with DataLoader.

    Args:
        df: pandas DataFrame (preprocessed so that categorical columns are label-encoded integers)
        num_cols: list of numeric column names
        cat_cols: list of categorical column names (already label encoded)
        label_col: optional target column name (if available)
        scaler: optional pre-fitted scaler used to transform numerics (expects numpy-transform signature)
    """
    def __init__(self, df, num_cols: List[str], cat_cols: List[str], label_col: Optional[str] = None, scaler=None):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.num_cols:
            num_vals = row[self.num_cols].values.astype(float).reshape(1, -1)
            if self.scaler is not None:
                num_vals = self.scaler.transform(num_vals).squeeze(0)
            x_num = torch.tensor(num_vals, dtype=torch.float32)
        else:
            x_num = torch.zeros(0, dtype=torch.float32)
        if self.cat_cols:
            x_cat = torch.tensor(row[self.cat_cols].astype(int).values, dtype=torch.long)
        else:
            x_cat = torch.zeros(0, dtype=torch.long)
        y = torch.tensor(int(row[self.label_col]) if (self.label_col is not None and self.label_col in row) else 0, dtype=torch.long)
        return x_num, x_cat, y
