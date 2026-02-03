import pandas as pd
import numpy as np
import torch
from typing import Tuple, List

def load_and_preprocess(counts_path, samples_path):
    """Loads and synchronizes the count table and sample metadata."""
    counts = pd.read_csv(counts_path)
    samples = pd.read_csv(samples_path)

    # Calculate Background B from 0M antigen [cite: 961]
    b_bg = samples[samples['concentration'] == 0]['mean_log_fluorescence'].mean()

    # Get unique concentrations sorted [cite: 232]
    concs = sorted(samples['concentration'].unique())

    return counts, samples, concs, b_bg


def prepare_gpu_tensors(counts_df: pd.DataFrame, samples_df: pd.DataFrame, concs: List[float]) -> torch.Tensor:
    """
    Prepare a 2-D tensor (variants x concentrations) of observed responses suitable for fit_gpu.

    This function tries to be flexible:
    - If your counts_df already contains direct observed values per concentration named like
      'conc_{c}' or 'c_{c}', those will be used.
    - If your table contains binned counts per concentration using the pattern 'c_{c}_b{b}',
      the function will compute a simple proxy observed value by summing over bins
      and taking log(total_counts + 1). If you have true fluorescence per bin values,
      replace this mapping with a weighted-average using those fluorescence values.
    """

    num_variants = len(counts_df)
    num_concs = len(concs)
    obs = np.zeros((num_variants, num_concs), dtype=float)

    # Try direct columns first: 'conc_{c}' or 'c_{c}'
    direct_cols_found = True
    for i, c in enumerate(concs):
        col_candidates = [f"conc_{c}", f"c_{c}", f"conc_{str(c)}", f"c_{str(c)}"]
        found = None
        for col in col_candidates:
            if col in counts_df.columns:
                found = col
                break
        if found is None:
            direct_cols_found = False
            break

    if direct_cols_found:
        # Use the direct columns
        for i, c in enumerate(concs):
            col = next(col for col in [f"conc_{c}", f"c_{c}", f"conc_{str(c)}", f"c_{str(c)}"] if col in counts_df.columns)
            obs[:, i] = counts_df[col].values
        return torch.tensor(obs, dtype=torch.float32)

    # Otherwise, try binned counts pattern 'c_{c}_b{b}' (bins e.g., b0..b3)
    # We'll sum counts across bins and use log(total_counts + 1) as a proxy observed response.
    # If you have actual fluorescence per bin levels, replace this with a weighted average.
    bins = None
    # detect number of bins by scanning for 'c_{c}_b' columns on first concentration
    for b in range(0, 20):  # assume less than 20 bins
        col = f"c_{concs[0]}_b{b}"
        if col in counts_df.columns:
            if bins is None:
                bins = []
            bins.append(b)
        else:
            if bins is not None:
                break

    if bins is not None and len(bins) > 0:
        for i, c in enumerate(concs):
            total_counts = np.zeros(num_variants, dtype=float)
            # sum bins if present
            for b in bins:
                col = f"c_{c}_b{b}"
                if col in counts_df.columns:
                    total_counts += counts_df[col].values
            # use log(total_counts + 1) so values are well-scaled for the log-space objective
            obs[:, i] = np.log(total_counts + 1.0)
        return torch.tensor(obs, dtype=torch.float32)

    # As a last resort, try to find columns with concentration suffixes using contains
    # e.g. columns that end with '_{conc}' or similar — fallback: look for any column with the concentration string
    for i, c in enumerate(concs):
        matches = [col for col in counts_df.columns if str(c) in col]
        if matches:
            obs[:, i] = counts_df[matches[0]].values
        else:
            # if still missing, fill with small positive baseline
            obs[:, i] = np.full(num_variants, 1e-3)

    return torch.tensor(obs, dtype=torch.float32)