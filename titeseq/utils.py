import pandas as pd
import numpy as np
import torch

def load_and_preprocess(counts_path, samples_path):
    """Loads and synchronizes the count table and sample metadata."""
    counts = pd.read_csv(counts_path)
    samples = pd.read_csv(samples_path)
    
    # Calculate Background B from 0M antigen [cite: 961]
    b_bg = samples[samples['concentration'] == 0]['mean_log_fluorescence'].mean()
    
    # Get unique concentrations sorted [cite: 232]
    concs = sorted(samples['concentration'].unique())
    
    return counts, samples, concs, b_bg

def prepare_gpu_tensors(counts_df, samples_df, concs):
    """Converts dataframes into PyTorch tensors for batch MLE."""
    num_variants = len(counts_df)
    num_concs = len(concs)
    
    # Create an empty tensor for counts: (variants, concentrations, bins)
    # This assumes 4 bins per concentration as per the paper [cite: 232]
    tensor_data = torch.zeros((num_variants, num_concs, 4))
    
    for i, c in enumerate(concs):
        for b in range(4):
            col_name = f"c_{c}_b{b}" # Format must match your CSV headers
            if col_name in counts_df.columns:
                tensor_data[:, i, b] = torch.tensor(counts_df[col_name].values)
                
    return tensor_data