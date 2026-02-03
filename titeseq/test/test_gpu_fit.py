import numpy as np
import torch
import pytest

from titeseq.gpu_backend import fit_gpu
from titeseq.utils import prepare_gpu_tensors

def hill_function(c, kd, a, b):
    return a * (c / (c + kd)) + b

def test_prepare_gpu_tensors_direct_and_binned():
    import pandas as pd

    concs = [0.1, 1.0]

    # Direct columns case: 'conc_{c}'
    df_direct = pd.DataFrame({
        'sequence': ['s1', 's2'],
        'conc_0.1': [1.0, 2.0],
        'conc_1.0': [3.0, 4.0],
    })
    t_direct = prepare_gpu_tensors(df_direct, None, concs)
    assert t_direct.shape == (2, 2)
    expected_direct = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float)
    assert np.allclose(t_direct.numpy(), expected_direct)

    # Binned counts case: 'c_{c}_b{b}'
    df_binned = pd.DataFrame({
        'sequence': ['s1'],
        'c_0.1_b0': [10],
        'c_0.1_b1': [5],
        'c_1.0_b0': [2],
        'c_1.0_b1': [3],
    })
    t_binned = prepare_gpu_tensors(df_binned, None, concs)
    # total counts per conc: [15, 5] -> stored as log(total + 1)
    expected_binned = np.log(np.array([[15.0, 5.0]]) + 1.0)
    assert t_binned.shape == (1, 2)
    assert np.allclose(t_binned.numpy(), expected_binned)

@pytest.mark.parametrize("device", ["cpu"])
def test_fit_gpu_batch_consistency_cpu(device):
    # Ensure reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    # Concentrations spanning several decades
    concs = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5], dtype=float)
    b_bg = 2.0

    n_variants = 10
    # Sample true parameters over ranges
    true_kds = 10 ** np.random.uniform(-9, -6, size=n_variants)
    true_as = np.random.uniform(5.0, 100.0, size=n_variants)

    # Build observed responses and add small noise
    obs = np.zeros((n_variants, len(concs)), dtype=float)
    for i in range(n_variants):
        obs[i] = hill_function(concs, true_kds[i], true_as[i], b_bg)
        obs[i] += np.random.normal(scale=0.01 * true_as[i], size=obs.shape[1])

    # Keep values positive (safety for log)
    obs = np.clip(obs, 1e-3, None)

    # Fit with a large batch (single batch)
    kd_large, a_large = fit_gpu(obs, concs, b_bg,
                                batch_size=1024, max_epochs=500, tol=1e-4, lr=0.05, device=device)

    # Fit with small batch size (multiple batches)
    kd_small, a_small = fit_gpu(obs, concs, b_bg,
                                batch_size=3, max_epochs=500, tol=1e-4, lr=0.05, device=device)

    # Basic sanity checks
    assert kd_large.shape[0] == n_variants
    assert a_large.shape[0] == n_variants
    assert kd_small.shape[0] == n_variants
    assert a_small.shape[0] == n_variants

    assert np.all(np.isfinite(kd_large)) and np.all(kd_large > 0)
    assert np.all(np.isfinite(a_large)) and np.all(a_large > 0)

    # The two runs (different batching) should give similar results on average
    # Compare mean absolute relative difference normalized by mean magnitude
    kd_mean = (np.mean(kd_large) + 1e-12)
    a_mean = (np.mean(a_large) + 1e-12)

    kd_rel_diff = np.mean(np.abs(kd_large - kd_small)) / kd_mean
    a_rel_diff = np.mean(np.abs(a_large - a_small)) / a_mean

    # Acceptable relative differences; these thresholds are intentionally generous
    assert kd_rel_diff < 0.2, f"kd_rel_diff too large: {kd_rel_diff:.3f}"
    assert a_rel_diff < 0.2, f"a_rel_diff too large: {a_rel_diff:.3f}"

    # Check that fitted parameters produce reasonably low error compared to true (noisy) observations
    def preds_from_params(kd_vals, a_vals):
        preds = np.zeros_like(obs)
        for i in range(n_variants):
            preds[i] = hill_function(concs, kd_vals[i], a_vals[i], b_bg)
        return preds

    preds = preds_from_params(kd_large, a_large)
    # Mean relative absolute error between preds and noisy obs
    mean_rel_err = np.mean(np.abs(preds - obs) / (np.abs(obs) + 1e-12))
    assert mean_rel_err < 0.5, f"mean_rel_err too large: {mean_rel_err:.3f}"