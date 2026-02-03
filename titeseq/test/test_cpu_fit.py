import os
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from titeseq.cpu_backend import fit_cpu
from titeseq.cli import main as cli_main
from pathlib import Path

def hill_function(c, kd, a, b):
    return a * (c / (c + kd)) + b

def test_fit_cpu_single_variant():
    # concentrations
    concs = np.array([1e-9, 1e-8, 1e-7, 1e-6], dtype=float)
    b_bg = 2.0

    # true parameters
    true_kd = 5e-8
    true_a = 25.0

    # generate noise-free observations and convert to log-space as expected by fit_cpu
    obs = hill_function(concs, true_kd, true_a, b_bg)
    mean_log_fluors = np.log(obs + 1e-12)

    kd_est, a_est = fit_cpu(concs, mean_log_fluors, b_bg)

    # sanity checks: positive, finite
    assert np.isfinite(kd_est) and kd_est > 0
    assert np.isfinite(a_est) and a_est > 0

    # relative error checks (loose tolerances)
    assert abs(np.log10(kd_est) - np.log10(true_kd)) < 1.0  # within 1 order of magnitude
    assert abs(a_est - true_a) / true_a < 0.5              # within 50%

def test_cli_cpu_end_to_end(tmp_path):
    """
    Create small synthetic counts and samples CSVs, run the CLI in CPU mode,
    and verify the produced output CSV contains kd and expression_a.
    """

    # Setup synthetic data for two sequences
    concs = [1e-9, 1e-8, 1e-7]
    b_bg = 1.5

    true_params = [
        {'sequence': 'seq1', 'kd': 5e-8, 'a': 20.0},
        {'sequence': 'seq2', 'kd': 2e-9, 'a': 40.0},
    ]

    # Build counts (here we store mean_log_fluorescence directly in conc_{c} columns,
    # because the CLI's CPU path passes those values into fit_cpu as mean_log_fluors)
    counts_rows = []
    for p in true_params:
        preds = hill_function(np.array(concs), p['kd'], p['a'], b_bg)
        row = {'sequence': p['sequence']}
        for c, val in zip(concs, preds):
            col = f'conc_{c}'
            row[col] = np.log(val + 1e-12)  # store in log-space to match CPU expectation
        counts_rows.append(row)
    counts_df = pd.DataFrame(counts_rows)

    # samples.csv must include a row for concentration 0 to compute b_bg
    samples_df = pd.DataFrame({
        'concentration': [0.0] + concs,
        'mean_log_fluorescence': [b_bg] + [np.nan for _ in concs]
    })

    counts_path = tmp_path / "counts.csv"
    samples_path = tmp_path / "samples.csv"
    out_path = tmp_path / "out.csv"

    counts_df.to_csv(counts_path, index=False)
    samples_df.to_csv(samples_path, index=False)

    runner = CliRunner()
    result = runner.invoke(cli_main, [
        '--counts', str(counts_path),
        '--samples', str(samples_path),
        '--mode', 'cpu',
        '--output', str(out_path)
    ])

    # CLI should exit successfully
    assert result.exit_code == 0, result.output

    # Output file should exist and contain kd/expression_a columns
    assert out_path.exists()
    out_df = pd.read_csv(out_path)
    assert 'kd' in out_df.columns
    assert 'expression_a' in out_df.columns

    # basic value checks: finite & positive
    assert np.all(np.isfinite(out_df['kd'].values))
    assert np.all(out_df['kd'].values > 0)
    assert np.all(np.isfinite(out_df['expression_a'].values))
    assert np.all(out_df['expression_a'].values > 0)