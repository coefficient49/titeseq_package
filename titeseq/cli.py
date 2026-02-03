import click
import pandas as pd
import numpy as np
import torch
from .cpu_backend import fit_cpu
from .gpu_backend import fit_gpu
from .utils import load_and_preprocess, prepare_gpu_tensors

@click.command()
@click.option('--counts', required=True, help='Path to count table.')
@click.option('--samples', required=True, help='Path to sample sheet.')
@click.option('--mode', type=click.Choice(['cpu', 'gpu']), default='cpu', help='Inference engine.')
@click.option('--output', default='titeseq_results.csv')
def main(counts, samples, mode, output):
    # Use the shared load_and_preprocess helper so background/concentrations are computed consistently
    counts_df, samples_df, concs, b_bg = load_and_preprocess(counts, samples)

    if mode == 'gpu' and torch.cuda.is_available():
        click.echo("Running GPU-accelerated fit (batched)...")
        # Prepare counts as a 2-D tensor (variants x concentrations) for the GPU fitter
        counts_tensor = prepare_gpu_tensors(counts_df, samples_df, concs)  # shape: (variants, num_concs)
        kd_results, a_results = fit_gpu(counts_tensor, concs, b_bg, batch_size=1024, max_epochs=1500, tol=1e-6, lr=0.05)

        # Attach results to the original counts_df (assumes same order)
        counts_df['kd'] = kd_results
        counts_df['expression_a'] = a_results
    else:
        if mode == 'gpu':
            click.echo("CUDA not available — falling back to CPU.")
        else:
            click.echo("Running CPU-based Scipy least-squares...")

        results = []
        for _, row in counts_df.iterrows():
            # Map observed columns to concentrations (this assumes your counts_df layout matches prepare_gpu_tensors logic)
            y_obs = [row.get(f'conc_{c}', row.get(f'c_{c}', np.nan)) for c in concs]
            y_obs = np.array(y_obs, dtype=float)
            # If any NaNs, fallback to zeros or skip — here we replace NaNs with small values:
            y_obs = np.nan_to_num(y_obs, nan=1e-3)
            kd, a = fit_cpu(np.array(concs), y_obs, b_bg)
            results.append({'sequence': row.get('sequence', ''), 'kd': kd, 'expression_a': a})
        counts_df = pd.DataFrame(results)

    counts_df.to_csv(output, index=False)
    click.echo(f"Finished. Results saved to {output}")

if __name__ == '__main__':
    main()