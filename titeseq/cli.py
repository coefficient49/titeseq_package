import click
import pandas as pd
import numpy as np
import torch
from .cpu_backend import fit_cpu
from .gpu_backend import fit_gpu

@click.command()
@click.option('--counts', required=True, help='Path to count table.')
@click.option('--samples', required=True, help='Path to sample sheet.')
@click.option('--mode', type=click.Choice(['cpu', 'gpu']), default='cpu', help='Inference engine.')
@click.option('--output', default='titeseq_results.csv')
def main(counts, samples, mode, output):
    counts_df = pd.read_csv(counts)
    samples_df = pd.read_csv(samples)
    
    # Pre-processing [cite: 961, 982]
    concs = sorted(samples_df['concentration'].unique())
    b_bg = samples_df[samples_df['concentration'] == 0]['mean_log_fluorescence'].mean()
    
    if mode == 'gpu' and torch.cuda.is_available():
        click.echo("Running GPU-accelerated Poisson MLE...")
        # Prepare counts as a large tensor (variants x concentrations)
        # Note: This requires mapping counts to concentrations based on sample sheet
        counts_array = counts_df.drop(columns=['sequence']).values
        kd_results, a_results = fit_gpu(torch.tensor(counts_array), concs, b_bg)
        
        counts_df['kd'] = kd_results
        counts_df['expression_a'] = a_results
    else:
        click.echo("Running CPU-based Scipy least-squares...")
        results = []
        for _, row in counts_df.iterrows():
            # Weighted average logic [cite: 966]
            y_obs = [row[f'conc_{c}'] for c in concs] # Simplified mapping
            kd, a = fit_cpu(np.array(concs), np.array(y_obs), b_bg)
            results.append({'sequence': row['sequence'], 'kd': kd, 'expression_a': a})
        counts_df = pd.DataFrame(results)

    counts_df.to_csv(output, index=False)
    click.echo(f"Finished. Results saved to {output}")

if __name__ == '__main__':
    main()