import pandas as pd
import numpy as np
import random

def generate_seeded_titeseq_data():
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # 1. Define Experimental Parameters (Appendix 2 & 4)
    # 11 concentrations used in the study
    concs = [0] + [10**i for i in np.arange(-9.5, -4.5, 0.5)]
    bins = [0, 1, 2, 3] # Four sorting bins
    b_bg = 1.2 # Simulated background log-fluorescence

    # 2. Generate Sample Sheet (Metadata)
    samples_data = []
    for c in concs:
        for b in bins:
            # Higher bins represent higher log-fluorescence gates
            f_bc = (b + 1) * 1.5 if c > 0 else b_bg
            samples_data.append({
                'sample_id': f"C_{c}_B{b}",
                'concentration': c,
                'bin_id': b,
                'mean_log_fluorescence': f_bc,
                'cells_sorted': 50000 # Typical sort depth
            })
    pd.DataFrame(samples_data).to_csv('test_samples.csv', index=False)

    # 3. Generate 100 Rows of Count Data
    variants = []
    for i in range(100):
        # Assign "Ground Truth" values
        if i < 10: # High-affinity "OPT-like"
            true_kd = 10**-9.5
            true_a = 8000
        elif i < 30: # Wild-Type-like
            true_kd = 10**-8.9
            true_a = 5000
        else: # Mutants with weakened affinity
            true_kd = 10**np.random.uniform(-7.5, -4.5)
            true_a = np.random.uniform(500, 3000)

        row = {'sequence': f"SEQ_{i:03d}", 'true_kd': true_kd, 'true_a': true_a}
        
        for c in concs:
            # Hill Function (Equation 1 / A1)
            f_pred = true_a * (c / (c + true_kd)) + b_bg
            
            # Distribute counts using a Gaussian profile across bins
            # Centers the "peak" read count on the bin closest to f_pred
            for b in bins:
                center = (f_pred / 2000) # Mapping fluorescence to bin index
                prob = np.exp(-((b - center)**2) / 0.8)
                # Add Poisson noise to the reads
                base_count = int(prob * 1000)
                row[f"c_{c}_b{b}"] = np.random.poisson(base_count)
                
        variants.append(row)

    df = pd.DataFrame(variants)
    # Save ground truth separately for validation, then drop for CLI input
    df[['sequence', 'true_kd', 'true_a']].to_csv('ground_truth.csv', index=False)
    df.drop(columns=['true_kd', 'true_a']).to_csv('test_counts.csv', index=False)
    print("Seeded test data generated: test_counts.csv, test_samples.csv, and ground_truth.csv")

if __name__ == "__main__":
    generate_seeded_titeseq_data()