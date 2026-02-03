import torch
import torch.nn as nn
import numpy as np

class TiteSeqModel(nn.Module):
    def __init__(self, num_variants, b_bg):
        super().__init__()
        # Use log-space parameters for numerical stability
        # Expect per-variant parameters: shape (num_variants, 1)
        self.log_kd = nn.Parameter(torch.full((num_variants, 1), -7.0))
        self.log_a = nn.Parameter(torch.full((num_variants, 1), 10.0))
        # keep background as a scalar float (not a parameter)
        self.b_bg = float(b_bg)

    def forward(self, c):
        # c should be (num_concs,) or (1, num_concs) and will broadcast
        kd = torch.exp(self.log_kd)  # (num_variants, 1)
        a = torch.exp(self.log_a)    # (num_variants, 1)
        # Broadcasting: kd -> (num_variants, num_concs), c -> (1, num_concs)
        return a * (c / (c + kd)) + self.b_bg


def fit_gpu(obs_tensor, concs, b_bg, batch_size=1024, max_epochs=1500, tol=1e-6, lr=0.05, device=None):
    """
    Batch-fit KD and A for many variants on GPU.

    Parameters
    - obs_tensor : torch.Tensor or array-like, shape (num_variants, num_concs)
        Observed responses (e.g., mean fluorescence or other target) per concentration.
    - concs : list or array of concentrations, length num_concs
    - b_bg : scalar background value (same scale as obs_tensor)
    - batch_size : variants per optimization batch (default 1024 for ~16GB GPU; tune if needed)
    - max_epochs : max epochs per batch
    - tol : stopping tolerance for the (sum) loss for the batch
    - lr : optimizer learning rate
    - device : torch device string or torch.device; if None auto-detects CUDA if available

    Returns
    - kd_array, a_array : numpy arrays of length num_variants with fitted values
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    eps = 1e-8

    # Ensure obs_tensor is a torch tensor on cpu (we'll move batches to device)
    if not isinstance(obs_tensor, torch.Tensor):
        obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32)

    num_variants, num_concs = obs_tensor.shape
    c = torch.tensor(concs, dtype=torch.float32, device=device).view(1, num_concs)

    kd_results = np.zeros(num_variants, dtype=float)
    a_results = np.zeros(num_variants, dtype=float)

    # iterate in batches
    for start in range(0, num_variants, batch_size):
        end = min(start + batch_size, num_variants)
        batch_y = obs_tensor[start:end].to(device).float()  # (batch, num_concs)
        batch_n = batch_y.shape[0]

        # build a model for this batch (parameters per variant in the batch)
        model = TiteSeqModel(batch_n, b_bg).to(device)

        # Heuristic initial guesses per variant (robust):
        # a_init = max(observed) - b_bg (clamped to small positive)
        # kd_init: find concentration corresponding to half-max if possible,
        # otherwise use geometric mean of concentration range.
        with torch.no_grad():
            max_vals, _ = torch.max(batch_y, dim=1)
            a_init = torch.clamp(max_vals - b_bg, min=1e-3)  # shape (batch,)

            # convert concentrations to tensor on cpu for indexing logic
            concs_tensor = torch.tensor(concs, dtype=torch.float32)
            half_max = b_bg + (a_init / 2.0)

            # Find the first concentration index where y >= half_max
            kd_init = torch.empty(batch_n, dtype=torch.float32)
            for i in range(batch_n):
                row = batch_y[i].detach().cpu()
                hm = half_max[i].detach().cpu()
                inds = (row >= hm).nonzero(as_tuple=False)
                if inds.numel() > 0:
                    idx = inds[0].item()
                    kd_guess = concs_tensor[idx].item()
                    # clamp to a reasonable interval
                    kd_guess = max(kd_guess, 1e-10)
                else:
                    # fallback: geometric mean of conc range
                    kd_guess = float(torch.sqrt(concs_tensor[0] * concs_tensor[-1]))
                kd_init[i] = kd_guess

            # set model parameters from these guesses
            model.log_kd.copy_(torch.log(kd_init.view(batch_n, 1).to(device) + eps))
            model.log_a.copy_(torch.log(a_init.view(batch_n, 1).to(device) + eps))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Loss: sum of squared residuals in log-space (matches CPU objective)
        for epoch in range(int(max_epochs)):
            optimizer.zero_grad()
            preds = model(c)  # shape (batch, num_concs)
            # Ensure preds positive and batch_y positive
            loss = torch.sum((torch.log(preds + eps) - torch.log(batch_y + eps)) ** 2)
            loss_value = float(loss.detach().cpu().numpy())
            loss.backward()
            optimizer.step()

            # Early stopping for this batch
            if loss_value <= tol:
                break

        # collect results for this batch
        with torch.no_grad():
            kd_vals = torch.exp(model.log_kd).detach().cpu().squeeze().numpy()
            a_vals = torch.exp(model.log_a).detach().cpu().squeeze().numpy()

        # handle 1-element squeeze edge-case
        if kd_vals.shape == ():
            kd_vals = np.array([float(kd_vals)])
            a_vals = np.array([float(a_vals)])

        kd_results[start:end] = kd_vals
        a_results[start:end] = a_vals

    return kd_results, a_results