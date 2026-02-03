import numpy as np
from scipy.optimize import minimize

def hill_function(c, kd, a, b):
    return a * (c / (c + kd)) + b

def fit_cpu(concentrations, mean_log_fluors, b_bg):
    """Fits KD for a single sequence using L-BFGS-B."""
    def objective(params):
        kd, a = params
        preds = hill_function(concentrations, kd, a, b_bg)
        # Residuals in log-space per Appendix 5
        return np.sum((np.log(preds + 1e-10) - mean_log_fluors)**2)

    init_guess = [1e-7, np.exp(max(mean_log_fluors))]
    bounds = [(1e-10, 1e-3), (1, 1e6)]
    res = minimize(objective, init_guess, bounds=bounds, method='L-BFGS-B')
    return res.x