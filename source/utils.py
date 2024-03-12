import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import torch
from tqdm import tqdm
from typing import Union
from scipy.stats import multivariate_normal


def numpy_to_tensor_decorator(func):
    def wrapper(self, *args, **kwargs):
        converted_args = [torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
        converted_kwargs = {key: torch.from_numpy(val) if isinstance(val, np.ndarray) else val for key, val in kwargs.items()}

        result = func(self, *converted_args, **converted_kwargs)
        
        #if isinstance(result, np.ndarray):
        #    result = torch.from_numpy(result)
        
        return result
    return wrapper

def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """
    Calculate the Effective Sample Size (ESS) for each parameter based on the autocorrelation.

    Parameters:
    - samples: Array containing the generated samples after burn-in (numpy.ndarray).

    Returns:
    - ess: Effective Sample Size for each parameter (numpy.ndarray).
    """
    n_samples = samples.shape[0]
    n_params = samples.shape[1]
    mean_samples = np.mean(samples, axis=0)
    var_samples = np.var(samples, axis=0, ddof=1)
    ess = np.zeros(n_params)

    for param in range(n_params):
        autocorr_sum = 0
        for lag in range(1, n_samples):
            autocorr_lag = np.corrcoef(samples[:n_samples-lag, param], samples[lag:, param])[0, 1]
            if autocorr_lag <= 0:
                break
            autocorr_sum += autocorr_lag

        ess[param] = n_samples / (1 + 2 * autocorr_sum)

    return ess
