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
    Estimate the Effective Sample Size (ESS) for a multivariate sample.

    Parameters:
    - samples: A 2D numpy array of shape (n_samples, n_parameters).

    Returns:
    - ess: Estimated Effective Sample Size for the multivariate sample.
    """
    n_samples, n_parameters = samples.shape
    cov_matrix = np.cov(samples, rowvar=False)  # Sample covariance matrix
    variances = np.var(samples, axis=0, ddof=1)  # Variances of individual parameters

    # Calculate determinants
    det_cov_matrix = np.linalg.det(cov_matrix)
    det_variances = np.prod(variances)

    # Normalize determinants by sample size and parameter count to estimate ESS
    ess = (det_variances / det_cov_matrix)**(1.0 / n_parameters) * n_samples
    return ess


def gelman_rubin(chains):
    """
    Calculate the multivariate potential scale reduction factor (PSRF) for MCMC chains.

    Parameters:
    - chains: A 3D numpy array of shape (n_chains, n_samples, n_parameters).

    Returns:
    - psrf: The multivariate PSRF.
    """
    n_chains, n_samples, n_parameters = chains.shape
    
    # Calculate within-chain covariance matrices
    within_chain_cov = np.mean([np.cov(chains[m], rowvar=False) for m in range(n_chains)], axis=0)
    
    # Calculate between-chain covariance matrix
    chain_means = np.mean(chains, axis=1)
    mean_of_means = np.mean(chain_means, axis=0)
    B_over_n = np.cov(chain_means, rowvar=False)
    between_chain_cov = B_over_n * n_samples

    # Calculate W and B
    W = within_chain_cov
    B = between_chain_cov
    
    # Estimate marginal posterior variance (V_hat)
    V_hat = ((n_samples - 1) / n_samples) * W + ((n_chains + 1) / (n_chains * n_samples)) * B

    # Calculate multivariate PSRF
    lambdas, _ = np.linalg.eig(V_hat @ np.linalg.inv(W))
    psrf = np.sqrt(np.max(lambdas) * (n_samples - 1) / n_samples + 1)

    return psrf
