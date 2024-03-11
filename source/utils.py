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
    