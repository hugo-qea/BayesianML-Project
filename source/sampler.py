from utils import *
from abc import ABC, abstractmethod

class Sampler(ABC):
    """
    Abstract base class representing a generic sampler.

    Attributes:
    - log_target (callable): Logarithm of the target distribution.
    - sigma_prop (float): Step size for the proposal mechanism.
    - theta_0 (numpy.ndarray): Initial value of the sampler.

    Methods:
    - sample(n_iter, n_burn_in=0): Abstract method to run the sampler.
    """

    def __init__(self, log_target: callable, theta_0: np.ndarray) -> None:
        """
        Initialize the Sampler.

        Parameters:
        - log_target: Logarithm of the target distribution (callable).
        - sigma_prop: Step size for the proposal mechanism (float).
        - theta_0: Initial value of the sampler (numpy.ndarray).
        """
        self.log_target = log_target
        self.theta_0 = theta_0

    @abstractmethod
    def sample(self, n_iter: int, n_burn_in: int = 0):
        """
        Abstract method to run the sampler.

        Parameters:
        - n_iter: Number of iterations for the sampler (int).
        - n_burn_in: Number of burn-in iterations (default is 0) (int).

        Returns:
        - sample: Array containing the generated samples (numpy.ndarray).
        - acceptance_rate: Acceptance rate of the sampler (float).
        """
        pass
