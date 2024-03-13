from utils import *
from sampler import Sampler


class Metropolis_Hastings(Sampler):
    """
    Class representing the Metropolis-Hastings algorithm for sampling from a target distribution.

    Attributes:
    - log_target (callable): Logarithm of the target distribution.
    - sigma_prop (float): Step size for the proposal mechanism.
    - theta_0 (numpy.ndarray): Initial value of the sampler.

    Methods:
    - sample(n_iter, n_burn_in=0): Run the Metropolis-Hastings sampler to generate samples from the target distribution.

    """

    def __init__(self,
                log_target: callable,
                theta_0: np.ndarray,
                sigma_prop: float =None) -> None:
        """
        Initialize the Metropolis-Hastings Sampler.

        Parameters:
        - log_target: Logarithm of the target distribution (callable).
        - sigma_prop: Step size for the proposal mechanism (float).
        - theta_0: Initial value of the sampler (numpy.ndarray).

        """
        super().__init__(log_target, theta_0)
        self.sigma_prop = sigma_prop

    def sample(self,
               n_iter : int,
               n_burn_in: int =0,
               verbose : bool =False,
               return_burn_in: bool = True) -> list:
        """
        Run the Metropolis-Hastings sampler to generate samples from the target distribution.

        Parameters:
        - n_iter: Number of iterations for the sampler (int).
        - n_burn_in: Number of burn-in iterations (default is 0) (int).

        Returns:
        - sample: Array containing the generated samples (numpy.ndarray).
        - acceptance_rate: Acceptance rate of the sampler (float).

        The Metropolis-Hastings sampler iteratively updates the current state using a proposal mechanism based on the
        log-target distribution and accepts or rejects proposed states using the Metropolis-Hastings acceptance criterion.
        The resulting samples approximate the target distribution.

        """
        if n_iter <= n_burn_in:
            raise ValueError("n_iter must be greater than n_burn_in.")
        
        theta = self.theta_0
        sample = np.zeros((n_iter + 1, len(theta)))
        sample[0] = theta
        acceptance_rate = 0.0
        flag = False
        accepted = 0.0
        if self.sigma_prop is None:
            flag = True
            l = 0
            self.sigma_prop = np.exp(l)

        for i in tqdm(range(1, n_iter + 1)):
            # Proposal
            theta_star = theta + self.sigma_prop * np.random.randn(len(theta))
            
            # Acceptance probability
            log_ratio = self.log_target(theta_star) - self.log_target(theta) 
                        
            # Acceptance
            if np.log(np.random.rand()) < min(0,log_ratio):
                theta = theta_star
                acceptance_rate += 1.0
                accepted += 1.0

            sample[i] = theta
            
            if flag and i %50 ==0:
                accepted /= 50
                delta = min(0.01, 1/np.sqrt(i))
                if accepted < 0.24:
                    l -= delta
                else:
                    l += delta
                self.sigma_prop = np.exp(l)
                accepted = 0.0
                
            if verbose and i % 10000 == 0:
                print(f"Acceptance rate at iteration {i}: {acceptance_rate / i}")

        if return_burn_in:
            return sample, acceptance_rate / n_iter
        return sample[n_burn_in::], acceptance_rate / n_iter
