from utils import *


class Metropolis_Hastings:
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
                sigma_prop: float,
                theta_0: np.ndarray) -> None:
        """
        Initialize the Metropolis-Hastings Sampler.

        Parameters:
        - log_target: Logarithm of the target distribution (callable).
        - sigma_prop: Step size for the proposal mechanism (float).
        - theta_0: Initial value of the sampler (numpy.ndarray).

        """
        self.log_target = log_target
        self.sigma_prop = sigma_prop
        self.theta_0 = theta_0

    def sample(self,
               n_iter : int,
               n_burn_in: int =0):
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
        theta = self.theta_0
        sample = np.zeros((n_iter + 1, len(theta)))
        sample[0] = theta
        acceptance_rate = 0.0

        for i in tqdm(range(1, n_iter + 1)):
            # Proposal
            theta_star = theta + self.sigma_prop * np.random.randn(len(theta))
            
            # Acceptance probability
            log_ratio = self.log_target(theta_star) - self.log_target(theta) 
                        
            # Acceptance
            if np.log(np.random.rand()) < min(0,log_ratio):
                theta = theta_star
                acceptance_rate += 1.0

            sample[i] = theta

        return sample[n_burn_in::], acceptance_rate / n_iter
