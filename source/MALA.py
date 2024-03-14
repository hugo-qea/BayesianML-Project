from utils import *
from sampler import Sampler


class MALA(Sampler):
    """ 
    Metropolis-Adjusted Langevin Algorithm (MALA) Sampler
    
    This class implements the Metropolis-Adjusted Langevin Algorithm (MALA) for sampling from a target distribution.

    Parameters
    ----------
    log_target : callable
        Logarithm of the target distribution.
    grad_log_target : callable
        Gradient of the logarithm of the target distribution.
    step_size : float
        Step size for the MALA sampler.
    theta_0 : array
        Initial value of the sampler.

    Methods
    -------
    sample(n_iter, n_burn_in=0)
        Run the MALA sampler to generate samples from the target distribution.

    """

    def __init__(self,
                log_target: callable,
                grad_log_target: callable,
                step_size: float,
                theta_0: np.ndarray) -> None:
        """
        Initialize the MALA Sampler.

        Parameters:
        - log_target: Logarithm of the target distribution (callable).
        - grad_log_target: Gradient of the logarithm of the target distribution (callable).
        - step_size: Step size for the MALA sampler (float).
        - theta_0: Initial value of the sampler (numpy.ndarray).

        """
        super().__init__(log_target,theta_0)
        self.step_size = step_size
        self.grad_log_target = grad_log_target

    def sample(self,
               n_iter : int,
               n_burn_in: int =0,
               verbose : bool =False) -> list:
        """
        Run the Metropolis-Adjusted Langevin Algorithm (MALA) sampler to generate samples from the target distribution.

        Parameters:
        - n_iter: Number of iterations for the MALA sampler (int).
        - n_burn_in: Number of burn-in iterations (default is 0) (int).

        Returns:
        - sample: Array containing the generated samples (numpy.ndarray).
        - acceptance_rate: Acceptance rate of the MALA sampler (float).

        The MALA sampler iteratively updates the current state using a proposal mechanism based on the gradient of the
        log-target distribution and accepts or rejects proposed states using the Metropolis-Hastings acceptance criterion.
        The resulting samples approximate the target distribution.

        """
        if n_iter <= n_burn_in:
            raise ValueError("n_iter must be greater than n_burn_in.")
        
        theta = self.theta_0
        sample = np.zeros((n_iter + 1, len(theta)))
        sample[0] = theta
        acceptance_rate = 0.0
        #print(theta.shape)
        print("Running MALA sampler...")
        # print("\n")
        # print("Number of iterations: {}".format(n_iter))
        # print("\n")

        for i in tqdm(range(1, n_iter + 1)):
            # mu_n
            mu_n = theta + 0.5 * self.step_size * self.grad_log_target(theta)
            # print(f"mu_n: {mu_n.shape}")

            # Proposal theta_star
            theta_star = theta + 0.5 * self.step_size * self.grad_log_target(theta) + np.sqrt(
                self.step_size) * np.random.randn(len(theta))
            # print(f"theta_star: {theta_star.shape}")

            # mu_star
            mu_star = theta_star + 0.5 * self.step_size * self.grad_log_target(theta_star)
            
            # print("log_ratio")
            # Log ratio
            log_ratio = self.log_target(theta_star) - (0.5 * np.sum((theta - mu_star) ** 2) / self.step_size) \
                        - self.log_target(theta) + 0.5 * np.sum((theta_star - mu_n) ** 2) / self.step_size
                        
            # Acceptance
            if np.log(np.random.rand()) < min(0,log_ratio):
                theta = theta_star
                acceptance_rate += 1.0

            sample[i] = theta
            if verbose and i % 10000 == 0:
                print(f'Iteration {i}/{n_iter} done, acceptance rate: {acceptance_rate/i}')

        return sample[n_burn_in::], acceptance_rate / n_iter
    
    
class AdaptiveMALA(Sampler):
    def __init__(self,
                 log_target: callable,
                 grad_log_target: callable,
                 theta_0: np.ndarray,
                 step_size: float = None,
                 target_acceptance_rate: float = 0.574,
                 adapt_interval: int = 50,
                 adapt_size: float = 0.1) -> None:
        super().__init__(log_target, theta_0)
        if step_size is None:
            self.step_size = np.power(len(theta_0), -0.33) * np.square(0.01)
        else :
            self.step_size = step_size
        self.grad_log_target = grad_log_target
        self.target_acceptance_rate = target_acceptance_rate
        self.adapt_interval = adapt_interval
        self.adapt_size = adapt_size

    def sample(self, n_iter: int, n_burn_in: int = 0, verbose: bool = False, return_burn_in: bool = True):
        theta = self.theta_0
        sample = np.zeros((n_iter + 1, len(theta)))
        sample[0] = theta
        accepted = 0
        acceptance_rate = 0.0

        for i in tqdm(range(1, n_iter + 1)):
            mu_n = theta + 0.5 * self.step_size * self.grad_log_target(theta)
            theta_star = mu_n + np.sqrt(self.step_size) * np.random.randn(len(theta))
            mu_star = theta_star + 0.5 * self.step_size * self.grad_log_target(theta_star)
            
            log_ratio = self.log_target(theta_star) - self.log_target(theta) \
                        - (0.5 / self.step_size) * np.sum((theta - mu_star) ** 2) \
                        + (0.5 / self.step_size) * np.sum((theta_star - mu_n) ** 2)
            
            if np.log(np.random.rand()) < min(0, log_ratio):
                theta = theta_star
                accepted += 1
                acceptance_rate += 1.0

            sample[i] = theta
            
            # Adaptive step size adjustment
            if i % self.adapt_interval == 0:
                current_acceptance_rate = accepted / self.adapt_interval
                if current_acceptance_rate < self.target_acceptance_rate:
                    self.step_size *= (1 - self.adapt_size)
                else:
                    self.step_size *= (1 + self.adapt_size)
                accepted = 0  # Reset accepted count for next interval

            if verbose and i % 1000 == 0:
                print(f"Iteration {i}/{n_iter}, Acceptance Rate: {acceptance_rate / i}, Step Size: {self.step_size}")

        if return_burn_in:
            return sample, acceptance_rate / n_iter
        else:
            return sample[n_burn_in:], acceptance_rate / n_iter
