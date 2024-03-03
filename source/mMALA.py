from utils import *
from metric_tensor import MetricTensor


class mMALA:
    """
    Riemann Manifold Metropolis-Adjusted Langevin Algorithm (mMALA) for sampling from a target distribution.

    Parameters:
    - metric_tensor: Object representing the metric tensor.
    - log_target: Logarithm of the target distribution.
    - grad_log_target: Gradient of the logarithm of the target distribution.
    - theta_0: Initial state of the Markov chain.
    - step_size: Step size for the algorithm.
    - flat: Flag indicating whether to use a flat metric (default is True).

    Methods:
    - sample: Run the mMALA algorithm to generate samples from the target distribution.

    """

    def __init__(self,
                metric_tensor: MetricTensor,
                log_target: callable,
                grad_log_target: callable,
                theta_0: np.ndarray,
                step_size: float,
                flat: bool = True) -> None:
        """
        Initialize the mMALA object.

        Parameters:
        - metric_tensor: Object representing the metric tensor.
        - log_target: Logarithm of the target distribution (callable).
        - grad_log_target: Gradient of the logarithm of the target distribution (callable).
        - theta_0: Initial state of the Markov chain (numpy.ndarray).
        - step_size: Step size for the algorithm (float).
        - flat: Flag indicating whether to use a flat metric (default is True).

        """
        self.metric_tensor = metric_tensor
        self.theta_0 = theta_0
        self.step_size = step_size
        self.log_target = log_target
        self.grad_log_target = grad_log_target
        self.flat = flat

    def sample(self,
               n_iter: int,
               n_burn_in: int = 0,
               verbose: bool = False) -> list:
        """
        Run the mMALA algorithm to generate samples from the target distribution.

        Parameters:
        - n_iter: Number of iterations for the algorithm (int).
        - n_burn_in: Number of burn-in iterations (default is 0) (int).
        - verbose: Flag indicating whether to print progress information (default is False) (bool).

        Returns:
        - samples: Array containing the generated samples (numpy.ndarray).
        - acceptance_rate: Acceptance rate of the samples (float).

        """
        theta = self.theta_0
        samples = np.zeros((n_iter+1, theta.shape[0]))
        samples[0] = theta
        acceptance_rates = 0
        
        print("Running mMALA sampler...")
        print("\n")

        for n in tqdm(range(1, n_iter+1)):
            # Proposal theta_star

            if self.flat:
                
                """print(self.metric_tensor.G_inv(theta).dot(self.grad_log_target(theta)))
                print("=====================================")
                print(np.sqrt(self.metric_tensor.G_inv(theta)))
                print("=====================================")
                print(theta)
                print("=====================================")"""
                
                theta_star = theta + 0.5 * self.step_size * self.metric_tensor.G_inv(theta).dot(self.grad_log_target(theta)) + np.sqrt(self.step_size) * la.sqrtm(self.metric_tensor.G_inv(theta)).dot(np.random.randn(theta.shape[0]))
                
            # Compute log ratio
            
            mu_n = theta + 0.5 * self.step_size * self.grad_log_target(theta)
            log_q_star = 0.5 * np.linalg.slogdet(self.metric_tensor.G(theta))[-1] - 0.5 * np.dot((theta_star - mu_n), np.dot(self.metric_tensor.G(theta), (theta_star - mu_n))) / self.step_size
            
            mu_star = theta_star + 0.5 * self.step_size * self.grad_log_target(theta_star)
            log_q = 0.5 * np.linalg.slogdet(self.metric_tensor.G(theta_star))[-1] - 0.5 * np.dot((theta - mu_star), np.dot(self.metric_tensor.G(theta_star), (theta - mu_star))) / self.step_size

            log_ratio = self.log_target(theta_star) - self.log_target(theta) + log_q - log_q_star

            # MH Acceptance
            
            if np.log(np.random.rand()) < min(0,log_ratio):
                #print("accept")
                theta = theta_star
                acceptance_rates += 1.0

            samples[n] = theta

            # Print progress information
            if verbose:
                print(f'Iteration {n}/{n_iter} done, acceptance rate: {acceptance_rates/n}, current sample: {theta}')

        return samples[n_burn_in::], acceptance_rates / n_iter
