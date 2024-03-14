from utils import *
from sampler import Sampler

class HMC(Sampler):
    """
    Hamiltonian Monte Carlo (HMC) Sampler

    Parameters
    ----------
    log_target : callable
        Logarithm of the target distribution.
    grad_log_target : callable
        Gradient of the logarithm of the target distribution.
    step_size : float
        Step size for the leapfrog integrator.
    n_leapfrog : int
        Number of leapfrog steps to perform for each sample.
    theta_0 : array
        Initial value of the parameters.
    """

    def __init__(self, log_target: callable, grad_log_target: callable, n_leapfrog: int, theta_0: np.ndarray,  step_size: float = None) -> None:
        super().__init__(log_target, theta_0)
        self.grad_log_target = grad_log_target
        self.n_leapfrog = n_leapfrog
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 0.0001 * np.power(len(theta_0), -0.5)

    def leapfrog(self, theta, p,step_size):
        """
        Performs the leapfrog integration to simulate Hamiltonian dynamics.

        Parameters:
        - theta: Current position (parameters).
        - p: Current momentum.
        - step_size: Step size for integration.

        Returns:
        - theta, p: New position and momentum after leapfrog steps.
        """
        p -= step_size / 2 * self.grad_log_target(theta)  # half step for momentum
        for _ in range(self.n_leapfrog - 1):
            theta += step_size * p  # full step for position
            p -= step_size * self.grad_log_target(theta)  # full step for momentum
        theta += step_size * p  # final full step for position
        p -= step_size / 2 * self.grad_log_target(theta)  # half step for momentum

        return theta, p

    def sample(self, n_iter: int, n_burn_in: int = 0, verbose: bool = False, return_burn_in: bool = True):
        if n_iter <= n_burn_in:
            raise ValueError("n_iter must be greater than n_burn_in.")

        theta = self.theta_0
        samples = np.zeros((n_iter + 1, len(theta)))
        samples[0] = theta
        acceptance_rate = 0.0

        for i in tqdm(range(1, n_iter + 1)):
            p_current = np.random.randn(*theta.shape)  # draw random momentum
            theta_proposed, p_proposed = self.leapfrog(np.copy(theta), np.copy(p_current), self.step_size)

            # Hamiltonian calculations
            current_H = -self.log_target(theta) + np.sum(p_current**2) / 2
            proposed_H = -self.log_target(theta_proposed) + np.sum(p_proposed**2) / 2

            # Metropolis acceptance step
            if np.random.rand() < np.exp(current_H - proposed_H):
                theta = theta_proposed
                acceptance_rate += 1.0

            samples[i] = theta

            if verbose and i % 10000 == 0:
                print(f'Iteration {i}/{n_iter} done, acceptance rate: {acceptance_rate/i}')
        if return_burn_in:
            return samples, acceptance_rate / n_iter
        else:
            return samples[n_burn_in:], acceptance_rate / n_iter
