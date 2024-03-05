import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Union


class SampleableMLP(nn.Module):
    """
    MLPs whose parameters can be sampled with MCMCs

    Parameters:
    - sizes (list(int)): model sizes list
    - activations (list(nn.Module)): activation lists
        must contain an activation for each layer or None otherwise
    - temperature (float): temperature for log-target

    Attributes:
    - loss (nn.Module): CrossEntropyLoss for multiclass classification, BCELoss for binary classification
    - prior (torch.distributions.Normal): prior distribution
    """

    def __init__(
            self,
            sizes: List[int],
            activations: List[nn.Module],
            temperature: float=1.,
        ):
        super(SampleableMLP, self).__init__()
        self.sizes = sizes
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.activations = activations
        self.temperature = temperature

        is_multiclass = (self.sizes[-1] > 1)
        self.loss = nn.CrossEntropyLoss(reduction="sum") if is_multiclass else nn.BCELoss(reduction="sum")

        self.prior = torch.distributions.Normal(
            loc=torch.zeros(self.num_parameters()),
            scale=torch.ones(self.num_parameters())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation is not None:
                x = activation(x)
        return x

    def get_prior(self) -> torch.distributions.Normal:
        """
        Returns the prior distribution

        Returns:
        - prior (torch.distributions.Normal): prior distribution
        """
        return self.prior

    def set_prior(
            self, 
            mu: Union[np.ndarray, torch.Tensor],
            sigma: Union[np.ndarray, torch.Tensor]
        ):
        """
        Set prior distribution

        Parameters:
        - mu (np.ndarray or torch.Tensor): mean of the distribution
        - sigma (np.ndarray or torch.Tensor): standard deviation of the distribution
        """
        num_parameters = self.num_parameters()
        assert len(mu) == num_parameters
        assert len(sigma) == num_parameters

        if isinstance(mu, np.ndarray):
            mu = torch.from_numpy(mu)
        if isinstance(sigma, np.ndarray):
            sigma = torch.from_numpy(sigma)
        
        self.prior = torch.distributions.Normal(loc=mu, scale=sigma)

    def num_parameters(self) -> int:
        """
        Returns the number of model parameters

        Returns:
        - #parameters (int): number of model parameters
        """
        return sum(p.numel() for p in self.parameters())

    def get_parameters(self) -> np.ndarray:
        """
        Returns model parameters as 1D numpy array

        Return:
        - parameters (np.ndarray): current model parameters
        """
        return np.concatenate([param.data.cpu().detach().numpy().flatten() for param in self.parameters()])

    def set_parameters(self, parameters: np.ndarray):
        """
        Updates model parameters from a 1D array

        Parameters:
        - parameters (np.ndarray): new model parameters
        """
        current_idx = 0
        for param in self.parameters():
            flat_size = np.prod(param.size())
            param.data = torch.from_numpy(
                parameters[current_idx:current_idx+flat_size].reshape(param.size())
            ).to(param.device)
            current_idx += flat_size

    def get_grad(self) -> np.ndarray:
        """
        Returns model parameter gradients in the form of a 1D array

        Return:
        - grad (np.ndarray): current model parameter gradients
        """
        return np.concatenate([param.grad.cpu().detach().numpy().flatten() for param in self.parameters()])

    def set_grad(self, grad: np.ndarray):
        """ 
        Updates model parameter gradients from a 1D array

        Parameters:
        - grad (np.ndarray): new model parameter gradients
        """
        current_idx = 0
        for param in self.parameters():
            flat_size = np.prod(param.size())
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
            param.grad = torch.from_numpy(
                grad[current_idx:current_idx+flat_size].reshape(param.size())
            ).to(param.device)
            current_idx += flat_size

    def compute_log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculation of the log-likelihood of data

        Parameters:
        - x (torch.Tensor): input
        - y (torch.Tensor): output

        Returns:
        - log-likelihood (torch.Tensor): log-likelihood
        """
        return -self.temperature * self.loss(x, y)

    def compute_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculation of the likelihood of data

        Parameters:
        - x (torch.Tensor): input
        - y (torch.Tensor): output

        Returns:
        - likelihood (torch.Tensor): likelihood
        """
        return torch.exp(self.compute_log_likelihood(x, y))

    def compute_log_prior(self) -> torch.Tensor:
        """
        Calculation of the log prior

        Returns:
        - log prior (torch.Tensor): log prior
        """
        parameters = torch.from_numpy(self.get_parameters())
        return self.temperature * torch.sum(self.prior.log_prob(parameters))

    def compute_log_target(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculation of the log target: log-likelihood + log prior

        Parameters:
        - x (torch.Tensor): input
        - y (torch.Tensor): output

        Returns:
        - log target (torch.Tensor): log target
        """
        log_likelihood = self.compute_log_likelihood(x, y)
        log_prior = self.compute_log_prior()
        return log_likelihood + log_prior

    def compute_grad_log_target(self, log_target: torch.Tensor) -> np.ndarray:
        """
        Calculation of the gradient of model parameters based on log-likelihood 

        Parameters:
        - log_target (torch.Tensor): log-likelihood value

        Returns:
        - grad (np.ndarray): model parameter gradients
        """
        grad_log_target = autograd.grad(log_target, self.parameters(), create_graph=True)
        return np.concatenate([grad.cpu().detach().numpy().flatten() for grad in grad_log_target])
         

if __name__ == "__main__":
    sizes = [2, 1, 2, 3]
    activations = 3 * [nn.ReLU()] + [None]

    mlp = SampleableMLP(sizes, activations)
    print("Number of parameters :", mlp.num_parameters())

    sampled_params = mlp.get_parameters()
    print("Initial parameters :", sampled_params)

    updated_params = sampled_params + np.random.normal(0, 0.1, len(sampled_params))
    print("New parameters :", updated_params)

    mlp.set_parameters(updated_params)
    updated_sampled_params = mlp.get_parameters()
    print("Update parameters :", updated_sampled_params)
