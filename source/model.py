import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Union


class Model(nn.Module):
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
        super(Model, self).__init__()
        self.sizes = sizes
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.activations = activations
        self.temperature = temperature

        self.loss = nn.CrossEntropyLoss(reduction="sum")
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
            mu: torch.Tensor,
            sigma: torch.Tensor
        ):
        """
        Set prior distribution

        Parameters:
        - mu (torch.Tensor): mean of the distribution
        - sigma (torch.Tensor): standard deviation of the distribution
        """
        num_parameters = self.num_parameters()
        assert len(mu) == num_parameters
        assert len(sigma) == num_parameters
        self.prior = torch.distributions.Normal(loc=mu, scale=sigma)

    def num_parameters(self) -> int:
        """
        Returns the number of model parameters

        Returns:
        - #parameters (int): number of model parameters
        """
        return sum(param.numel() for param in self.parameters())

    def get_parameters(self) -> torch.Tensor:
        """
        Returns model parameters as 1D tensor

        Return:
        - parameters (torch.Tensor): current model parameters
        """
        return torch.cat([param.view(-1) for param in self.parameters()])

    def set_parameters(self, parameters: torch.Tensor):
        """
        Updates model parameters from a 1D tensor

        Parameters:
        - parameters (torch.Tensor): new model parameters
        """
        current_idx = 0
        for param in self.parameters():
            flat_size = np.prod(param.size())
            param.data = parameters[current_idx:current_idx+flat_size].view(param.size()).to(param.device)
            current_idx += flat_size

    def get_grad(self) -> torch.Tensor:
        """
        Returns model parameter gradients in the form of a 1D tensor

        Return:
        - grad (torch.Tensor): current model parameter gradients
        """
        return torch.cat([param.grad.view(-1) for param in self.parameters()])

    def set_grad(self, grad: torch.Tensor):
        """ 
        Updates model parameter gradients from a 1D tensor

        Parameters:
        - grad (torch.Tensor): new model parameter gradients
        """
        current_idx = 0
        for param in self.parameters():
            flat_size = np.prod(param.size())
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
            param.grad = grad[current_idx:current_idx+flat_size].reshape(param.size()).to(param.device)
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
        return -self.temperature * self.loss(self(x), y)

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
        parameters = self.get_parameters()
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

    def compute_grad_log_target(self, log_target: torch.Tensor) -> torch.Tensor:
        """
        Calculation of the gradient of model parameters based on log-likelihood 

        Parameters:
        - log_target (torch.Tensor): log-likelihood value

        Returns:
        - grad (torch.Tensor): model parameter gradients
        """
        grad_log_target = autograd.grad(log_target, self.parameters(), create_graph=True)
        return torch.cat([grad.view(-1) for grad in grad_log_target])

    def predictive_posterior(
            self,
            thetas,
            x,
            y,

        ):
        
        integral = 0
        n_kept_samples = 1
        n_drop_samples = 0

        for theta in thetas:

            self.set_parameters(theta)
            t_integral = self.compute_likelihood(x, y)

            if torch.isnan(t_integral):
                n_drop_samples += 1
            else:
                integral = ((n_kept_samples - 1) * integral + t_integral) / n_kept_samples
                n_kept_samples += 1
