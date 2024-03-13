import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Union, Optional
from utils import numpy_to_tensor_decorator


class Model(nn.Module):
    """
    MLPs whose parameters can be sampled with MCMCs

    Parameters:
    - sizes (list(int)): model sizes list
    - activations (list(nn.Module)): activation lists
        must contain an activation for each layer or None otherwise
    - temperature (float): temperature for log-target
    - device: device

    Attributes:
    - loss (nn.Module): CrossEntropyLoss for multiclass classification, BCELoss for binary classification
    - prior (torch.distributions.Normal): prior distribution
    """

    def __init__(
            self,
            sizes: List[int],
            activations: List[nn.Module],
            temperature: float=1.,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ):
        super(Model, self).__init__()
        self.sizes = sizes
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.activations = activations
        self.temperature = temperature
        self.device = device

        self.loss = nn.CrossEntropyLoss(reduction="sum")
        self.set_prior(
            mu=torch.zeros(self.num_parameters()),
            sigma=torch.ones(self.num_parameters())
        )
        
        self.to(self.device)

    @numpy_to_tensor_decorator
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
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

    @numpy_to_tensor_decorator
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
        mu = mu.to(self.device)
        sigma = sigma.to(self.device)
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

    @numpy_to_tensor_decorator
    def set_parameters(self, parameters: torch.Tensor):
        """
        Updates model parameters from a 1D tensor

        Parameters:
        - parameters (torch.Tensor): new model parameters
        """
        parameters = parameters.to(self.device)
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

    @numpy_to_tensor_decorator
    def set_grad(self, grad: torch.Tensor):
        """ 
        Updates model parameter gradients from a 1D tensor

        Parameters:
        - grad (torch.Tensor): new model parameter gradients
        """
        grad = grad.to(self.device)
        current_idx = 0
        for param in self.parameters():
            flat_size = np.prod(param.size())
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
            param.grad = grad[current_idx:current_idx+flat_size].reshape(param.size()).to(param.device)
            current_idx += flat_size

    @numpy_to_tensor_decorator
    def compute_log_likelihood(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor,
            parameters: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:
        """
        Calculation of the log-likelihood of data

        Parameters:
        - x (torch.Tensor): input
        - y (torch.Tensor): output

        Optional parameters:
        - parameters (torch.Tensor): new parameters

        Returns:
        - log-likelihood (torch.Tensor): log-likelihood
        """
        if parameters is not None:
            self.set_parameters(parameters)
        return -self.temperature * self.loss(self(x), y.to(self.device))

    @numpy_to_tensor_decorator
    def compute_likelihood(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor,
            parameters: Optional[torch.Tensor]=None,
        ) -> torch.Tensor:
        """
        Calculation of the likelihood of data

        Parameters:
        - x (torch.Tensor): input
        - y (torch.Tensor): output

        Optional parameters:
        - parameters (torch.Tensor): new parameters

        Returns:
        - likelihood (torch.Tensor): likelihood
        """
        return torch.exp(self.compute_log_likelihood(x, y, parameters))

    @numpy_to_tensor_decorator
    def compute_log_prior(
            self, 
            parameters: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        Calculation of the log prior

        Optional parameters:
        - parameters (torch.Tensor): new parameters

        Returns:
        - log prior (torch.Tensor): log prior
        """
        if parameters is None:
            parameters = self.get_parameters()
        parameters = parameters.to(self.device)
        return self.temperature * torch.sum(self.prior.log_prob(parameters))

    @numpy_to_tensor_decorator
    def compute_log_target(
            self, 
            x: torch.Tensor, 
            y: torch.Tensor,
            parameters: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
        """
        Calculation of the log target: log-likelihood + log prior

        Parameters:
        - x (torch.Tensor): input
        - y (torch.Tensor): output

        Optional parameters:
        - parameters (torch.Tensor): new parameters

        Returns:
        - log target (torch.Tensor): log target
        """
        log_likelihood = self.compute_log_likelihood(x, y, parameters)
        log_prior = self.compute_log_prior(parameters)
        return log_likelihood + log_prior

    @numpy_to_tensor_decorator
    def compute_grad_log_target(self, log_target: torch.Tensor) -> torch.Tensor:
        """
        Calculation of the gradient of model parameters based on log-likelihood 

        Parameters:
        - log_target (torch.Tensor): log-likelihood value

        Returns:
        - grad (torch.Tensor): model parameter gradients
        """
        log_target = log_target.to(self.device)
        grad_log_target = autograd.grad(log_target, self.parameters(), create_graph=True)
        return torch.cat([grad.view(-1) for grad in grad_log_target])

    @numpy_to_tensor_decorator
    def predictive_posterior(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            parameters_chain: torch.Tensor,
        ) -> tuple:
        """
        Calculation of the posterior predictive distribution

        Parameters:
        - x (torch.Tensor): input
        - y (torch.Tensor): output
        - parameters_chain (torch.Tensor): (chain_length, num_parameters) parameters 

        Returns
        - posterior predictive distribution (torch.Tensor)
        - fail (int): number of fails
        """
        posterior = 0.
        sucess = 1
        fail = 0

        for parameters in parameters_chain:
            posterior_param = self.compute_likelihood(x, y, parameters)
            if not torch.isnan(posterior_param):
                posterior = ((sucess - 1) * posterior + posterior_param) / sucess
                sucess += 1
            else:
                fail += 1
        
        return posterior, fail

    @numpy_to_tensor_decorator
    def predict(
            self,
            x: torch.Tensor,
            parameters_chain: torch.Tensor,
            return_probas: bool=False,
            return_fails: bool=False
        )-> tuple:
        """
        Calculation of the posterior predictive distribution

        Parameters:
        - x (torch.Tensor): input
        - parameters_chain (torch.Tensor): (chain_length, num_parameters) parameters 
        - return_probas (bool): return probas of each class
        - return_fails (bool): return number of fails 

        Returns
        - predicted classes 
        - posterior predictive distribution (torch.Tensor) 
        - fail (int): number of fails
        """
        n_classes = self.sizes[-1]
        batch_size = x.size(0) # must be 1

        y_posterior_probas = torch.zeros((batch_size, n_classes), dtype=torch.float32)
        y_fails = torch.zeros((n_classes,), dtype=torch.long)

        for i in range(n_classes):
            y = torch.LongTensor([i] * batch_size).to(self.device)
            posterior, fail = self.predictive_posterior(x, y, parameters_chain)

            y_posterior_probas[:, i] = posterior
            y_fails[i] = fail

            if n_classes == 2:
                y_posterior_probas[:, 1] = 1 - posterior
                y_fails[1] = fail
                break

        y_pred = torch.argmax(y_posterior_probas, dim=1)

        ret = [y_pred]
        if return_probas:
            ret += [y_posterior_probas]
        if return_fails:
            ret += [y_fails]

        return tuple(ret)

