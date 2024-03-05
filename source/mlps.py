import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import List, Union


class SampleableMLP(nn.Module):
    """
    MLPs whose parameters can be sampled with MCMCs

    Parameters:
    - input_size (int): model input size
    - hidden_size (list(int)): model hiddens sizes list
    - output_size (int): model output size
    - activations (list(nn.Module)): activation lists
        must contain an activation for each layer or None otherwise
    - temperature (float): temperature for log-target

    Attributes:
    - multiclass: True for multilclass classification, False for binary classification

    Methods:
    - forward:
    - num_parameters: returns the number of model parameters
    - get_parameters: returns model parameters as 1D numpy array
    - set_parameters: updates model parameters from a 1D array
    - get_grad: returns model parameter gradients in the form of a 1D array
    - set_grad: updates model parameter gradients from a 1D array
    - compute_grad_log_target:
    - 
    """

    def __init__(
            self, 
            input_size: int, 
            hidden_sizes: List[int],
            output_size: int,
            activations: List[nn.Module],
            temperature: float=1.,
        ):
        super(SampleableMLP, self).__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.activations = activations
        self.temperature = temperature

        self.multiclass = (output_size > 1)
        self.loss = nn.CrossEntropyLoss(reduction="sum") if self.multiclass else nn.BCELoss(reduction="sum")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation is not None:
                x = activation(x)
        return x

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
    input_size = 2
    hidden_sizes = [3, 2]
    output_size = 2
    activations = 3 * [nn.ReLU()] + [None]

    mlp = SampleableMLP(input_size, hidden_sizes, output_size, activations)
    print("Number of parameters :", mlp.num_parameters())

    sampled_params = mlp.get_parameters()
    print("Initial parameters :", sampled_params)

    updated_params = sampled_params + np.random.normal(0, 0.1, len(sampled_params))
    print("New parameters :", updated_params)

    mlp.set_parameters(updated_params)
    updated_sampled_params = mlp.get_parameters()
    print("Update parameters :", updated_sampled_params)
