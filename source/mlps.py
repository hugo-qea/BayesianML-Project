import torch
import torch.nn as nn
import numpy as np
from typing import List

class SampleableMLP(nn.Module):

    def __init__(
            self, 
            input_size: int, 
            hidden_sizes: List[int], 
            output_size: int
        ):
        super(SampleableMLP, self).__init__()
        sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def get_parameters(self) -> np.ndarray:
        return np.concatenate([param.data.cpu().numpy().flatten() for param in self.parameters()])

    def set_parameters(self, new_parameters: np.ndarray):
        current_idx = 0
        for param in self.parameters():
            flat_size = np.prod(param.size())
            param.data = torch.from_numpy(new_parameters[current_idx:current_idx+flat_size].reshape(param.size())).to(param.device)
            current_idx += flat_size


if __name__ == "__main__":
    input_size = 2
    hidden_sizes = [3, 2]
    output_size = 2

    mlp = SampleableMLP(input_size, hidden_sizes, output_size)

    sampled_params = mlp.get_parameters()
    print("Initial parameters :", sampled_params)

    updated_params = sampled_params + np.random.normal(0, 0.1, len(sampled_params))
    print("New parameters :", updated_params)

    mlp.set_parameters(updated_params)
    updated_sampled_params = mlp.get_parameters()
    print("Update parameters :", updated_sampled_params)
