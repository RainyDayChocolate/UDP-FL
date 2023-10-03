
import torch
import random
import numpy as np
from torch.distributions import Laplace


## Add Noise to Model gradients


class NoiseGenerator:

    def mechanism(self, gradient: torch.Tensor):
        raise NotImplementedError

    def get_noise(self, gradient: torch.Tensor):
        return self.mechanism(gradient)


class LaplaceNoiseGenerator(NoiseGenerator):

    def __init__(self, sensitivity=1, epsilon=1, delta=0.000001):
        self.sensitivity = sensitivity
        self.epsilon = epsilon 
        self.delta = delta

    def mechanism(self, gradient: torch.Tensor):
        laplace_distribution = Laplace(0, self.sensitivity)
        return laplace_distribution.sample(gradient.size())
        #return torch.empty(gradient.size()).laplace_(0, 0.798)


class GaussianNoiseGenerator(NoiseGenerator):

    def __init__(self, sensitivity=0.1, epsilon=0.1, delta=0.000001):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta

    def mechanism(self, gradient: torch.Tensor):
        return torch.empty(gradient.size()).normal_(0, self.sensitivity)

class StaircaseNoiseGenerator(NoiseGenerator):
    def __init__(self, sensitivity=0.01, epsilon=5, gamma=0.498):
        self.sensitivity = sensitivity
        self.epsilon = epsilon 
        self.gamma = gamma
    def mechanism(self, gradient: torch.Tensor):
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Convert parameters to tensors
        epsilon = torch.tensor(self.epsilon)
        gamma = torch.tensor(self.gamma)
        sensitivity = torch.tensor(self.sensitivity)
        # Get the shape of the gradients tensor
        shape = gradient.shape

        # Generate random values for sign, geometric_rv, unif_rv, and binary_rv
        sign = torch.where(torch.rand(shape) < 0.5, -1, 1)
        geometric_rv = torch.distributions.geometric.Geometric(1 - torch.exp(-epsilon)).sample(shape) - 1
        unif_rv = torch.rand(shape)
        binary_rv = torch.where(torch.rand(shape) < gamma / (gamma + (1 - gamma) * torch.exp(-epsilon)), 0, 1)

        # Calculate noise based on the given mechanism
        noise = sign * (
            (1 - binary_rv) * ((geometric_rv + gamma * unif_rv) * sensitivity) +
            binary_rv * ((geometric_rv + gamma + (1 - gamma) * unif_rv) * sensitivity)
        )
        
        return noise

class NoNoiseGenerator(NoiseGenerator):

    # def __init__(self, sensitivity, epsilon, delta):
    #     self.sensitivity = sensitivity
    #     self.epsilon = epsilon
    #     self.delta = delta

    def mechanism(self, gradient: torch.Tensor):
        return torch.zeros(gradient.size())