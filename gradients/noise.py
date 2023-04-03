
import torch

## Add Noise to Model gradients


class NoiseGenerator:

    def mechanism(self, gradient: torch.Tensor):
        raise NotImplementedError

    def get_noise(self, gradient: torch.Tensor):
        return self.mechanism(gradient)


class LaplaceNoiseGenerator(NoiseGenerator):

    def __init__(self, sensitivity, epsilon, delta):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta

    def mechanism(self, gradient: torch.Tensor):
        return torch.empty(gradient.size()).laplace_(0, self.sensitivity / self.epsilon)


class GaussianNoiseGenerator(NoiseGenerator):

    def __init__(self, sensitivity=0.1, epsilon=0.001, delta=0.001):
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.delta = delta

    def mechanism(self, gradient: torch.Tensor):
        return torch.empty(gradient.size()).normal_(0, self.sensitivity / self.epsilon)



class NoNoiseGenerator(NoiseGenerator):

    def mechanism(self, gradient: torch.Tensor):
        return torch.zeros(gradient.size())