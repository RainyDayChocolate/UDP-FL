# Import Dataset
# MNIST

from hashlib import sha256
from gradients.noise import NoNoiseGenerator
from models.base_model import BaseModel


class BaseClient:
    def __init__(self,
                 model: BaseModel,
                 client_id,
                 importance=1,
                 noise_generator=None):
        """_summary_

        Args:
            model (_type_): _description_
            client_id (_type_): _description_
            importance (int, optional): Importance of this client. Defaults to 1.
        """
        self.model = model
        self.is_train = False
        self.device_id = client_id
        self.importance = importance
        if noise_generator is None:
            self.noise_generator = NoNoiseGenerator()
        else:
            self.noise_generator = noise_generator

        # Default training method as train_for_model
        self.set_training_mode()

    def set_importance(self, importance):
        self.importance = importance

    @property
    def weights(self):
        self.model.eval()
        for param in self.model.parameters():
            yield param.data

    @property
    def gradients(self):
        self.model.train()
        add_noise = lambda grad: grad + self.noise_generator.get_noise(grad)
        return [add_noise(param.grad.clone().detach())
                for param in self.model.parameters()]

    def set_training_mode(self, for_gradient=False):
        if for_gradient:
            self.train = self.train_for_gradient
        else:
            self.train = self.train_for_model

    def train_for_model(self, dataset):
        # Set the model to training mode
        self.model.train()
        # Get the loss sum
        for inputs, labels in dataset:
            self.model.train_for_model(inputs, labels)

    def train_for_gradient(self, dataset):
        # Set the model to evaluation mode
        self.model.train()
        for inputs, labels in dataset:
            self.model.train_for_gradient(inputs, labels)
