# Import Dataset
# MNIST

import random
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
        self.fake_id = client_id

        self.set_training_mode()
        if noise_generator is None:
            self.noise_generator = NoNoiseGenerator()
        else:
            self.noise_generator = noise_generator

    def set_importance(self, importance):
        self.importance = importance

    def set_fake_id(self, fakeid):
        self.fake_id = fakeid

    def get_id(self):
        return self.fake_id

    @property
    def weights(self):
        # What to upload
        self.model.eval()
        for param in self.model.parameters():
            yield param.data #+ self.noise_generator.get_noise(param.data)

    @property
    def gradients(self):
        #
        self.model.train()
        return [param.grad.clone().detach()
                for param in self.model.parameters()]

    def train(self, dataset, client_epochs=2):
        return self.train_for_model(dataset, client_epochs)

    def set_training_mode(self, for_gradient=False):
        if for_gradient:
            self.train = self.train_for_gradient
        else:
            self.train = self.train_for_model

    def train_for_model(self, dataset, client_epochs=2):
        # Set the model to training mode
        self.model.train()
        # Get the loss sum
        for epoch in range(client_epochs):
            running_loss = 0
            sample_dataloder = next(dataset)
            self.model.train_dpsgd(sample_dataloder, self.noise_generator)

    @property
    def fake_device_id(self):
        return random.random()