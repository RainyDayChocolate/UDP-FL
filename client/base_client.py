# Import Dataset
# MNIST

import random
from hashlib import sha256

from gradients.clipper import base_clip
from gradients.noise import NoNoiseGenerator
from models.base_model import BaseModel


class BaseClient:
    def __init__(self,
                 model: BaseModel,
                 client_id,
                 importance=1,
                 clipper=base_clip,
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
        if noise_generator is None:
            self.noise_generator = NoNoiseGenerator()
        else:
            self.noise_generator = noise_generator

        if clipper is None:
            self.clipper = base_clip
        else:
            self.clipper = clipper
        self.set_training_mode()

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
<<<<<<< Updated upstream
            yield param.data
=======
            yield param.data #+ self.noise_generator.get_noise(param.data)
>>>>>>> Stashed changes

    @property
    def gradients(self):
        #
        self.model.train()
        # clip, add noise
        # clip(noise(param.grad.clone().detach()))
        def helper(grad):
            clipped = self.clipper(grad)
            add_noise = lambda grad: grad + self.noise_generator.get_noise(grad)
            return add_noise(clipped)

        # grads = [param.grad.clone().detach() for param in self.model.parameters]
        # clipped = [self.clipper(grad) for grad in grads]
        # noise_grad = [self.noise_generator.get_noise(grad) + grad for grad in cliped]
        # return noise_grad

        return [helper(param.grad.clone().detach())
                for param in self.model.parameters()]

    def set_training_mode(self, for_gradient=False):
        if for_gradient:
            self.train = self.train_for_gradient
        else:
            self.train = self.train_for_model

    def add_noise(self, params):
        """This is a function to add noise to the params inplace, very dangerous

        Args:
            params: Pytorch parameter

        """
        for param in params:
            grad = param.grad
            clipped = self.clipper(grad)
            param.grad = clipped + self.noise_generator.get_noise(clipped)

    def train_for_model(self, dataset, client_epochs=2):
        # Set the model to training mode
        self.model.train()
        # Get the loss sum
        for _ in range(client_epochs):
            for inputs, labels in dataset:
                self.model.zero_grad()
                self.model.get_gradient(inputs, labels)
                self.add_noise(self.model.parameters())
                self.model.step()

    def train_for_gradient(self, dataset, client_epochs=1):
        # Set the model to evaluation mode
        self.model.train()
        for _ in range(client_epochs):
            for inputs, labels in dataset:
                self.model.zero_grad()
                self.model.get_gradient(inputs, labels)
                # Should add yield gradient here
                self.model.step()

    @property
    def fake_device_id(self):
        return random.random()