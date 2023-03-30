# Import Dataset
# MNIST

from hashlib import sha256
from gradients.noise import NoNoiseGenerator

class BaseClient:
    def __init__(self, dataset,
                 model,
                 loss_fn,
                 optimizer,
                 device_id,
                 importance=1):
        """_summary_

        Args:
            data (_type_): _description_
            target (_type_): _description_
            model (_type_): _description_
            loss_fn (_type_): _description_
            optimizer (_type_): _description_
            device_id (_type_): _description_
            importance (int, optional): Importance of this client. Defaults to 1.
        """
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.is_train = False
        self.device_id = device_id
        self.importance = importance
        self.noise_generator = NoNoiseGenerator()

    def set_importance(self, importance):
        self.importance = importance

    def set_model_train(self):
        if not self.is_train:
            self.is_train = not self.is_train
            self.model.train()

    def set_model_eval(self):
        if self.is_train:
            self.is_train = not self.is_train
            self.model.eval()

    @property
    def weights(self):
        self.set_model_eval()
        for param in self.model.parameters():
            yield param.data

    @property
    def gradients(self):
        self.set_model_train()
        add_noise = lambda grad: grad + self.noise_generator.get_noise(grad)
        return [add_noise(param.grad.clone().detach())
                for param in self.model.parameters()]

    def train(self):
        # Set the model to training mode
        self.set_model_train()
        # Get the loss sum
        loss_sum = 0

        for inputs, labels in self.dataset:
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.loss_fn(output, labels)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()

        return loss_sum / len(self.data)

    def train_for_gradient(self):
        # Set the model to training mode
        self.set_model_train()
        # Get the loss sum
        loss_sum = 0

        for inputs, labels in self.dataset:
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.loss_fn(output, labels)
            loss.backward()
            loss_sum += loss.item()

        return loss_sum / len(self.data)

