# Base torch MOdel include optimizer and loss function
# Compare this snippet from models/base_model.py:


import torch


class BaseModel:

    def forward(self, x):
        raise NotImplementedError

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def zero_weights(self):
        self.eval()
        for param in self.parameters():
            param.data = torch.zeros_like(param.data)

    def train_for_gradient(self, inputs, labels):
        self.optimizer.zero_grad()
        output = self.forward(inputs)
        loss = self.loss_fn(output, labels)
        loss.backward()

    def train_for_model(self, inputs, labels):
        self.optimizer.zero_grad()
        output = self.forward(inputs)
        loss = self.loss_fn(output, labels)
        loss.backward()
        self.optimizer.step()

    def copy_a_new(self):
        return self.__class__()
