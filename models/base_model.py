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

    def get_gradient(self, inputs, labels):
        """For this function, it should always work as follows
        zero_grad() -> get_gradient() -> step() -> zero_grad() -> ...

        Args:
            inputs (_type_): _description_
            labels (_type_): _description_
        """
        output = self.forward(inputs)
        loss = self.loss_fn(output, labels)
        loss.backward()

    def copy_a_new(self):
        return self.__class__()
