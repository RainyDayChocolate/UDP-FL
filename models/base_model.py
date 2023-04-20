# Base torch MOdel include optimizer and loss function
# Compare this snippet from models/base_model.py:


import torch
from torch.nn.utils import clip_grad_norm_

class BaseModel:

    def forward(self, x):
        raise NotImplementedError

    def step(self):
        self.optimizer.step()

    def zero_weights(self):
        self.eval()
        for param in self.parameters():
            param.data = torch.zeros_like(param.data)

    def train_dpsgd(self, data_loader, noise_generator):
        self.optimizer.zero_grad()

        clipped_grads = {name: torch.zeros_like(param)
                         for name, param in self.named_parameters()}

        for batch_x, batch_y in data_loader:
            pred_y = self.forward(batch_x)
            loss = self.loss_fn(pred_y, batch_y)
            # bound l2 sensitivity (gradient clipping)
            # clip each of the gradient in the "Lot"
            for i in range(loss.size()[0]):
                loss[i].backward(retain_graph=True)
                clip_grad_norm_(self.parameters(),
                                max_norm=self.max_norm)
                for name, param in self.named_parameters():
                    clipped_grads[name] += param.grad
                self.zero_grad()

        # add noise
        for name, param in self.named_parameters():
            clipped_grads[name] += noise_generator.get_noise(clipped_grads[name])

        # scale back
        data_size = len(data_loader.sampler)
        for name, param in self.named_parameters():
            clipped_grads[name] /= data_size
        for name, param in self.named_parameters():
            param.grad = clipped_grads[name]

        # update local model
        self.step()

    def copy_a_new(self):
        return self.__class__()
