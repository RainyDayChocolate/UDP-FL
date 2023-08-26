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
        self.zero_grad()

        # Create a dictionary to store the sum of clipped gradients for each parameter
        summed_clipped_grads = {name: torch.zeros_like(param)
                                 for name, param in self.named_parameters()}

        # Go through each batch of data
        for batch_x, batch_y in data_loader:
            pred_y = self.forward(batch_x)
            loss = self.loss_fn(pred_y, batch_y)
            if loss.numel() > 1:
                loss = loss.sum()
            # Compute gradients for the entire batch
            loss.backward()

            # Clip gradients
            clip_grad_norm_(self.parameters(), max_norm=self.max_norm)

            # Aggregate clipped gradients
            for name, param in self.named_parameters():
                if param.grad is not None:
                    summed_clipped_grads[name] += param.grad

            # Clear gradients for the next iteration
            self.zero_grad()

        # Calculate the noise to add
        for name, param in self.named_parameters():
            noise = noise_generator.get_noise(summed_clipped_grads[name])
            summed_clipped_grads[name] += noise

        # Average the aggregated gradients
        data_size = len(data_loader.dataset)
        for name, param in self.named_parameters():
            summed_clipped_grads[name] /= data_size 
            param.grad = summed_clipped_grads[name]

        # Update model parameters
        self.step()



# class BaseModel:

#     def forward(self, x):
#         raise NotImplementedError

#     def step(self):
#         self.optimizer.step()

#     def zero_weights(self):
#         self.eval()
#         for param in self.parameters():
#             param.data = torch.zeros_like(param.data)

#     def train_dpsgd(self, data_loader, noise_generator):
#         self.optimizer.zero_grad()

#         clipped_grads = {name: torch.zeros_like(param)
#                          for name, param in self.named_parameters()}

#         for batch_x, batch_y in data_loader:
#             pred_y = self.forward(batch_x)
#             loss = self.loss_fn(pred_y, batch_y)
#             # bound l2 sensitivity (gradient clipping)
#             # clip each of the gradient in the "Lot"
#             for i in range(loss.size()[0]):
#                 loss[i].backward(retain_graph=True)
#                 clip_grad_norm_(self.parameters(),
#                                 max_norm=self.max_norm)
#                 for name, param in self.named_parameters():
#                     clipped_grads[name] += param.grad
#                 self.zero_grad()

#         # add noise
#         for name, param in self.named_parameters():
#             clipped_grads[name] += noise_generator.get_noise(clipped_grads[name])

#         # scale back
#         data_size = len(data_loader.sampler)
#         for name, param in self.named_parameters():
#             clipped_grads[name] /= data_size
#         for name, param in self.named_parameters():
#             param.grad = clipped_grads[name]

#         # update local model
#         self.step()


def copy_a_new(self):
    return self.__class__()
