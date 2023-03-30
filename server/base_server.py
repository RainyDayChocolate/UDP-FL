"""Federated Learning can aggregate either models or gradients. Both approaches have their own advantages and trade-offs.

Model Averaging:
In this approach, clients send their local models to the server, and the server aggregates these models by averaging their parameters. This is also known as Federated Averaging (FedAvg).

Pros:

Simple and easy to implement.
Works well when the clients' models have similar structures and are not too different from each other.
Cons:

Not as efficient as gradient aggregation when the clients' models have significant differences.
Can be sensitive to noisy client updates, as a single client's poor update might negatively impact the global model.
Gradient Averaging:
In this approach, clients compute gradients of their local models and send these gradients to the server. The server then averages these gradients and updates the global model accordingly.

Pros:

More efficient when the clients' models have significant differences.
Can potentially reduce the impact of noisy client updates, as the server can apply additional strategies (e.g., gradient clipping) before updating the global model.
Cons:

More complex to implement than model averaging.
Requires clients to compute gradients and send them to the server, which might increase communication overhead.
In practice, both approaches can be used depending on the specific requirements and constraints of the Federated Learning system. In general, model averaging is simpler and more commonly used, while gradient averaging can be more suitable for certain scenarios where clients have significantly different models or when additional control over the aggregation process is needed.

"""
import torch

from typing import List
from client.base_client import BaseClient


class BaseServer:
    def __init__(self, model, clients: List[BaseClient], optimizer=None):
        self.model = model
        self.clients = clients
        self.is_train = True
        if not optimizer:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.optimizer = optimizer

    def set_model_train(self):
        if not self.is_train:
            self.is_train = not self.is_train
            self.model.train()

    def set_model_eval(self):
        if self.is_train:
            self.is_train = not self.is_train
            self.model.eval()

    def aggeragate_model(self):
        importance_sum = sum([c.importance for c in self.clients])
        for client in self.clients:
            importance = client.importance / importance_sum
            for server_param, client_weight in zip(self.model.parameters(), client.weights):
                if server_param.data is None:
                    server_param.data = client_weight.clone() * importance
                    continue
                server_param.data += client_weight * importance

        return self.model

    def aggeragate_gradient(self):
        self.set_model_eval()
        importance_sum = sum([c.importance for c in self.clients])
        for client in self.clients:
            importance = client.importance / importance_sum
            for server_param, client_grad in zip(self.model.parameters(), client.gradients):
                if server_param.grad is None:
                    server_param.grad = client_grad.clone() * importance
                else:
                    server_param.grad += client_grad * importance
        return self.model

    def update_with_gradient(self):
        self.set_model_train()
        self.optimizer.step()
        return self.model

    def aggregate_and_update(self):
        self.aggeragate_gradient()
        self.update_with_gradient()
        return self.model