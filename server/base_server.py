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
from typing import List
from client.base_client import BaseClient
from models.base_model import BaseModel


class BaseServer:
    def __init__(self,
                 model: BaseModel):
        self.model = model

    def aggeragate_model(self, clients):
        self.model.zero_weights()
        importance_sum = sum([c.importance for c in clients])
        for client in clients:
            importance = client.importance / importance_sum
            for server_param, client_weight in zip(self.model.parameters(), client.weights):
                server_param.data += client_weight * importance

        return self.model

    def aggeragate_gradient(self, clients):
        self.model.train()
        self.model.zero_grad()
        importance_sum = sum([c.importance for c in clients])
        for client in clients:
            importance = client.importance / importance_sum
            for server_param, client_grad in zip(self.model.parameters(), client.gradients):
                if server_param.grad is None:
                    server_param.grad = client_grad.clone() * importance
                else:
                    server_param.grad += client_grad * importance
        return self.model

    # def aggregate_gradient(self, clients):
    #     self.model.train()
    #     self.model.zero_grad()
        
    #     importance_sum = sum([c.importance for c in clients])
        
    #     for client in clients:
    #         importance = client.importance / importance_sum
    #         client_grads = client.gradients
            
    #         # Check if the client's gradients match the model's parameters
    #         if len(list(self.model.parameters())) != len(client_grads):
    #             raise ValueError("Mismatch between client gradients and model parameters.")
            
    #         for server_param, client_grad in zip(self.model.parameters(), client_grads):
    #             if server_param.shape != client_grad.shape:
    #                 raise ValueError("Mismatch in shape between server parameters and client gradients.")
                
    #             if server_param.grad is None:
    #                 server_param.grad = client_grad.clone() * importance
    #             else:
    #                 server_param.grad += client_grad * importance

    # No need to return self.model as changes are in-place


    def update_with_gradient(self):
        self.model.train()
        self.model.step()
        return self.model

    def aggregate_grad_update(self, clients):
        self.aggeragate_gradient(clients)
        self.update_with_gradient()
        return self.model

    def predict(self, inputs):
        self.model.eval()
        return self.model(inputs)
