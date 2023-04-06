from abc import abstractmethod
from typing import List

import torch

from client.base_client import BaseClient
from server.base_server import BaseServer

# from torch.utils.tensorboard import SummaryWriter


class BaseExperiment:

    @abstractmethod
    def get_validation_result(self):
        # This should be the result on validation dataset
        raise NotImplementedError

    @abstractmethod
    def get_test_result(self):
        # This should be thr result on test dataset
        raise NotImplementedError

    @abstractmethod
    def aggeragate(self):
        raise NotImplementedError

    @abstractmethod
    def distribute_model(self):
        for client in self.clients:
            client.model.load_state_dict(self.server.model.state_dict())

    @abstractmethod
    def shuffled_data(self, to_shuffle=False):
        if not to_shuffle:
            return zip(self.clients, self.client_train_datas)
        
        #### IMPLEMENT A SHUFFLE CODE
        
        #return 
    
    @abstractmethod
    def run(self, epochs: int):
        # templated
        for epoch in range(epochs):
            for client, client_train_data in self.shuffled_data():
                client.train(client_train_data)

            self.aggeragate()
            if epoch and not (epoch % 3):
                print(self.get_validation_result())

            self.distribute_model()

