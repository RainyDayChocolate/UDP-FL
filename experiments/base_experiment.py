from typing import List


from client.base_client import BaseClient
from server.base_server import BaseServer


class BaseFedSGDExperiment:

    def __init__(self, clients: List[BaseClient], server: BaseServer):
        self.clients = clients
        self.server = server

    def run(self, epochs: int):
        for _ in range(epochs):
            for client in self.clients:
                client.train()
            self.server.aggregate_and_update()


class BaseFedAvgExperiment:

    def __init__(self, clients: List[BaseClient], server: BaseServer):
        self.clients = clients
        self.server = server

    def run(self, epochs: int):
        for _ in range(epochs):
            for client in self.clients:
                client.train()
            self.server.aggeragate_model()
