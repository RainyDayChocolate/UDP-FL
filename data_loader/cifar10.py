import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchvision import datasets, transforms

from utils import CURRENT_FOLDER


# class Cifar10DatasetManager:
#     def __init__(self, train_split=0.8, batch_size=64, n_parties=2, sampling_lot_rate=0.01):
#         torch.manual_seed(0)
#         np.random.seed(0)

#         self.train_transformer = transforms.Compose([
#             transforms.RandomResizedCrop(32),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
        
#         self.valid_test_transformer = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(32),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])

#         self.sampling_rate = sampling_lot_rate
#         self.train_split = train_split
#         self.batch_size = batch_size

#         train_data = datasets.CIFAR10(root=f'{CURRENT_FOLDER}/datasets',
#                                       train=True, download=True, transform=self.train_transformer)
#         test_data = datasets.CIFAR10(root=f'{CURRENT_FOLDER}/datasets',
#                                      train=False, download=True, transform=self.valid_test_transformer)

#         train_data, valid_data = self._split_dataset(train_data)
#         n_parties_train_data = self.split_train_data(train_data, n_parties)

#         self.train_loaders = [self.create_sampling_dataloader(data) for data in n_parties_train_data]
#         self.validation_loader = self._create_dataloader(valid_data)
#         self.test_loader = self._create_dataloader(test_data)

#     def split_train_data(self, train_dataset, n):
#         num_samples = len(train_dataset)
#         sizes = [num_samples // n] * n
#         remainder = num_samples % n
#         for i in range(remainder):
#             sizes[i] += 1
#         return random_split(train_dataset, sizes)

#     def _split_dataset(self, dataset):
#         train_size = int(len(dataset) * self.train_split)
#         validation_size = len(dataset) - train_size
#         return random_split(dataset, [train_size, validation_size])

#     def _create_dataloader(self, dataset):
#         return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#     def _create_sampling_dataloader(self, dataset):
#         num_samples = int(len(dataset) * self.sampling_rate)
#         indices = np.random.choice(len(dataset), num_samples, replace=False)
#         sampler = SubsetRandomSampler(indices)
#         return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)

#     def create_sampling_dataloader(self, dataset):
#         while True:
#             yield self._create_sampling_dataloader(dataset)



class Cifar10DatasetManager:
    def __init__(self, train_split=0.8, batch_size=64, n_parties=2, sampling_lot_rate=0.01):
        # Seed to Shuffle the dataset
        torch.manual_seed(0)
        np.random.seed(0)

        # Data Augmentation for training set
        self.train_transformer = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])
        
        # No augmentation for validation and test sets
        self.valid_test_transformer = transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])

        self.sampling_rate = sampling_lot_rate
        self.train_split = train_split
        self.batch_size = batch_size

        # Load datasets
        train_data = datasets.CIFAR10(root=f'{CURRENT_FOLDER}/datasets',
                                      train=True,
                                      download=True,
                                      transform=self.train_transformer)
        test_data = datasets.CIFAR10(root=f'{CURRENT_FOLDER}/datasets',
                                     train=False,
                                     download=True,
                                     transform=self.valid_test_transformer)

        train_data, valid_data = self._split_dataset(train_data)
        #n_parties_train_data = self.split_train_data(train_data, n_parties)




        self.train_loaders = [self.create_sampling_dataloader(train_data) for _ in range(n_parties)]
        self.validation_loader = self._create_dataloader(valid_data)
        self.test_loader = self._create_dataloader(test_data)

    def split_train_data(self, train_dataset, n):
        num_samples = len(train_dataset)
        sizes = [num_samples // n] * n
        remainder = num_samples % n
        for i in range(remainder):
            sizes[i] += 1
        partitions = random_split(train_dataset, sizes)
        return partitions

    def _split_dataset(self, dataset):
        train_size = int(len(dataset) * self.train_split)
        validation_size = len(dataset) - train_size
        train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
        return train_dataset, validation_dataset

    def _create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _create_sampling_dataloader(self, dataset):
        num_samples = int(len(dataset) * self.sampling_rate)
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        sampled_indices = indices[:num_samples]
    # Create a DataLoader using the SubsetRandomSampler with the sampled_indices
        sampler = SubsetRandomSampler(sampled_indices)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        return data_loader

    def create_sampling_dataloader(self, dataset):
        while True:
            yield self._create_sampling_dataloader(dataset)