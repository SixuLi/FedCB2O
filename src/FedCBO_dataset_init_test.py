import numpy as np
import logging
import tqdm
import copy
from sklearn.model_selection import train_test_split
import os
import time
import jenkspy
from scipy.stats import entropy

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils import data
from torch.utils.data import Subset
from torchvision import datasets, transforms
from PIL import Image
from typing import Tuple

from src.model import CNN_CIFAR10
from src.dataset import RotatedCIFAR10, CustomDataset, MyCIFAR10
from src.FedCBO_robustness import FedCBO_NN
from src.utils.util import chunkify, split_list_uneven, AverageMeter

class FedCBO_Data_Init_Test(FedCBO_NN):
    def __init__(self, train_init, args):
        super().__init__(train_init, args)

        self.initialization()
        self.setup_datasets()

    def setup_datasets(self):
        np.random.seed(self.args.seed)

        # Generate indices for each dataset, also write cluster info

        self.dataset = {}

        train_val_data = []
        test_data = []

        for cluster_idx in range(self.args.p):
            if self.args.data_name == 'cifar10':
                mean_cifar = (0.485, 0.456, 0.406)
                std_cifar = (0.229, 0.224, 0.225)

                if self.args.p == 1:
                    cluster_ninety_rot = 0
                elif self.args.p == 2:
                    cluster_ninety_rot = 2 * cluster_idx
                elif self.args.p == 4:
                    cluster_ninety_rot = cluster_idx
                else:
                    raise NotImplementedError

                self.train_transform = transforms.Compose([
                    transforms.RandomCrop(size=(24, 24)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_cifar, std_cifar)
                ])
                self.test_transform = transforms.Compose([
                    transforms.RandomCrop(size=(24, 24)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean_cifar, std_cifar)
                ])
                if self.args.adversarial:
                    self.num_mali_agents = int(self.args.N * self.args.malicious_prop)
                    self.num_benign_agents = self.args.N - self.num_mali_agents
                    self.num_total_data = self.num_benign_agents * self.args.num_local_data + self.num_mali_agents * self.args.num_mali_local_data
                    fullTrainSet = MyCIFAR10(args=self.args, train=True, download=True,
                                             transform=self.train_transform,
                                             times_ninety_rot=cluster_ninety_rot,
                                             source_class=self.args.source_class,
                                             prop_source_class=self.args.prop_source_class,
                                             subset_size=self.num_total_data,
                                             seed=self.seed)
                    testset = MyCIFAR10(args=self.args, train=False, download=True,
                                        transform=self.test_transform, times_ninety_rot=cluster_ninety_rot)
                else:
                    fullTrainSet = RotatedCIFAR10(self.args.data_path, train=True, download=True,
                                                  transform=self.train_transform,
                                                  times_ninety_rot=cluster_ninety_rot,
                                                  prop_to_full_dataset=self.args.prop_to_full_dataset[0])
                    testset = RotatedCIFAR10(self.args.data_path, train=False, download=True,
                                             transform=self.test_transform,
                                             times_ninety_rot=cluster_ninety_rot)

            train_val_data.append(fullTrainSet)
            test_data.append(testset)

        if not self.args.adversarial:
            self.num_local_data = len(fullTrainSet) * self.args.p // self.args.N

        train_val_dataset = {}
        train_val_dataset['full_data_indices'], train_val_dataset['cluster_assign'] = (
            self._setup_dataset(len(fullTrainSet), self.p, self.N))
        train_val_dataset['data'] = train_val_data


        # Train and validation set splitting for each agent.
        np.random.seed(self.args.seed)
        train_data_indices = []
        val_data_indices = []
        for agent_idx, data_indices in enumerate(train_val_dataset['full_data_indices']):
            perturbed_data_indices = np.random.permutation(data_indices)
            # print('full_data_idx of agent {}: {}'.format(agent_idx+1, perturbed_data_indices))
            val_set_size = int(self.args.val_set_prop * len(data_indices))
            train_data_idx = perturbed_data_indices[:-val_set_size]
            val_data_idx = perturbed_data_indices[-val_set_size:]
            # print('Training data indices of agent {}: {}'.format(agent_idx + 1, train_data_idx))
            # print('Validation data indices of agent {}: {}'.format(agent_idx + 1, val_data_idx))
            train_data_indices.append(train_data_idx)
            val_data_indices.append(val_data_idx)
        # print('Train_data_indices:', train_data_indices)
        # print('Val_data_indices:', val_data_indices)
        train_val_dataset['train_data_indices'] = train_data_indices
        train_val_dataset['val_data_indices'] = val_data_indices
        self.dataset['train_val'] = train_val_dataset

        # for agent_idx in self.agents_idx:
        #     print('Num of training data of agent {}: {}'.format(agent_idx+1, len(self.dataset['train_val']['train_data_indices'][agent_idx])))
        #     print('Num of validation data of agent {}: {}'.format(agent_idx + 1,
        #                                                          len(self.dataset['train_val']['val_data_indices'][agent_idx])))


    def _setup_dataset(self, num_data, p, N, random=True):
        data_indices = []
        cluster_assign = []
        agents_per_cluster = N // p
        malicious_agents_per_cluster = int(agents_per_cluster * self.args.malicious_prop)
        benign_agents_per_cluster = agents_per_cluster - malicious_agents_per_cluster

        overall_lst_agents_num_local_data = []

        for p_i in range(p):
            if random:
                ll = list(np.random.permutation(num_data))
            else:
                ll = list(range(num_data))

            if self.args.adversarial_aug_mali:
                lst_benign_agents_num_local_data = [self.args.num_local_data] * benign_agents_per_cluster
                lst_mali_agents_num_local_data = [self.args.num_mali_local_data] * malicious_agents_per_cluster

                lst_agents_num_local_data = lst_benign_agents_num_local_data + lst_mali_agents_num_local_data
                overall_lst_agents_num_local_data.append(lst_agents_num_local_data)

                ll2 = split_list_uneven(ll, lst_agents_num_local_data) # Splits ll into agents_per_cluster_list unevenly.

            else:
                ll2 = chunkify(ll, agents_per_cluster) # splits ll into agents_per_cluster lists with size num_local_data evenly.

            data_indices += ll2
            cluster_assign += [p_i for _ in range(agents_per_cluster)]

        data_indices = np.array(data_indices, dtype=object)
        cluster_assign = np.array(cluster_assign)
        assert data_indices.shape[0] == cluster_assign.shape[0]
        assert data_indices.shape[0] == N

        self.cluster_assign = cluster_assign

        return data_indices, cluster_assign


















