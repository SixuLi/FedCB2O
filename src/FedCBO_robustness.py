import numpy as np
import logging
import tqdm
import copy
import os
import time
import math
from scipy.stats import entropy

import torch.utils.data
from torch import q_per_channel_scales
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from src import init
from src.model import CNN_EMNIST
from src.dataset import MyEMNIST
from src.utils.util import chunkify, split_list_uneven, AverageMeter

import torch.multiprocessing as mp
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures as cf

def get_optimizer(args, model):
    logging.info('Optimizer is {}'.format(args.optimizer))
    if args.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SimpleSGD':
        return torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                               momentum=args.momentum)
    elif args.optimizer == 'SGD':
        return torch.optim.SGD([
            {'params': model.convLayer1.parameters()},
            {'params': model.convLayer2.parameters()},
            {'params': model.linearLayer1.parameters(), 'weight_decay': 0.004},
            {'params': model.linearLayer2.parameters(), 'weight_decay': 0.004},
            {'params': model.linearLayer3.parameters()}
        ], lr=args.lr, momentum=args.momentum, weight_decay=0.0)
    else:
        raise NotImplementedError

def get_lr_scheduler(args, optimizer):
    logging.info('LR Scheduler is {}'.format(args.lr_scheduler))
    if args.lr_scheduler == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size,
                                               gamma=args.lr_gamma)
    elif args.lr_scheduler == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones,
                                                    gamma=args.lr_gamma)
    elif args.lr_scheduler == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.1)
    else:
        raise NotImplementedError

def local_sgd_function_to_pass_to_mp(data):
    agent_idx, local_model, dataloader, args = data
    if agent_idx in args.benign_agents_indices:
        num_epochs = args.benign_agent_local_epochs
    else:
        num_epochs = args.mali_agent_local_epochs
    local_model.cuda()

    optimizer = get_optimizer(args, local_model)
    lr_scheduler = get_lr_scheduler(args, optimizer)
    local_model.train()

    for epoch in range(1, num_epochs+1):
        tbar = tqdm.tqdm(dataloader)
        loss_logger = AverageMeter()

        # Mini_batch sgd
        for batch_idx, (images, labels) in enumerate(tbar):
            images = images.cuda()
            labels = labels.cuda()

            logits = local_model(images.float())
            loss = F.cross_entropy(logits, labels)
            loss_logger.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
    local_model.cpu()
    return agent_idx, local_model

def mp_evaluate(data):
    # Using model_idx to load the nn we are going to evaluate.
    # Using data_idx to load the dataset we are going to use.
    model_idx, model, data_idx, dataloader, tag, args = data
    logging.info("Eval: model: {} on dataset: {}".format(model_idx, data_idx))
    num_classes = args.num_classes

    model.cuda()
    model.eval()

    total = 0
    correct = 0
    loss_logger = AverageMeter()
    total_target_class = 0
    correct_target_class = 0

    class_correct = [0] * num_classes
    classwise_loss = [0.0] * num_classes
    class_total = [0] * num_classes

    criterion = nn.CrossEntropyLoss(reduction='none')  # Use reduction='none' to get losses for each sample

    for batch_idx, (images, labels) in enumerate(dataloader):
        if args.cuda:
            images = images.cuda()
            labels = labels.cuda()

        logits = model(images.float())
        loss = F.cross_entropy(logits, labels)
        prediction = torch.argmax(logits, dim=1)
        total += images.size(0)
        correct += torch.sum(labels == prediction)
        loss_logger.update(loss.item())

        help_loss = criterion(logits, labels) # Shape: (batch_size, )

        # Get class-wise loss
        for i in range(labels.size(0)):
            classwise_loss[labels[i]] += help_loss[i].item()
            class_total[labels[i]] += 1

        if tag == 'test':
            c = (prediction == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                # class_total[label] += 1
        if args.adversarial:
            # Get the indices of source_class in the batch
            indices_of_source_class = (labels == args.source_class).nonzero(as_tuple=False).squeeze()

            if indices_of_source_class.nelement() == 0:
                continue
            elif len(indices_of_source_class.size()) == 0:
                indices_of_source_class = torch.Tensor(indices_of_source_class)

            images_source_class = images[indices_of_source_class]
            if len(images_source_class.shape) == 3:
                images_source_class = torch.unsqueeze(images_source_class, 0)
            labels_source_class = labels[indices_of_source_class]
            labels_poisoned_target_class = torch.full_like(labels_source_class, args.target_class)

            if args.cuda:
                images_source_class = images_source_class.cuda()
                labels_poisoned_target_class = labels_poisoned_target_class.cuda()

            logits = model(images_source_class)
            prediction = torch.argmax(logits, dim=1)

            # Calculate accuracy
            if len(labels_poisoned_target_class.size()) == 0:
                total_target_class += 1
            else:
                total_target_class += labels_poisoned_target_class.size(dim=0)
            correct_target_class += (prediction == labels_poisoned_target_class).sum().item()

    for i in range(len(classwise_loss)):
        if class_total[i] != 0:
            classwise_loss[i] = classwise_loss[i] / class_total[i]
        else:
            classwise_loss[i] = 0

    if tag == 'test':
        acc_per_class = [100 * x / y if y!=0 else 0 for x, y in zip(class_correct, class_total)]
    else:
        acc_per_class = []
    model.cpu()
    accuracy = 100.0 * correct / total
    if args.adversarial:
        attack_succ_rate = 100.0 * correct_target_class / total_target_class if total_target_class!=0 else 0.0
    else:
        attack_succ_rate = 0.0

    return model_idx, data_idx, accuracy, loss_logger, acc_per_class, attack_succ_rate, classwise_loss





class FedCBO_NN:
    def __init__(self, train_init, args):
        self.train_init = train_init
        self.args = args
        self.p = args.p
        self.N = args.N
        self.M = args.M
        self.T = args.T
        self.num_participant = args.num_participant_per_round
        self.Lambda = args.Lambda
        self.Sigma = args.Sigma
        self.Alpha = args.Alpha
        self.Gamma = args.Gamma
        self.seed = args.seed

        self.data_name = args.data_name
        self.prop_to_full_dataset = args.prop_to_full_dataset
        self.bias = args.bias

        if len(args.agents_per_cluster) == 0:
            self.agents_per_cluster = [int(args.N / args.p)] * args.p
        self.initialization()
        self.setup_datasets()
        self.store_test_acc = torch.zeros(self.N)
        self.store_test_acc_per_class = np.array([[0] * self.args.num_classes] * self.N)
        self.store_attack_succ_rate = torch.zeros(self.N)


    def initialization(self):
        """
        Initialize local NNs.
        """

        # Initialize the models
        if self.args.load_checkpoint:
            checkpoint_path = os.path.join(self.train_init.output_path, 'models.pt')
            checkpoint = torch.load(checkpoint_path)
            self.agents = checkpoint['models']
            self.starting_epoch = checkpoint['epoch'] + 1
            self.args.lr = checkpoint['lr']
            logging.info('starting_epsilon: {}'.format(self.epsilon))
            logging.info('starting_epoch: {}'.format(self.starting_epoch))

            if torch.cuda.is_available():
                for model in self.agents:
                    model.cuda()
        else:
            if self.args.model_name == 'CNN_EMNIST':
                model = CNN_EMNIST(num_classes=self.args.num_classes)
            else:
                raise NotImplementedError

            if torch.cuda.is_available():
                model.cuda()

            self.agents = []
            for _ in range(self.args.N):
                self.agents.append(copy.deepcopy(model))

            self.selected_time = np.zeros((self.N, self.N))
            self.estimate_selection_reward = np.full((self.N, self.N), 0.0)

        self.agents_idx = np.arange(0, self.N, 1)


    def setup_datasets(self):
        np.random.seed(self.args.seed)
        if self.args.adversarial:
            self.num_mali_agents = int(self.args.N * self.args.malicious_prop)
            self.num_benign_agents = self.args.N - self.num_mali_agents
            self.num_total_data = (self.num_benign_agents * self.args.num_local_data + self.num_mali_agents * self.args.num_mali_local_data) // self.args.p
        else:
            self.num_total_data = (self.args.N // self.args.p) * self.args.num_local_data

        # Generate indices for each dataset, also write cluster info

        self.dataset = {}

        train_val_data = []
        test_data = []

        for cluster_idx in range(self.args.p):
            if self.args.p == 1:
                cluster_ninety_rot = 0
            elif self.args.p == 2:
                cluster_ninety_rot = 2 * cluster_idx
            elif self.args.p == 4:
                cluster_ninety_rot = cluster_idx
            else:
                raise NotImplementedError
            if self.args.data_name == 'emnist':
                self.transform = transforms.Compose([transforms.ToTensor()])
                if self.args.adversarial:
                    fullTrainSet = MyEMNIST(args=self.args, train=True, download=True,
                                            transform=self.transform,
                                            times_ninety_rot=cluster_ninety_rot,
                                            source_class=self.args.source_class,
                                            subset_size=self.num_total_data,
                                            seed=self.seed)
                    testset = MyEMNIST(args=self.args, train=False, download=True,
                                       transform=self.transform, times_ninety_rot=cluster_ninety_rot)
                else:
                    fullTrainSet = MyEMNIST(args=self.args, train=True, download=True,
                                            transform=self.transform,
                                            times_ninety_rot=cluster_ninety_rot,
                                            subset_size=self.num_total_data,
                                            seed=self.seed)
                    testset = MyEMNIST(args=self.args, train=False, download=True,
                                       transform=self.transform, times_ninety_rot=cluster_ninety_rot)

            train_val_data.append(fullTrainSet)
            test_data.append(testset)

        train_val_dataset = {}
        train_val_dataset['full_data_indices'], train_val_dataset['cluster_assign'] = \
            self._setup_dataset(len(fullTrainSet), self.args.p, self.args.N)
        train_val_dataset['data'] = train_val_data

        # Associate agent indices to benign and malicious agents
        if self.args.adversarial:
            self.benign_agents_indices_lists, self.malicious_agents_indices_lists = self.separate_agents()
            self.benign_agents_indices = copy.deepcopy(self.benign_agents_indices_lists).reshape(-1)
            self.malicious_agents_indices = copy.deepcopy(self.malicious_agents_indices_lists).reshape(-1)
            self.args.benign_agents_indices = self.benign_agents_indices
            self.args.malicious_agents_indices = self.malicious_agents_indices

            with open(os.path.join(self.train_init.output_path, 'result.txt'), 'a') as f:
                f.write('Indices of malicious agents: ')
                for agent_idx in self.malicious_agents_indices:
                    f.write('{} '.format(agent_idx + 1))
                f.write('\n')
                f.write('Indices of benign agents: ')
                for agent_idx in self.benign_agents_indices:
                    f.write('{} '.format(agent_idx + 1))
                f.write('\n')

        # Train and validation set splitting for each benign agent (including remove proportion of source class data points)
        np.random.seed(self.args.seed)
        train_data_indices = []
        val_data_indices = []
        for agent_idx in self.agents_idx:
            data_indices = train_val_dataset['full_data_indices'][agent_idx]
            cluster_idx = train_val_dataset['cluster_assign'][agent_idx] # Obtain the cluster identity of each agent
            data = train_val_dataset['data'][cluster_idx] # Obtain the data points associated to each agent
            targets = data.targets[data_indices.astype(int)]

            if agent_idx in self.benign_agents_indices:

                # uniformly over each class
                train_idx = None
                val_idx = None
                for i in range(self.args.num_classes):
                    class_i_indices = (targets == i).nonzero(as_tuple=False).squeeze()
                    perturbed_class_i_indices = np.random.permutation(class_i_indices)

                    # Remove 1-prop_source_class proportion of data points in source_class
                    if i == self.args.source_class:
                        perturbed_class_i_indices = perturbed_class_i_indices[:int(len(perturbed_class_i_indices)*self.args.prop_source_class)]
                    if self.args.val_set_prop > 0:
                        val_set_class_i = max(int(self.args.val_set_prop * len(class_i_indices)), 1)
                        train_class_i_idx = perturbed_class_i_indices[:-val_set_class_i]
                        val_class_i_idx = perturbed_class_i_indices[-val_set_class_i:]
                    else:
                        train_class_i_idx = perturbed_class_i_indices
                        val_class_i_idx = np.array([-1])
                    if train_idx is None:
                        train_idx = train_class_i_idx
                        val_idx = val_class_i_idx
                    else:
                        train_idx = np.concatenate((train_idx, train_class_i_idx), axis=None)
                        val_idx = np.concatenate((val_idx, val_class_i_idx), axis=None)

                train_data_indices.append(data_indices[train_idx])
                val_data_indices.append(data_indices[val_idx])

            else:
                train_data_indices.append(data_indices)
                val_data_indices.append(np.array([-1]))

        train_val_dataset['train_data_indices'] = train_data_indices
        train_val_dataset['val_data_indices'] = val_data_indices
        self.dataset['train_val'] = train_val_dataset

        test_dataset = {}
        test_dataset['data'] = test_data
        self.dataset['test'] = test_dataset

        # Create malicious agents
        if self.args.adversarial:
            if self.args.remove_adversarial_effect:
                for agent_idx in self.malicious_agents_indices:
                    self.remove_source_class(agent_idx=agent_idx, source_class_label=self.args.source_class)
                for agent_idx in self.agents_idx:
                    self.check_num_class(agent_idx=agent_idx, class_label=self.args.source_class)
            elif self.args.remove_malicious_agents:
                self.remove_malicious_agents()
                for agent_idx in self.agents_idx:
                    self.check_num_class(agent_idx=agent_idx, class_label=self.args.source_class)
            elif self.args.fake_malicious_agents:
                logging.info('No backdoor attack!!!')
                pass
            else:
                for agent_idx in self.agents_idx:
                    self.check_num_class(agent_idx=agent_idx, class_label=self.args.source_class)
                for agent_idx in self.malicious_agents_indices:
                    self.data_poisoning(agent_idx, from_label=self.args.source_class, to_label=self.args.target_class)
        else:
            self.benign_agents_indices = self.agents_idx
            self.args.benign_agents_indices = self.agents_idx
            self.malicious_agents_indices = np.array([-1])
            self.args.malicious_agents_indices = self.malicious_agents_indices

        self.num_local_train_data_list = np.array([len(self.dataset['train_val']['train_data_indices'][agent_idx]) for agent_idx in self.agents_idx])
        self.num_local_val_data_list = np.array(
            [len(self.dataset['train_val']['val_data_indices'][agent_idx]) for agent_idx in self.agents_idx])
        if self.args.adversarial and self.args.remove_malicious_agents == False:
            for agent_idx in self.malicious_agents_indices:
                self.num_local_train_data_list[agent_idx] = len(self.dataset['train_val']['full_data_indices'][agent_idx])
                self.num_local_val_data_list[agent_idx] = 0
        print('Num local training data list:', self.num_local_train_data_list)
        print('Num local validation data list:', self.num_local_val_data_list)

    def separate_agents(self):
        # Calculate the number of agents per cluster
        agents_per_cluster = self.N // self.p
        malicious_agents_per_cluster = int(agents_per_cluster * self.args.malicious_prop)

        # Initialize lists to store indices of benign and malicious agents
        benign_indices = []
        malicious_indices = []

        # Split agents into p groups and identify benign and malicious agents
        for i in range(self.p):
            start_index = i * agents_per_cluster
            end_index = start_index + agents_per_cluster

            # All agents in the cluster
            cluster_indices = copy.deepcopy(self.agents_idx[start_index:end_index])

            # Mark the malicious agents
            malicious_cluster_indices = cluster_indices[-malicious_agents_per_cluster:]
            malicious_indices.append(malicious_cluster_indices)

            # Remaining are benign
            benign_cluster_indices = cluster_indices[:-malicious_agents_per_cluster]
            benign_indices.append(benign_cluster_indices)

        return np.array(benign_indices), np.array(malicious_indices)

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
                if self.args.adversarial_aug_mali:
                    lst_benign_agents_num_local_data = [self.args.num_local_data] * benign_agents_per_cluster
                    lst_mali_agents_num_local_data = [self.args.num_mali_local_data] * malicious_agents_per_cluster

                    lst_agents_num_local_data = lst_benign_agents_num_local_data + lst_mali_agents_num_local_data
                    overall_lst_agents_num_local_data.append(lst_agents_num_local_data)

                    ll2 = split_list_uneven(ll,
                                            lst_agents_num_local_data)  # Splits ll into agents_per_cluster_list unevenly.

            else:
                ll2 = chunkify(ll, agents_per_cluster)  # splits ll into agents_per_cluster lists with size num_local_data evenly.

            data_indices += ll2
            cluster_assign += [p_i for _ in range(agents_per_cluster)]

        data_indices = np.array(data_indices, dtype=object)
        cluster_assign = np.array(cluster_assign)
        assert data_indices.shape[0] == cluster_assign.shape[0]
        assert data_indices.shape[0] == N

        self.cluster_assign = cluster_assign

        return data_indices, cluster_assign

    def data_poisoning(self, agent_idx, from_label, to_label):
        # 'from_label' is the class of data picked to be poisoned. (source class)
        # 'to_label' is the new label the poisoned data being assigned. (target class)

        from_label = int(from_label)
        to_label = int(to_label)

        cluster_idx = self.dataset['train_val']['cluster_assign'][agent_idx]
        dataset = self.dataset['train_val']
        data_indices = dataset['full_data_indices'][agent_idx]

        data = dataset['data'][cluster_idx]
        targets = data.targets[data_indices]
        targets[targets == from_label] = to_label  # Flip the label
        self.dataset['train_val']['data'][cluster_idx].targets[data_indices] = targets

        logging.info('Check whether label poisoning success.')
        labels = self.dataset['train_val']['data'][cluster_idx].targets[data_indices]
        if len(labels[labels == from_label]) == 0:
            logging.info('Label poisoning success!!!')
        else:
            logging.info('Label poisoning fail!!!')

    def check_num_class(self, agent_idx, class_label):
        cluster_idx = self.dataset['train_val']['cluster_assign'][agent_idx]
        dataset = self.dataset['train_val']
        data_indices = dataset['train_data_indices'][agent_idx].astype(int)

        data = dataset['data'][cluster_idx]
        targets = data.targets[data_indices]

        num_target_class_labels = len(targets[targets == class_label])
        with open(os.path.join(self.train_init.output_path, 'result.txt'), 'a') as f:
            f.write('Agent {} has {} digit {}\n'.format(agent_idx + 1, num_target_class_labels, class_label))

    def remove_source_class(self, agent_idx, source_class_label):
        cluster_idx = self.dataset['train_val']['cluster_assign'][agent_idx]
        dataset = self.dataset['train_val']
        data_indices = dataset['full_data_indices'][agent_idx]

        data = dataset['data'][cluster_idx]

        non_source_class_indices = [data_idx for data_idx in data_indices if
                                    data.targets[data_idx] != source_class_label]

        dataset['full_data_indices'][agent_idx] = np.array(non_source_class_indices)

    def remove_malicious_agents(self):
        self.agents_idx = self.benign_agents_indices
        if self.num_participant > len(self.agents_idx):
            self.num_participant = len(self.agents_idx)
        if self.M >= len(self.agents_idx):
            self.M = len(self.agents_idx) - 1

    def get_agent_dataloader(self, agent_idx, tag='train'):
        cluster_idx = self.dataset['train_val']['cluster_assign'][agent_idx]

        if tag == 'test':
            dataset = self.dataset['test']
            data = dataset['data'][cluster_idx]
            images, targets = data.data, data.targets
            batch_size = self.args.test_batch_size
        else:
            dataset = self.dataset['train_val']
            data = dataset['data'][cluster_idx]
            if self.args.adversarial and agent_idx in self.malicious_agents_indices:
                data_indices = dataset['full_data_indices'][agent_idx]
            else:
                if tag == 'train':
                    data_indices = dataset['train_data_indices'][agent_idx]
                elif tag == 'val':
                    data_indices = dataset['val_data_indices'][agent_idx]
            data_indices = copy.deepcopy(data_indices.astype(int))
            images, targets = data.data[data_indices], data.targets[data_indices]
            batch_size = self.args.batch_size

        local_data = torch.utils.data.TensorDataset(images, targets)
        dataloader = torch.utils.data.DataLoader(local_data, batch_size=batch_size, shuffle=True)

        return dataloader

    def weighted_avg_single_layer(self, thetas, mu, layer):
        # weighted average the parameters for given layer
        avg_weight = None
        avg_bias = None
        with torch.no_grad():
            for j, nn in enumerate(thetas):
                if avg_weight is None:
                    avg_weight = nn.get_layer_weights(layer_num=layer) * mu[j]
                else:
                    avg_weight += nn.get_layer_weights(layer_num=layer) * mu[j]
                if avg_bias is None:
                    avg_bias = nn.get_layer_bias(layer_num=layer) * mu[j]
                else:
                    avg_bias += nn.get_layer_bias(layer_num=layer) * mu[j]
            avg_weight /= torch.sum(mu)
            avg_bias /= torch.sum(mu)
        return (avg_weight, avg_bias)

    def agents_selection_prob(self, agent_idx):
        if np.any(self.selected_time[agent_idx] == 0) == True:
            if len(np.where(self.selected_time[agent_idx] == 0)[0]) >= self.M:
                return np.random.choice(np.where(self.selected_time[agent_idx] == 0)[0], self.M, replace=False)
            else:
                return np.where(self.selected_time[agent_idx] == 0)[0]
        else:
            prob = copy.deepcopy(self.estimate_selection_reward[agent_idx])
            prob /= prob.sum()
            A_t = np.random.choice(self.agents_idx, self.M, replace=False, p=prob)
            return A_t

    def train_with_comm(self):
        if self.args.load_checkpoint:
            t = self.starting_epoch
        else:
            t = 0
        while t < self.T:
            starting_time_one_epoch = time.time()
            if t % 10 == 0:
                logging.info('Training epoch {}'.format(t))

            G_t = np.random.choice(self.agents_idx, self.num_participant, replace=False)

            # Local sgd for selected agent
            starting_local_sgd = time.time()
            numAgentsPassing = len(G_t)
            curr_list_train = [self.get_agent_dataloader(i, tag='train') for i in G_t]

            for local_model in self.agents:
                local_model.cpu()
            # os.chdir('/content/drive/MyDrive/FedCB2O') ### TODO: UPDATE!!!!
            logging.info(os.getcwd())

            with ProcessPoolExecutor(mp_context=mp.get_context('spawn'),
                                     max_workers=4,
                                     initializer=init.get_worker_logger,
                                     initargs=(self.train_init.q,)) as pool:
                results = pool.map(local_sgd_function_to_pass_to_mp,
                                   [(agent_idx, self.agents[agent_idx], curr_list_train[i], self.args) for i, agent_idx in enumerate(G_t)])

            # with mp.get_context('spawn').Pool() as pool:
            #     logging.info('Begin the multiprocessing.')
            #     results = pool.starmap(local_sgd_function_to_pass_to_mp,
            #                            [(agent_idx, self.agents[agent_idx], curr_list_train[i], self.args) for i, agent_idx in
            #                             enumerate(G_t)])
            logging.info('Check whether finished the multi-processing running.')

            for i, local_model in results:
                self.agents[i] = copy.deepcopy(local_model)
                self.agents[i].cuda()
            logging.info('Finished multiprocessing part for {} agents'.format(numAgentsPassing))
            ending_local_sgd = time.time()
            logging.info('total elapsed time for local sgd = {}'.format(ending_local_sgd - starting_local_sgd))

            logging.info(os.getcwd())

            asdj

            # Local aggregation at time step t
            with open(os.path.join(self.train_init.output_path, 'check_state.txt'), 'a') as f:
                f.write('Communication round: {} \n'.format(t))
            with open(os.path.join(self.train_init.output_path, 'loss.txt'), 'a') as f:
                f.write('Communication round: {} \n'.format(t))
            self.local_aggregation(t=t)

            if t % 10 == 0:
                self.save_checkpoint(epoch=t)
            if t % self.args.record_time_freq == 0:
                with open(os.path.join(self.train_init.output_path, 'result.txt'), 'a') as f:
                    f.write('Communication round: {} \n'.format(t))
                    f.write('Average acc of local agents: {} \n'.format(
                        torch.sum(self.store_test_acc) / self.store_test_acc.size(0)))
                    if self.args.adversarial:
                        f.write('Average acc of benign agents: {} \n'.format(
                            torch.sum(self.store_test_acc[self.benign_agents_indices] / self.benign_agents_indices.size)))
                        acc_per_class_benign_agents = [sum(elements) / self.benign_agents_indices.size for elements in
                                                       zip(*self.store_test_acc_per_class[self.benign_agents_indices])]
                        f.write('Average acc of benign agents for each class of labels: \n')
                        for j in range(self.args.num_classes):
                            f.write('Average acc of label {}: {} \n'.format(j, acc_per_class_benign_agents[j]))
                        f.write('\n')
                        f.write(
                            'Average acc of benign agents classify source class {} as poisoned target label {}: {} \n'.format(
                                self.args.source_class, self.args.target_class,
                                torch.sum(self.store_attack_succ_rate[
                                              self.benign_agents_indices] / self.benign_agents_indices.size)))
                        f.write('\n')

                        f.write('Average acc of malicious agents: {} \n'.format(
                            torch.sum(
                                self.store_test_acc[self.malicious_agents_indices] / self.malicious_agents_indices.size)))
                        acc_per_class_malicious_agents = [sum(elements) / self.malicious_agents_indices.size for elements in
                                                          zip(*self.store_test_acc_per_class[
                                                              self.malicious_agents_indices])]
                        f.write('Average acc of malicious agents for each class of labels: \n')
                        for j in range(self.args.num_classes):
                            f.write('Average acc of label {}: {} \n'.format(j, acc_per_class_malicious_agents[j]))
                        f.write('\n')

            self.args.lr *= self.args.lr_decay_per_round
            t += 1
            ending_time_one_epoch = time.time()
            logging.info('The total time for one epoch FedCBO = {}'.format(ending_time_one_epoch - starting_time_one_epoch))

        logging.info("Training finished at epoch {}".format(t))
        with open(os.path.join(self.train_init.output_path, 'result.txt'), 'a') as f:
            f.write('Final result: \n')
            f.write('Average acc of local agents: {} \n'.format(torch.sum(self.store_test_acc) / self.store_test_acc.size(0)))

    def local_aggregation(self, t):
        start_local_agg = time.time()
        eval_list = []
        selected_models = []
        logging.info("Select agents for local aggregation.")

        for agent_idx in self.benign_agents_indices:
            if self.args.uniform_weight or self.args.only_weighted_avg:
                # Random agents selections
                selection_prop = np.ones(len(self.agents_idx)) / (len(self.agents_idx) - 1)
                selection_prop[np.argwhere(self.agents_idx == agent_idx)[0][0]] = 0
                A_t = np.random.choice(self.agents_idx, self.M, replace=False, p=selection_prop)
            else:
                A_t = self.agents_selection_prob(agent_idx=agent_idx)
            # Update number of agents' selected time
            self.selected_time[agent_idx][A_t] += 1

            # Check the correctness of selecting process
            same_cluster_agents = np.where(self.dataset['train_val']['cluster_assign'][A_t] ==
                                           self.dataset['train_val']['cluster_assign'][agent_idx])[0]
            with open(os.path.join(self.train_init.output_path, 'check_state.txt'), 'a') as f:
                f.write('Num of agents in same cluster / Num of selected agents: {}/{}\n'.format(same_cluster_agents.size, A_t.size))

            if np.isin(agent_idx, A_t) == False:
                A_t = np.append(A_t, agent_idx)  # Make sure always pick agent i itself.
            A_t = np.sort(A_t, axis=None)

            selected_models.append(A_t)

            for j in A_t:
                eval_list.append((j, agent_idx))

        # Evaluate other agents' models on the validation set (only for benign agents)
        if self.args.uniform_weight == False:
            # Parallel eval of selected agents on each local dataset
            logging.info('Evaluation of agents on local dataset.')
            curr_list_validation = [self.get_agent_dataloader(i, tag='val') for i in self.benign_agents_indices]
            for local_model in self.agents:
                local_model.cpu()
            os.chdir('/content/drive/MyDrive/FedCB2O') ### TODO: UPDATE!!!!
            logging.info(os.getcwd())

            logging.info('Check multi-processing for model evaluation.')

            with mp.get_context('spawn').Pool() as pool:
                eval_results = pool.starmap(mp_evaluate,
                                            [(model_idx, self.agents[model_idx], dataset_idx,
                                              curr_list_validation[np.argwhere(self.benign_agents_indices == dataset_idx)[0][0]], 'val', self.args)
                                             for model_idx, dataset_idx in eval_list])
            logging.info('Multi-processing for model evaluation successes!!!!')
            for local_model in self.agents:
                local_model.cuda()
            eval_results_dict = [{} for _ in self.benign_agents_indices]
            eval_results_avg_loss = [{} for _ in self.benign_agents_indices]
            eval_results_classwise_loss = [{} for _ in self.benign_agents_indices]

            for model_idx, dataset_idx, acc, loss, _, _, classwise_loss in eval_results:
                idx = np.argwhere(self.benign_agents_indices == dataset_idx)[0][0]
                eval_results_dict[idx][model_idx] = acc, loss, _
                eval_results_avg_loss[idx][model_idx] = loss.avg
                eval_results_classwise_loss[idx][model_idx] = classwise_loss


        # Perform local aggregation below
        cur_agents = []
        for agent_idx in self.agents_idx:
            # Select other agents
            # Assume malicious agents know who are in the same cluster as them and they could collaborate
            if agent_idx in self.malicious_agents_indices:
                cluster_idx = self.dataset['train_val']['cluster_assign'][agent_idx]
                mali_agents_in_same_cluster = self.malicious_agents_indices_lists[cluster_idx]
                benign_agents_in_same_cluster = self.benign_agents_indices_lists[cluster_idx]

                idx = np.where(mali_agents_in_same_cluster == agent_idx)
                mali_agents_in_same_cluster = np.delete(mali_agents_in_same_cluster, idx)
                if self.M >= (len(mali_agents_in_same_cluster) + len(benign_agents_in_same_cluster)):
                    M = len(mali_agents_in_same_cluster) + len(benign_agents_in_same_cluster)
                else:
                    M = self.M
                # Choose the malicious agents in the same clusters as much as possible
                if M > len(mali_agents_in_same_cluster):
                    # If having more budget to download models, randomly choose benign agents in the same clusters
                    A_t = mali_agents_in_same_cluster
                    A_t = np.append(A_t, np.random.choice(benign_agents_in_same_cluster, M-len(A_t), replace=False)).reshape(-1)
                else:
                    A_t = np.random.choice(mali_agents_in_same_cluster, M, replace=False)

                new_A_t = np.append(A_t, agent_idx)
                new_A_t = np.sort(new_A_t, axis=None)

            else:
                j = np.argwhere(self.benign_agents_indices == agent_idx)[0][0]
                new_A_t = selected_models[j]
            thetas = np.take(self.agents, new_A_t, axis=0)
            theta_p = copy.deepcopy(self.agents[agent_idx])

            # Benign agents
            if agent_idx in self.benign_agents_indices:
                agent_vali_loss = []
                if self.args.uniform_weight:
                    mu = torch.from_numpy(copy.deepcopy(self.num_local_train_data_list[new_A_t])).float()
                    mu /= torch.sum(mu)
                else:
                    mu = torch.zeros(new_A_t.size)
                    reward = torch.zeros(new_A_t.size)
                    idx = np.argwhere(self.benign_agents_indices == agent_idx)[0][0]
                    agents_avg_loss = np.array(list(eval_results_avg_loss[idx].values())) # with size new_A_t.size
                    agents_classwise_loss = np.array(list(eval_results_classwise_loss[idx].values()))  # with size num_classes X new_A_t.size

                    for i, model in enumerate(thetas):
                        validation_acc, validation_loss, _ = eval_results_dict[idx][new_A_t[i]]
                        val_loss = validation_loss.avg
                        agent_vali_loss.append(val_loss)
                        reward_i = math.exp(-val_loss / self.args.Temp)
                        reward[i] = reward_i

                        if self.estimate_selection_reward[agent_idx][new_A_t[i]] == 0.0:
                            self.estimate_selection_reward[agent_idx][new_A_t[i]] = reward_i
                        else:
                            self.estimate_selection_reward[agent_idx][new_A_t[i]] \
                                = self.args.moving_avg_alpha * reward_i + (1 - self.args.moving_avg_alpha) * \
                                              self.estimate_selection_reward[agent_idx][new_A_t[i]]
                        mu_i = torch.exp(torch.tensor([-self.Alpha]) * val_loss)
                        mu[i] = mu_i
                    mu /= torch.sum(mu)

                    with open(os.path.join(self.train_init.output_path, 'loss.txt'), 'a') as f:
                        f.write('Agent {} selected agent: {}\n'.format(agent_idx + 1, new_A_t + 1))
                        if self.args.adversarial:
                            f.write('The avg validation loss of dataset from agent {}: {}\n'.format(agent_idx + 1,
                                                                                                agents_avg_loss.tolist()))
                            f.write('The classwise loss of dataset from agent {}: {}\n'.format(agent_idx + 1,
                                                                                           agents_classwise_loss.tolist()))
            # Malicious agents
            else:
                mu = torch.from_numpy(copy.deepcopy(self.num_local_train_data_list[new_A_t])).float()
                mu /= torch.sum(mu)

            with open(os.path.join(self.train_init.output_path, 'check_state.txt'), 'a') as f:
                f.write('agent {} mu = {}\n'.format(agent_idx + 1, mu))
            if self.args.cuda:
                mu = mu.cuda()

            # Local aggregation layer by layer
            for i in range(1, theta_p.num_layers + 1):
                # calculate the consensus point for layer i
                mu_p_weight, mu_p_bias = self.weighted_avg_single_layer(thetas=thetas, mu=mu, layer=i)

                # Update theta_p for layer i
                target_weight = theta_p.get_layer_weights(layer_num=i)
                target_bias = theta_p.get_layer_bias(layer_num=i)

                target_weight.data = mu_p_weight
                target_bias.data = mu_p_bias
            cur_agents.append(theta_p)

        if t % self.args.record_time_freq == 0:
            # The number of agents to be passed could be adjusted here depending on the size of dataset
            logging.info("Evaluation of agents on test set")
            curr_list_test = [self.get_agent_dataloader(i, tag='test') for i in self.agents_idx]
            for local_model in cur_agents:
                local_model.cpu()
            os.chdir('/content/drive/MyDrive/FedCB2O')
            logging.info(os.getcwd())
            with mp.get_context('spawn').Pool(processes=5) as pool:
                eval_results = pool.starmap(mp_evaluate,
                                            [(i, cur_agents[i], i, curr_list_test[i], 'test', self.args)
                                             for i in range(len(cur_agents))])
            for local_model in cur_agents:
                local_model.cuda()

            for i, (agent_idx, _, test_acc, test_loss, acc_per_class, attack_succ_rate, _) in enumerate(eval_results):
                # Update models after local aggregation.
                self.agents[agent_idx] = copy.deepcopy(cur_agents[i])
                # Store the test accuracy.
                self.store_test_acc[agent_idx] = test_acc
                self.store_test_acc_per_class[agent_idx] = acc_per_class
                self.store_attack_succ_rate[agent_idx] = attack_succ_rate
                logging.info('Agent {} acc test: {}'.format(agent_idx + 1, test_acc))
        else:
            for i, agent_idx in enumerate(self.agents_idx):
                self.agents[agent_idx] = copy.deepcopy(cur_agents[i])

        end_local_agg = time.time()
        print('total elapsed time for local agg = ', end_local_agg - start_local_agg)

        with open(os.path.join(self.train_init.output_path, 'reward.txt'), 'a') as f:
            f.write('Communication round {} \n'.format(t))
            for agent_idx in self.agents_idx:
                f.write('Estimated selection reward of agent {}: {}\n'.format(agent_idx+1, self.estimate_selection_reward[agent_idx]))
                f.write('Selection time of agent {}: {}\n'.format(agent_idx+1, self.selected_time[agent_idx]))

    def save_checkpoint(self, epoch):
        selected_time = torch.from_numpy(self.selected_time)
        estimate_selection_reward = torch.from_numpy(self.estimate_selection_reward)
        torch.save({'models': self.agents,
                        'estimate_selection_reward': estimate_selection_reward,
                        'selected_time': selected_time,
                        'epoch': epoch,
                        'lr': self.args.lr},
                        os.path.join(self.train_init.output_path, 'models.pt'))


class FedCBO_Bilevel_NN(FedCBO_NN):
    def __init__(self, train_init, args):
        super().__init__(train_init, args)
        self.Beta = args.Beta
        self.max_diff = np.full((self.N, self.N), 0.0)

    def local_aggregation(self, t):
        start_local_agg = time.time()
        eval_list = []
        selected_models = []
        logging.info("Select agents for local aggregation.")

        for agent_idx in self.benign_agents_indices:
            if self.args.uniform_weight or self.args.only_weighted_avg:
                # Random agents selections
                selection_prop = np.ones(len(self.agents_idx)) / (len(self.agents_idx) - 1)
                idx = np.argwhere(self.agents_idx == agent_idx)[0][0]
                selection_prop[idx] = 0
                A_t = np.random.choice(self.agents_idx, self.M, replace=False, p=selection_prop)
            else:
                A_t = self.agents_selection_prob(agent_idx=agent_idx)

            # Update number of agents' selected time
            self.selected_time[agent_idx][A_t] += 1

            # Check the correctness of selecting process
            same_cluster_agents = np.where(self.dataset['train_val']['cluster_assign'][A_t] ==
                                           self.dataset['train_val']['cluster_assign'][agent_idx])[0]
            with open(os.path.join(self.train_init.output_path, 'check_state.txt'), 'a') as f:
                f.write('Num of agents in same cluster / Num of selected agents: {}/{}\n'.format(same_cluster_agents.size, A_t.size))

            if agent_idx not in A_t:
                A_t = np.append(A_t, agent_idx)  # Make sure always pick agent i itself.
            A_t = np.sort(A_t, axis=None)

            selected_models.append(A_t)

            for j in A_t:
                eval_list.append((j, agent_idx))

        # Evaluate other agents' models on the validation set (only for benign agents)
        if self.args.uniform_weight == False:
            # Parallel eval of selected agents on each local dataset
            logging.info('Evaluation of agents on local dataset.')
            curr_list_validation = [self.get_agent_dataloader(i, tag='val') for i in self.benign_agents_indices]
            for local_model in self.agents:
                local_model.cpu()
            os.chdir('/content/drive/MyDrive/FedCB2O') ### TODO: UPDATE!!!!
            logging.info(os.getcwd())

            logging.info('Check multi-processing for model evaluation.')

            with mp.get_context('spawn').Pool() as pool:
                eval_results = pool.starmap(mp_evaluate,
                                            [(model_idx, self.agents[model_idx], dataset_idx,
                                              curr_list_validation[np.argwhere(self.benign_agents_indices == dataset_idx)[0][0]], 'val', self.args)
                                             for model_idx, dataset_idx in eval_list])
            logging.info('Multi-processing for model evaluation successes!!!!')
            for local_model in self.agents:
                local_model.cuda()
            eval_results_dict = [{} for _ in self.benign_agents_indices]
            eval_results_avg_loss = [{} for _ in self.benign_agents_indices]
            eval_results_classwise_loss = [{} for _ in self.benign_agents_indices]

            for model_idx, dataset_idx, acc, loss, _, _, classwise_loss in eval_results:
                idx = np.argwhere(self.benign_agents_indices == dataset_idx)[0][0]
                eval_results_dict[idx][model_idx] = acc, loss, _
                eval_results_avg_loss[idx][model_idx] = loss.avg
                eval_results_classwise_loss[idx][model_idx] = classwise_loss

        # Perform local aggregation below
        cur_agents = []
        for agent_idx in self.agents_idx:
            # Select other agents
            # Assume malicious agents know who are in the same cluster as them and they could collaborate
            if agent_idx in self.malicious_agents_indices:
                cluster_idx = self.dataset['train_val']['cluster_assign'][agent_idx]
                mali_agents_in_same_cluster = self.malicious_agents_indices_lists[cluster_idx]
                benign_agents_in_same_cluster = self.benign_agents_indices_lists[cluster_idx]

                idx = np.where(mali_agents_in_same_cluster == agent_idx)
                mali_agents_in_same_cluster = np.delete(mali_agents_in_same_cluster, idx)
                if self.M >= (len(mali_agents_in_same_cluster) + len(benign_agents_in_same_cluster)):
                    M = len(mali_agents_in_same_cluster) + len(benign_agents_in_same_cluster)
                else:
                    M = self.M
                # Choose the malicious agents in the same clusters as much as possible
                if M > len(mali_agents_in_same_cluster):
                    # If having more budget to download models, randomly choose benign agents in the same clusters
                    A_t = mali_agents_in_same_cluster
                    A_t = np.append(A_t, np.random.choice(benign_agents_in_same_cluster, M - len(A_t),
                                                          replace=False)).reshape(-1)
                else:
                    A_t = np.random.choice(mali_agents_in_same_cluster, M, replace=False)

                new_A_t = np.append(A_t, agent_idx)
                new_A_t = np.sort(new_A_t, axis=None)
            else:
                j = np.argwhere(self.benign_agents_indices == agent_idx)[0][0]
                new_A_t = selected_models[j]
            theta_p = copy.deepcopy(self.agents[agent_idx])

            # Benign agents
            if agent_idx in self.benign_agents_indices:
                idx = np.argwhere(self.benign_agents_indices == agent_idx)[0][0]
                agents_avg_loss = np.array(list(eval_results_avg_loss[idx].values())) # with size new_A_t.size
                agents_classwise_loss = np.array(list(eval_results_classwise_loss[idx].values())) # with size new_A_t.size X num_classes

                # Update reward
                agent_vali_loss = []
                reward = torch.zeros(new_A_t.size)
                for i, _ in enumerate(new_A_t): # Could potentially optimize to remove the for loop
                    val_acc, val_loss, _ = eval_results_dict[idx][new_A_t[i]]
                    val_loss = val_loss.avg
                    agent_vali_loss.append(val_loss)
                    reward_i = math.exp(-val_loss / self.args.Temp)
                    reward[i] = reward_i

                    if self.estimate_selection_reward[agent_idx][new_A_t[i]] == 0.0:
                        self.estimate_selection_reward[agent_idx][new_A_t[i]] = reward_i
                    else:
                        self.estimate_selection_reward[agent_idx][new_A_t[i]] \
                            = self.args.moving_avg_alpha * reward_i + (1 - self.args.moving_avg_alpha) * \
                                self.estimate_selection_reward[agent_idx][new_A_t[i]]

                if t < self.args.G_func_starting_time: # Try to "utilize" the malicious agent to get faster convergence
                    # Record the maximum difference for class-wise loss (but not using it for weight average criterion)
                    idx = np.argwhere(new_A_t == agent_idx)[0][0]
                    cur_agent_classwise_loss = agents_classwise_loss[idx]  # size: num_classes
                    cur_agent_classwise_loss = np.tile(cur_agent_classwise_loss[np.newaxis, :], (len(new_A_t), 1))
                    max_difference = np.max(agents_classwise_loss - cur_agent_classwise_loss, axis=1)

                    # Record the maximum difference for class-wise loss
                    self.max_diff[agent_idx][new_A_t] = max_difference
                    with open(os.path.join(self.train_init.output_path, 'loss.txt'), 'a') as f:
                        f.write('Agent {} selected agent: {}\n'.format(agent_idx, new_A_t))
                        f.write('Maximum difference class-wise loss evaluating on dataset from agent {}: {}\n'.format(
                            agent_idx + 1, np.round(self.max_diff[agent_idx], decimals=4).tolist()))

                    # Use average loss as weighted average criterion before G_func_starting_time
                    weighted_avg_criterion = agents_avg_loss
                else:
                    # Compute weighted average based on class-wise loss
                    if self.args.G_func == 'entropy':
                        weighted_avg_criterion = -entropy(agents_classwise_loss[new_A_t], axis=1) # with quantile_selected_agents_idx.size

                        # Record the entropy of class-wise loss.
                        with open(os.path.join(self.train_init.output_path, 'loss.txt'), 'a') as f:
                            f.write('The negative entropy of class-wise loss of agents being selected: {}\n'.format(np.round(weighted_avg_criterion, decimals=4).tolist()))

                    elif self.args.G_func == 'max_diff':
                        idx = np.argwhere(new_A_t == agent_idx)[0][0]
                        cur_agent_classwise_loss = agents_classwise_loss[idx] # size: num_classes
                        cur_agent_classwise_loss = np.tile(cur_agent_classwise_loss[np.newaxis, :], (len(new_A_t), 1))
                        weighted_avg_criterion = np.max(agents_classwise_loss - cur_agent_classwise_loss, axis=1)
                        # Record the maximum difference for class-wise loss
                        self.max_diff[agent_idx][new_A_t] = weighted_avg_criterion
                        with open(os.path.join(self.train_init.output_path, 'loss.txt'), 'a') as f:
                            f.write('Agent {} selected agent: {}\n'.format(agent_idx, new_A_t))
                            f.write('Maximum difference class-wise loss evaluating on dataset from agent {}: {}\n'.format(
                                agent_idx + 1, np.round(self.max_diff[agent_idx], decimals=4).tolist()))

                    elif self.args.G_func == 'avg_loss':
                        weighted_avg_criterion = agents_avg_loss

                weighted_avg_criterion -= np.min(weighted_avg_criterion)
                mu = torch.exp(torch.tensor([-self.Alpha]) * torch.from_numpy(weighted_avg_criterion))

                if t >= self.args.G_func_starting_time:
                    # Adjust the weight giving to agent i itself.
                    idx = np.argwhere(new_A_t == agent_idx)[0][0]
                    mu[idx] = 0.0
                    mu[idx] = torch.max(mu)

                mu /= torch.sum(mu)

                if self.args.cuda:
                    mu = mu.cuda()

                # Local aggregation layer by layer
                thetas = np.take(self.agents, new_A_t, axis=0)
                for i in range(1, theta_p.num_layers + 1):
                    # calculate the consensus point for layer i
                    mu_p_weight, mu_p_bias = self.weighted_avg_single_layer(thetas=thetas, mu=mu, layer=i)

                    # Update theta_p for layer i
                    target_weight = theta_p.get_layer_weights(layer_num=i)
                    target_bias = theta_p.get_layer_bias(layer_num=i)

                    target_weight.data = mu_p_weight
                    target_bias.data = mu_p_bias

            # Malicious agents
            else:
                mu = torch.from_numpy(copy.deepcopy(self.num_local_train_data_list[new_A_t])).float()
                mu /= torch.sum(mu)
                if self.args.cuda:
                    mu = mu.cuda()

                # Local aggregation layer by layer
                thetas = np.take(self.agents, new_A_t, axis=0)
                for i in range(1, theta_p.num_layers + 1):
                    # calculate the consensus point for layer i
                    mu_p_weight, mu_p_bias = self.weighted_avg_single_layer(thetas=thetas, mu=mu, layer=i)

                    # Update theta_p for layer i
                    target_weight = theta_p.get_layer_weights(layer_num=i)
                    target_bias = theta_p.get_layer_bias(layer_num=i)

                    target_weight.data = mu_p_weight
                    target_bias.data = mu_p_bias

            cur_agents.append(theta_p)

            with open(os.path.join(self.train_init.output_path, 'check_state.txt'), 'a') as f:
                f.write('agent {} mu = {}\n'.format(agent_idx + 1, mu))

        if t % self.args.record_time_freq == 0:
            # The number of agents to be passed could be adjusted here depending on the size of dataset
            logging.info("Evaluation of agents on test set")
            curr_list_test = [self.get_agent_dataloader(i, tag='test') for i in self.agents_idx]
            for local_model in cur_agents:
                local_model.cpu()
            os.chdir('/content/drive/MyDrive/FedCB2O')
            logging.info(os.getcwd())
            with mp.get_context('spawn').Pool(processes=5) as pool:
                eval_results = pool.starmap(mp_evaluate,
                                            [(i, cur_agents[i], i, curr_list_test[i], 'test', self.args)
                                             for i in range(len(cur_agents))])
            for local_model in cur_agents:
                local_model.cuda()

            for i, (agent_idx, _, test_acc, test_loss, acc_per_class, attack_succ_rate, _) in enumerate(eval_results):
                # Update models after local aggregation.
                self.agents[agent_idx] = copy.deepcopy(cur_agents[i])
                # Store the test accuracy
                self.store_test_acc[agent_idx] = test_acc
                self.store_test_acc_per_class[agent_idx] = acc_per_class
                self.store_attack_succ_rate[agent_idx] = attack_succ_rate
                logging.info('Agent {} acc test: {}'.format(agent_idx + 1, test_acc))

        else:
            for i, agent_idx in enumerate(self.agents_idx):
                self.agents[agent_idx] = copy.deepcopy(cur_agents[i])

        end_local_agg = time.time()
        print('total elapsed time for local agg = ', end_local_agg - start_local_agg)

        with open(os.path.join(self.train_init.output_path, 'reward.txt'), 'a') as f:
            f.write('Communication round {} \n'.format(t))
            for agent_idx in self.agents_idx:
                f.write('Estimated selection reward of agent {}: {}\n'.format(agent_idx + 1,
                                                                              self.estimate_selection_reward[
                                                                                  agent_idx]))
                f.write('Selection time of agent {}: {}\n'.format(agent_idx + 1, self.selected_time[agent_idx]))






































