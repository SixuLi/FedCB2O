import argparse
import os
import logging
import torch
import imageio.v2 as imageio

import src.init as init
from src.FedCBO_robustness import FedCBO_NN, FedCBO_Bilevel_NN
from src.init import make_dirs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='CNN_EMNIST', choices=['CNN_EMNIST'])
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--data_name', type=str, default='emnist', choices=['emnist'])
    parser.add_argument('--num_classes', type=int, default=47)
    parser.add_argument('--alg', type=str, default='FedCBO', choices=['FedCBO', 'FedAvg', 'FedCBO_Bilevel', 'test'])

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--p', type=int, default=2, help='Number of clusters.')
    parser.add_argument('--N', type=int, default=100, help='Total number of particles.')
    parser.add_argument('--M', type=int, default=20, help='Number of particles involved in one round.')
    parser.add_argument('--num_participant_per_round', type=int, default=20, help='Num of agents attending in one communication round.')
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--T', type=int, default=100, help='Total time steps.')
    parser.add_argument('--G_func_starting_time', type=int, default=20, help='The starting epoch of using G_func as weight average criterion.')
    parser.add_argument('--Lambda', type=int, default=1)
    parser.add_argument('--Sigma', type=float, default=5)
    parser.add_argument('--Alpha', type=float, default=30)
    parser.add_argument('--Gamma', type=float, default=0.01)
    parser.add_argument('--Beta', type=float, default=0.05)
    parser.add_argument('--Temp', type=float, default=0.25)
    parser.add_argument('--G_func', type=str, default='entropy', choices=['entropy', 'max_diff', 'avg_loss'])
    parser.add_argument('--init_type', type=str, default='uniform', choices=['uniform', 'gaussian'])

    parser.add_argument('--prop_to_full_dataset', type=float, nargs='+', default=[])
    parser.add_argument('--agents_per_cluster', type=int, nargs='+', default=[])
    parser.add_argument('--val_set_prop', type=float, default=0.2)

    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--optimizer', type=str, default='SimpleSGD')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--benign_agent_local_epochs', type=int, default=5)
    parser.add_argument('--mali_agent_local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'ExponentialLR'])
    parser.add_argument('--lr_step_size', type=int, default=10000)
    parser.add_argument('--lr_gamma', type=float, default=1.0)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[1000])
    parser.add_argument('--lr_decay_per_round', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--is_communication', default=False, action='store_true')

    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--no_cuda', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--record_time_freq', type=int, default=10)

    parser.add_argument('--is_oracle', default=False, action='store_true')
    parser.add_argument('--uniform_weight', default=False, action='store_true')
    parser.add_argument('--uniform_weight_with_agent_selection', default=False, action='store_true')
    parser.add_argument('--only_weighted_avg', default=False, action='store_true')
    parser.add_argument('--agent_selection_method', type=str, default='prob', choices=['prob'])
    parser.add_argument('--moving_avg_alpha', type=float, default=0.5)

    parser.add_argument('--adversarial', default=False, action='store_true')
    parser.add_argument('--malicious_prop', type=float, default=0.0)
    parser.add_argument('--num_local_data', type=int, default=500)
    parser.add_argument('--num_mali_local_data', type=int, default=1000)
    parser.add_argument('--adversarial_aug_mali', default=False, action='store_true')
    parser.add_argument('--prop_source_class', type=float, default=1.0)
    parser.add_argument('--remove_adversarial_effect', default=False, action='store_true')
    parser.add_argument('--remove_malicious_agents', default=False, action='store_true')
    parser.add_argument('--fake_malicious_agents', default=False, action='store_true')
    parser.add_argument('--source_class', type=int, default=0)
    parser.add_argument('--target_class', type=int, default=2)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Running training for {}".format(args.experiment_name))
    logging.info("Seed {}".format(args.seed))

    train_init = init.Init(args=args)


    os.environ['CUDA_VISBLE_DEVICES'] = args.gpu_ids
    args.gpu_id_list = [int(s) for s in args.gpu_ids.split(',')]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.alg == 'FedCBO':
        FedCBO = FedCBO_NN(train_init=train_init, args=args)
    elif args.alg == 'FedCBO_Bilevel':
        FedCBO = FedCBO_Bilevel_NN(train_init=train_init, args=args)

    FedCBO.train_with_comm()
    logging.info('Average acc of local agents: {}'.format(torch.sum(FedCBO.store_test_acc) / FedCBO.store_test_acc.size(0)))











