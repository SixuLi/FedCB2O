import numpy as np
import os
import logging
import shutil

import torch


def generate_data(data_name, num_data, args):
    np.random.seed(args.seed)

    if data_name == '1d_data':
        data = np.random.normal(loc=0, scale=0.1, size=num_data)

        np.savez(os.path.join(args.data_path, args.data_name + '.npz'), data=data)

def make_dirs(dirname, rm=False):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif rm:
        logging.info('Rm and mkdir {}'.format(dirname))
        shutil.rmtree(dirname)
        os.makedirs(dirname)

class Init:
    def __init__(self, args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.output_path = os.path.join(args.result_path, args.experiment_name)
        if args.data_name in ['synthetic_data']:
            if args.seed == 0:
                make_dirs(args.result_path)
                make_dirs(args.data_path)
                make_dirs(self.output_path)
                args_state = {k: v for k, v in args._get_kwargs()}
                with open(os.path.join(self.output_path, 'result.txt'), 'w') as f:
                    print(args_state, file=f)
                with open(os.path.join(self.output_path, 'check_state.txt'), 'w') as f:
                    print(args_state, file=f)
                # with open(os.path.join(self.output_path, 'CBO_result.txt'), 'w') as f:
                #     print(args_state, file=f)
        elif args.data_name in ['cifar10', 'emnist']:
            if not args.load_checkpoint:
                make_dirs(args.result_path)
                make_dirs(args.data_path)
                make_dirs(self.output_path)
                args_state = {k: v for k, v in args._get_kwargs()}
                with open(os.path.join(self.output_path, 'result.txt'), 'w') as f:
                    print(args_state, file=f)
                with open(os.path.join(self.output_path, 'check_state.txt'), 'w') as f:
                    print(args_state, file=f)
                with open(os.path.join(self.output_path, 'loss.txt'), 'w') as f:
                    print(args_state, file=f)
                with open(os.path.join(self.output_path, 'reward.txt'), 'w') as f:
                    print(args_state, file=f)










