import numpy as np
import os
import logging
import logging.config
import logging.handlers
import multiprocessing as mp
import threading
import shutil
import torch

def listener_configurer(logfile):
    logging.basicConfig(filename=logfile,
                        level=logging.INFO,
                        format='%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())

# Lister method for logging using a separate process
def listener_process(logfile, queue):
    listener_configurer(logfile)
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys, traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

# Logging using a separate thread
def logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)

def get_worker_logger(q):
    qh = logging.handlers.QueueHandler(q)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(qh)
    return logger


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
        # Default option is to use another thread for logging. Set to false if
        # you want to use another process for logging.
        self.use_thread_for_logging = True
        self.output_path = os.path.join(args.result_path, args.experiment_name)
        self.checkpoint_path = os.path.join(self.output_path, 'checkpoint.pt')
        self.logfile = os.path.join(self.output_path, 'debug.log')
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
            if os.path.isfile(self.logfile):
                os.remove(self.logfile)

        # Queue setup for mp logging
        self.q = mp.Manager().Queue(-1)
        if self.use_thread_for_logging:
            listener_configurer(self.logfile)  # When using threads

    def start_logger(self):
        if self.use_thread_for_logging:
            self.lp = threading.Thread(target=logger_thread, args=(self.q,))
        else:
            self.lp = mp.Process(target=listener_process, args=(self.logfile, self.q))
        self.lp.start()

    def stop_logger(self):
        self.q.put(None)
        self.lp.join()










