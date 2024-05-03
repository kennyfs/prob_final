import os
import sys
import json
import threading
import time
import numpy as np
import random
import nltk
np.set_printoptions(threshold=np.inf)

import torch
from ChickenRabbit import ChickenRabbitDataset, eval_split
from GCD import GCDDataset
from torch.utils.data.dataloader import DataLoader
torch.set_printoptions(profile="full")

from mingpt.model_multiplier import GPT
from mingpt.trainer_multiplier import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from itertools import permutations
# -----------------------------------------------------------------------------

def get_config(seed, task, initialization):
    C = CN()

    # system
    C.system = CN()
    # TODO: random seed for model can be set here
    C.system.init_seed = seed # will change the weight initialization
    C.system.work_dir = './test'

    # data
    if task == "gcd":
        C.data = GCDDataset.get_default_config()
    elif task == "ChickenRabbit":
        C.data = ChickenRabbitDataset.get_default_config()
    else:
        raise ValueError(f"task {task} is not supported")

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    C.model.initialization = initialization # "xavier" or "normal"
    
    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.task = task # ChickenRabbit or gcd
    return C

def batch_end_callback(trainer, model, train_dataset, test_dataset):
    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    if trainer.iter_num % 50 == 0:
        # evaluate both the train and test acc
        model.eval()
        with torch.no_grad():
            train_mean = eval_split(trainer.device, model, train_dataset)
            test_mean  = eval_split(trainer.device, model, test_dataset)
        print(f'the mean of train and test are {train_mean}, {test_mean}')
        # save the model and terminate the training
        if test_mean >= 0.9:
            print(f"reach threshold 0.9 in iteration: {trainer.iter_num}")
            print(f"saving model with test_mean: {test_mean}")
            ckpt_path = os.path.join(f"test/{trainer.config.task}", "model_last.pt")
            torch.save(model.state_dict(), ckpt_path)
            return trainer.iter_num
        # revert model to training mode
        model.train()
    return -1

# -----------------------------------------------------------------------------

def run(seed, task, initialization):
    config = get_config(seed, task, initialization)
    setup_logging(config)

    # TODO: try different seed for model
    set_seed(config.system.init_seed)

    # TODO: try different seed to adjust the data order of train/test-set
    if task == "gcd":
        train_dataset = GCDDataset(config.data, split='train', seed=0)
        test_dataset  = GCDDataset(config.data, split='test', seed=0)
    elif task == "ChickenRabbit":
        train_dataset = ChickenRabbitDataset(config.data, split='train', seed=0)
        test_dataset  = ChickenRabbitDataset(config.data, split='test', seed=0)

    # set the correct vocab size: 10, block size: chickenrabbit -> 10, gcd -> 6
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    stop_iteration = trainer.run()
    return stop_iteration
def run_seeds(seeds, start, end, results):
    for i in range(start, end):
        results[i]=run(seeds[i])
if __name__ == '__main__':
    random.seed(sys.argv[3])
    seed=random.randrange(2**64)
    task = sys.argv[1]
    initialization = sys.argv[2]
    stop_iteration = run(seed, task, initialization)
    with open(f"result-{task}-{initialization}.txt", "a") as f:
        f.write(f"{seed}, {stop_iteration}\n")
