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
def add_noise(seed, training_data):
    # training data: a 2D list, shape: [rows, 6]
    rng = random.Random(seed)
    for i in range(1, len(training_data), 2):
        # swap the two rows
        if rng.random() < 0.5:
            training_data[i], training_data[i-1] = training_data[i-1], training_data[i]
    for i in range(2, len(training_data), 2):
        # swap the two rows
        if rng.random() < 0.5:
            training_data[i], training_data[i-1] = training_data[i-1], training_data[i]
    return training_data

def shuffle(seed, training_data):
    training_data = training_data.tolist()
    random.Random(seed).shuffle(training_data)
    return torch.tensor(training_data, dtype=torch.long)

def chickrab_ab_with_batch_fifty(seed, training_data):
    training_data = training_data.tolist()
    for i in range(0, len(training_data), 50):
            training_data[i:i+50] = sorted(training_data[i:i+50], key=lambda x: (x[0]*100+x[1]*10+x[2])*(x[3]*100+x[4]*10+x[5]))
    training_data = add_noise(seed, training_data)
    return torch.tensor(training_data, dtype=torch.long)
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

def run(seed, dataseed, task, initialization, data_rearrange_fn):
    config = get_config(seed, task, initialization)
    setup_logging(config)

    # TODO: try different seed for model
    set_seed(config.system.init_seed)

    # TODO: try different seed to adjust the data order of train/test-set
    if task == "gcd":
        train_dataset = GCDDataset(config.data, split='train', seed=dataseed) # [row,6]
        train_dataset.ixes = data_rearrange_fn(dataseed, train_dataset.ixes)
        test_dataset  = GCDDataset(config.data, split='test', seed=dataseed)
    elif task == "ChickenRabbit":
        train_dataset = ChickenRabbitDataset(config.data, split='train', seed=dataseed) # [row,8]
        train_dataset.ixes = data_rearrange_fn(dataseed, train_dataset.ixes)
        test_dataset  = ChickenRabbitDataset(config.data, split='test', seed=dataseed)

    # set the correct vocab size: 10, block size: chickenrabbit -> 10, gcd -> 6
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)
    stop_iteration = trainer.run()
    return stop_iteration
if __name__ == '__main__':
    task = sys.argv[1]
    initialization = sys.argv[2]
    seed=int(sys.argv[3])
    print(f"get seed {seed}")
    dataset_seed = int(sys.argv[4])
    print(f"get dataset seed {dataset_seed}")
    rearrange_fn = chickrab_ab_with_batch_fifty
    print(f"rearrange function: {rearrange_fn.__name__}")
    stop_iteration = run(seed, dataset_seed, task, initialization, rearrange_fn)
    with open(f"result-{rearrange_fn.__name__}-{task}.csv", "a") as f:
        f.write(f"{seed}, {dataset_seed}, {stop_iteration}\n")
