from multiprocessing import Pool
import random
import subprocess
import argparse
import time

import requests


def run_command(args):
    task, initialization, seed, dataset_seed = args  # 解包元組
    subprocess.run(
        [
            "python3",
            "run.py",
            str(task),
            str(initialization),
            str(seed),
            str(dataset_seed),
        ]
    )


def getrandom(len):
    r = requests.get("https://www.random.org/cgi-bin/randbyte?nbytes=8&format=d")
    q = 5
    while r.status_code != 200 and q > 0:
        r = requests.get("https://www.random.org/cgi-bin/randbyte?nbytes=8&format=d")
        q -= 1
    if r.status_code == 200:
        seed = 0
        for i in r.text.split():
            seed = seed * 256 + int(i)
        print(f"seed: {seed}")
    else:
        seed = time.time()
        print(f"time as seed: {seed}")
    random.seed(seed)
    seeds = []
    for _ in range(len):
        seeds.append(random.randrange(2**64))
    return seeds


def get_args():
    parser = argparse.ArgumentParser(
        description="Run tasks with different seeds and initializations."
    )
    parser.add_argument("--task", type=str, help="The task to run.")
    parser.add_argument("--len", type=int, help="The length of the seed list.")
    parser.add_argument("--initialization", type=str, help="The initialization method.")
    parser.add_argument(
        "--initialization_seed", type=str, help="The seed for initialization."
    )
    parser.add_argument("--dataset_seed", type=str, help="The seed for the dataset.")
    parser.add_argument(
        "--pools", type=int, help="The number of pools for multiprocessing."
    )
    return parser.parse_args()


if __name__ == "__main__":
    st = time.time()
    args = get_args()
    if args.initialization_seed == "random" or args.initialization_seed == "r":
        seeds = getrandom(args.len)
    else:
        seeds = [int(args.initialization_seed) for _ in range(args.len)]
    if args.dataset_seed == "random" or args.dataset_seed == "r":
        dataset_seeds = getrandom(args.len)
    else:
        dataset_seeds = [int(args.dataset_seed) for _ in range(args.len)]
    args_list = [
        (args.task, args.initialization, seed, dataseed)
        for seed, dataseed in zip(seeds, dataset_seeds)
    ]
    if args.pools == 1:
        for args in args_list:
            run_command(args)
    else:
        with Pool(args.pools) as pool:
            pool.map(run_command, args_list)
    print(f"total time: {time.time()-st:.2f}s")
#command: python run_total.py --task gcd --len 100 --initialization constant --initialization_seed 0 --dataset_seed r --pools 4