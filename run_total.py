from multiprocessing import Pool
import random
import subprocess
import sys
import time

import requests


def run_command(args):
    task, initialization, seed = args  # 解包元組
    subprocess.run(["python3", "run.py", task, initialization, str(seed)])


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


if __name__ == "__main__":
    st = time.time()
    task = sys.argv[1]
    len = int(sys.argv[2])
    initialization = sys.argv[3]
    if initialization == "random" or initialization == "r":
        initialization_seed = getrandom(len)
    else:
        initialization_seed = [int(initialization) for _ in range(len)]
    dataset = sys.argv[4]
    if dataset == "random" or dataset == "r":
        dataset_seed = getrandom(len)
    else:
        dataset_seed = [int(dataset) for _ in range(len)]
    args_list = [
        (task, init, dataseed)
        for init, dataseed in zip(initialization_seed, dataset_seed)
    ]  # 創建一個包含31個相同元組的列表
    pools = sys.argv[5]
    if pools == "1":
        for args in args_list:
            run_command(args)
    else:
        with Pool(pools) as pool:
            pool.map(run_command, args_list)
    print(f"total time: {time.time()-st:.2f}s")
