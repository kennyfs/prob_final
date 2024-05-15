from multiprocessing import Pool
import random
import subprocess
import sys
import time

import requests


def run_command(args):
    task, initialization, seed, dataset_seed = args  # 解包元組
    subprocess.run(["python3", "run.py", str(task), str(initialization), str(seed), str(dataset_seed)])


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
    initialization_seed = sys.argv[4]
    if initialization_seed == "random" or initialization_seed == "r":
        seeds = getrandom(len)
    else:
        seeds = [int(initialization_seed) for _ in range(len)]
    dataset_seed = sys.argv[5]
    if dataset_seed == "random" or dataset_seed == "r":
        dataset_seeds = getrandom(len)
    else:
        dataset_seeds = [int(dataset_seed) for _ in range(len)]
    dataset_seeds = [2930343802894567464,2782449217923232664,2180212886986422689,97153724350858039,9333075139285733321,7208607079989587122,11266968294869642971,14442195557557080347,3359790119616197621,2608339041415264851,16682608125961105803,5351007782696044995,12058667363950800367,11830780559826849050,6897657190249890931,14204854080110022879,1351714700882209485,11762627264842317819,11829283031967775573,13023068868739930551]
    dataset_seeds = [seed for seed in dataset_seeds for _ in range(5)]
    len = 20*5
    seeds = getrandom(len)
    args_list = [
        (task, initialization, seed, dataseed)
        for seed, dataseed in zip(seeds, dataset_seeds)
    ]
    pools = int(sys.argv[6])
    if pools == 1:
        for args in args_list:
            run_command(args)
    else:
        with Pool(pools) as pool:
            pool.map(run_command, args_list)
    print(f"total time: {time.time()-st:.2f}s")
