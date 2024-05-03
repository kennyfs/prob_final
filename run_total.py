from multiprocessing import Pool
import random
import subprocess
import sys
import time

import requests

def run_command(args):
    task, initialization, seed= args  # 解包元組
    subprocess.run(["python3", "run.py", task, initialization, str(seed)])

if __name__ == "__main__":
    r=requests.get("https://www.random.org/cgi-bin/randbyte?nbytes=8&format=d")
    q=5
    while r.status_code!=200 and q>0:
        r=requests.get("https://www.random.org/cgi-bin/randbyte?nbytes=8&format=d")
        q-=1
    if r.status_code==200:
        seed=0
        for i in r.text.split():
            seed=seed*256+int(i)
        print(f"seed: {seed}")
    else:
        seed=time.time()
        print(f"time as seed: {seed}")
    random.seed(seed)
    st = time.time()
    task = sys.argv[1]
    initialization = sys.argv[2]
    seeds=[]
    for _ in range(sys.argv[3]):
        seeds.append(random.randrange(2**64))
    args_list = [(task, initialization, seed) for seed in seeds]  # 創建一個包含31個相同元組的列表
    with Pool(5) as pool:
        pool.map(run_command, args_list)
    print(f"total time: {time.time()-st:.2f}s")
