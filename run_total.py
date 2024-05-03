from multiprocessing import Pool
import subprocess
import time

def run_command(args):
    gcd, normal = args  # 解包元組
    subprocess.run(["python3", "run.py", gcd, normal])

if __name__ == "__main__":
    st = time.time()
    gcd = "ChickenRabbit"
    normal = "normal"
    seeds=[]
    
    args_list = [(gcd, normal) for seed in seeds]  # 創建一個包含31個相同元組的列表
    with Pool(5) as pool:
        pool.map(run_command, args_list)
    print(f"total time: {time.time()-st:.2f}s")
