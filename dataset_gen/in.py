import multiprocessing as mp
import sys
from pathlib import Path
import subprocess
import pandas as pd
import requests
import time

ROOT_PATH = "/home/charles/AI4SAR/dataset_gen/data"
NUM_CPU_THREADS = 46
NUM_IMGS_TO_GENERATE = 1000000
GET_TOKEN_INTER = 3400  # time in seconds before new access token is required

TILE_CSV = pd
def get_cmds()


def worker(counter):
    last = time.time()
    token = requests.get(
        "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
    )["token"]

    while counter.value < NUM_IMGS_TO_GENERATE:
        if (now := time.time()) - last >= GET_TOKEN_INTER:
            token = requests.get(
                "https://planetarycomputer.microsoft.com/api/sas/v1/token/naip"
            )["token"]
            last = now

        load_tile_cmd = r""

        cmd1 = subprocess.run(args=[load_tile_cmd], capture_output=True)

        if cmd1.stderr != "":
            print("Error with download worker:", cmd1.stderr, file=sys.stderr)

        counter.value += 1


def main():
    counter = mp.Value(int, 0)  # shared count between processes
    for i in range(NUM_CPU_THREADS):
        mp.Process(target=worker, args=(counter,)).start()


if __name__ == "__main__":
    main()
