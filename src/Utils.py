import os
import sys
import numpy as np
import torch

from random import random


# File with some helpful methods that may be used by multiple other parts of the program


def print_gpu_information(device_, use_gpu: bool):
    if use_gpu:
        print('torch version: ' + torch.__version__,
              "\tAvailable GPUs: " + str(torch.cuda.device_count()),
              " Device: " + torch.cuda.get_device_name(),
              "\tMemory: ", int(torch.cuda.get_device_properties(device_).total_memory / (1024 * 1024)), "MB",
              "\tMulti processors: ", torch.cuda.get_device_properties(device_).multi_processor_count,
              "\tGPU used? YES")
    else:
        print('torch version: ' + torch.__version__,
              "\tAvailable GPUs: " + str(torch.cuda.device_count()),
              "\tGPU used? NO")


def time_string(seconds: float) -> str:
    seconds = int(seconds)
    hours = int(seconds/3600)
    minutes = int((seconds - 3600*hours) / 60)
    rest_seconds = int(seconds - 3600*hours - 60*minutes)
    return f'{hours} hrs {minutes} min {rest_seconds} sec'


def print_process_bar(i, max, post_text=""):
    n_bar = 40   # size of progress bar
    j = i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {post_text}")
    sys.stdout.flush()


def seed_everything(seed=42):
    """"
    Seed everything.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
