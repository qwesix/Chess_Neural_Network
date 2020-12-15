import time

import torch
import torch.nn as nn
from torch.backends import cudnn

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tinydb import TinyDB

from Model import Model
from CustomDataset import CustomDataset
from Utils import print_gpu_information, print_process_bar, time_string, seed_everything

# ===== PREFERENCES =====
DATABASE_PATH = "../database/chess_db.json"
USE_GPU = True
BATCH_SIZE = 512
NR_EPOCHS = 10

seed_everything(42)     # set the seed for everything to 42 to guarantee reproducibility
sns.set_style("darkgrid")


if __name__ == '__main__':
    # ===== Handle GPU: =====
    use_cuda = torch.cuda.is_available() and USE_GPU
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Select device to use:
    device = "cuda" if use_cuda else "cpu"
    torch.backends.cudnn.benchmark = True

    # ===== Create model =====
    model = Model()
    # model.load_state_dict(torch.load(""))  # improve already trained parameters
    model.train()
    model = model.to(device)

    # ===== Get features and labels =====
