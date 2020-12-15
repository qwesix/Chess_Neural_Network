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
DATABASE_PATH = "../../database/chess_db_sample.json"
USE_GPU = True
BATCH_SIZE = 512
NR_EPOCHS = 10
TEST_SIZE = 0.1  # between 0 and 1
GLOBAL_SEED = 42

seed_everything(GLOBAL_SEED)     # set the seed for everything to 42 to guarantee reproducibility
sns.set_style("darkgrid")


def games_to_list(tinydb_entries):
    positions_list = []
    for game in tinydb_entries:
        result = game["result"] + 1
        # entry["result"] in {-1, 0, 1} but result is categorical label -> result in {0, 1, 2}
        positions = game["states"]

        for position in positions:
            positions_list.append((position, result))

    return positions_list


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
    db = TinyDB(DATABASE_PATH)
    table = db.table('default_table')
    start_time = time.time()
    data = table.all()
    db.close()
    end_time = time.time()

    nr_of_games = len(data)
    print(f'Collecting data from db needed {time_string(time.time() - start_time)}. '
          f'{nr_of_games} games available')

    # Typically the samples get splitted but here we split the games so that the positions in the
    # test and training samples are from different games.
    games_for_train, games_for_test = train_test_split(data, test_size=TEST_SIZE, random_state=GLOBAL_SEED)

    train_positions = games_to_list(games_for_train)
    test_positions = games_to_list(games_for_test)
