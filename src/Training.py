import time

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tinydb import TinyDB


from src.ChessANN import ChessANN


DATABASE_PATH = "../database/chess_db_sample.json"
USE_GPU = True

# Stores for every piece the channel it gets saved in and the value:
channel_encoder = {
    'K': [0, 0],
    'Q': [0, 1],
    'R': [0, 2],
    'N': [0, 3],
    'B': [0, 4],
    'P': [0, 5],

    'k': [1, 0],
    'q': [1, 1],
    'r': [1, 2],
    'n': [1, 3],
    'b': [1, 4],
    'p': [1, 5]
}


def process_epd(epd_: str) -> torch.Tensor:
    tensor = torch.zeros([2, 8, 8])

    # 2 channels -> for every color one
    # figures encoded like in channel encode
    rows = epd_.split(" ")[0].split("/")
    for i in range(8):
        row = list(rows[i])

        j = 0
        pos = 0
        while j < 8:
            if row[pos] in channel_encoder:
                encoded = channel_encoder[row[pos]]
                tensor[encoded[0]][i][j] = encoded[1]
            else:
                j += int(row[pos])

            j += 1
            pos += 1

    return tensor


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


if __name__ == '__main__':
    # ===== Handle GPU: =====
    cuda_available = torch.cuda.is_available()

    device = torch.device("cuda:0" if (cuda_available and USE_GPU) else "cpu")
    cudnn.benchmark = True

    print_gpu_information(device, cuda_available and USE_GPU)

    # Select device to use:
    device = "cuda" if cuda_available and USE_GPU else "cpu"

    # ===== Create model =====
    model = ChessANN()
    model.train()
    model.to(device)

    # ===== Get features and labels =====
    db = TinyDB(DATABASE_PATH)
    table = db.table('default_table')
    start_time = time.time()
    data = table.all()
    end_time = time.time()
    print(f'Collecting data from db needed {time.time() - start_time:.0f} seconds. '
          f'{len(data)} games available')

    labels = []
    features = []

    start_time = time.time()
    for entry in data[:200]:
        result = entry["result"] + 1
        # entry["result"] in {-1, 0, 1} but result is categorical label -> result in {0, 1, 2}
        game = entry["states"]

        for state in game:
            features.append(process_epd(state))
            labels.append(result)

    features_tensor = torch.Tensor(len(features), 2, 8, 8)
    torch.cat(features, out=features_tensor)

    labels_tensor = torch.LongTensor(labels)
    end_time = time.time()

    print(f'Processing data needed {time.time() - start_time:.0f} seconds. '
          f'{len(labels)} data points available')

    # ===== Training loop =====
    losses = []
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
