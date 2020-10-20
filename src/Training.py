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
    db.close()
    end_time = time.time()
    print(f'Collecting data from db needed {time.time() - start_time:.0f} seconds. '
          f'{len(data)} games available')

    labels = []
    features = []

    start_time = time.time()
    for entry in data[0:200]:
        result = entry["result"] + 1
        # entry["result"] in {-1, 0, 1} but result is categorical label -> result in {0, 1, 2}
        game = entry["states"]

        for state in game:
            features.append(model.process_epd(state).unsqueeze(0))
            labels.append(result)

    features_tensor = torch.Tensor(len(features), 2, 8, 8)
    torch.cat(features, out=features_tensor)

    labels_tensor = torch.LongTensor(labels)
    end_time = time.time()

    print(f'Processing data needed {time.time() - start_time:.0f} seconds. '
          f'{len(labels)} data points available')

    if device != "cpu":
        labels_tensor.to(device)
        features_tensor.to(device)

    # ===== Training loop =====
    losses = []
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)

    start_time = time.time()
    for i in range(10):
        pred = model(features_tensor, train=True)

        loss = criterion(pred, labels_tensor)
        losses.append(loss)

        correct = 0
        for idx in range(len(labels)):
            if torch.argmax(pred[idx]).float() == labels_tensor[idx].item():
                correct += 1
        print(f'epoch: {i:3}  loss: {loss.item():11.8f}  accuracy: {100 * correct / len(labels):3.2f}%')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Training needed {time.time() - start_time:.0f} seconds')

    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    axes = plt.gca()
    axes.set_ylim([0, None])
    plt.show()

    name = input("Save model? >>> ")
    if name != "":
        torch.save(model.state_dict(), "../models/" + name)
