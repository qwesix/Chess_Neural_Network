import time

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torchvision import datasets, transforms

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tinydb import TinyDB


from src.ChessANN import ChessANN


DATABASE_PATH = "../database/chess_db.json"
USE_GPU = True
BATCH_SIZE = 25000
NR_EPOCHS = 15
torch.manual_seed(42)
sns.set_style("darkgrid")


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


if __name__ == '__main__':
    # ===== Handle GPU: =====
    cuda_available = torch.cuda.is_available()

    device = torch.device("cuda:0" if (cuda_available and USE_GPU) else "cpu")
    cudnn.benchmark = True

    print_gpu_information(device, cuda_available and USE_GPU)

    # Select device to use:
    device = "cuda" if cuda_available and USE_GPU else "cpu"
    torch.backends.cudnn.benchmark = True

    # ===== Create model =====
    model = ChessANN()
    model.train()
    model = model.to(device)

    # ===== Get features and labels =====
    db = TinyDB(DATABASE_PATH)
    table = db.table('default_table')
    start_time = time.time()
    data = table.all()
    db.close()
    end_time = time.time()
    print(f'Collecting data from db needed {time_string(time.time() - start_time)}. '
          f'{len(data)} games available')

    labels = []
    features = []

    start_time = time.time()
    for entry in data:
        result = entry["result"] + 1
        # entry["result"] in {-1, 0, 1} but result is categorical label -> result in {0, 1, 2}
        game = entry["states"]

        for state in game:
            features.append(model.process_epd(state).unsqueeze(0))
            labels.append(result)

    features_tensor = torch.Tensor(len(features), 2, 8, 8)
    features_tensor.requires_grad_(False)
    torch.cat(features, out=features_tensor)

    labels_tensor = torch.LongTensor(labels)
    labels_tensor.requires_grad_(False)

    train_x, test_x, train_y, test_y = train_test_split(features_tensor,
                                                        labels_tensor,
                                                        test_size=0.1 if len(labels_tensor) < 200000 else 20000/len(labels_tensor),
                                                        random_state=42)

    train_x = torch.FloatTensor(train_x)
    test_x = torch.FloatTensor(test_x)
    train_y = torch.LongTensor(train_y)   # .reshape(-1, 1)
    test_y = torch.LongTensor(test_y)   # .reshape(-1, 1)

    val_size = 10000
    validation_set = train_x[0:val_size]
    validation_set = validation_set.to(device)
    validation_labels = train_y[0:val_size]
    validation_labels = validation_labels.to(device)

    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    end_time = time.time()
    print(f'Preparing data needed {time_string(time.time() - start_time)}. '
          f'{len(labels)} data points available')

    # ===== Training loop =====
    print("Starting training...")
    losses = []
    percentages = []
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    start_time = time.time()

    for i in range(NR_EPOCHS):
        if i == 5:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        total_loss = 0

        for features, labels in train_loader:
            pred = model(features.to(device), train=True)
            loss = criterion(pred, labels.to(device))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = None

        losses.append(total_loss)
        with torch.no_grad():
            pred = model(validation_set, train=True)
            correct = 0
            for idx in range(val_size):
                if torch.argmax(pred[idx]).float() == validation_labels[idx].item():
                    correct += 1
            accuracy = 100 * correct / val_size
            percentages.append(accuracy)
            print(f'epoch: {i:3}  loss: {total_loss:11.8f}  accuracy: {accuracy:3.2f}%')

    print(f'Training needed {time_string(time.time() - start_time)}')
    print("Validating with test data...")
    torch.cuda.empty_cache()
    correct = 0
    with torch.no_grad():
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        pred = model(test_x, train=True)
        loss = criterion(pred, test_y)

        for idx in range(len(test_x)):
            if torch.argmax(pred[idx]).float() == labels_tensor[idx].item():
                correct += 1
    print(f'On Validation data: loss: {loss.item():11.8f}  accuracy: {100 * correct / len(test_x):3.2f}%')

    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(percentages)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    axes = plt.gca()
    axes.set_ylim([0, 100])
    plt.show()

    name = input("Save model? >>> ")
    if name != "":
        torch.save(model.state_dict(), "../models/" + name)
