import time

import torch
import torch.nn as nn
from torch.backends import cudnn

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tinydb import TinyDB

from Model import Model, ChessANN2Dataset
from Training import print_gpu_information, print_process_bar, time_string


DATABASE_PATH = "../database/chess_db.json"
USE_GPU = True
BATCH_SIZE = 512
NR_EPOCHS = 10
torch.manual_seed(42)
sns.set_style("darkgrid")


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
    model = Model()
    # model.load_state_dict(torch.load("../models/ChessANNv2v3-90_82.pt"))  # improve already trained parameters
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
    positions = []

    states_white_wins = []
    states_black_wins = []
    states_draw = []

    # record who is on turn for the game state with the same index
    on_turn_white_wins = []
    on_turn_black_wins = []
    on_turn_draw = []

    start_time = time.time()

    i = 0
    data_length = len(data)
    for entry in data[:2000]:
        print_process_bar(i, data_length, "of data processing completed.")

        result = entry["result"] + 1
        # entry["result"] in {-1, 0, 1} but result is categorical label -> result in {0, 1, 2}
        game = entry["states"]

        processed_epds = []
        on_turns = []

        white_on_turn = False

        for state in game:
            processed_epds.append(model.process_epd(state))
            on_turns.append(torch.HalfTensor([0]) if white_on_turn else torch.Tensor([1]))
            white_on_turn = not white_on_turn

        if result == 0:  # black wins
            states_black_wins.extend(processed_epds)
            on_turn_black_wins.extend(on_turns)

        elif result == 2:  # white wins
            states_white_wins.extend(processed_epds)
            on_turn_white_wins.extend(on_turns)

        else:
            states_draw.extend(processed_epds)
            on_turn_draw.extend(on_turns)

        processed_epds = None
        i += 1

    # The same number of examples for every possible game ending:
    min_length = min(len(states_black_wins), len(states_draw), len(states_white_wins))
    labels = [0] * min_length           # black
    labels.extend([1] * min_length)     # draw
    labels.extend([2] * min_length)     # white

    # position features:
    positions = states_black_wins[:min_length]
    states_black_wins = None
    positions.extend(states_draw[:min_length])
    states_draw = None
    positions.extend(states_white_wins[:min_length])
    states_white_wins = None

    position_features_tensor = torch.Tensor(len(positions), 2, 8, 8)
    position_features_tensor.requires_grad_(False)
    torch.cat(positions, out=position_features_tensor)
    positions = None

    # recordings on who has to move next:
    on_turn_features = on_turn_black_wins[:min_length]
    on_turn_features.extend(on_turn_draw[:min_length])
    on_turn_features.extend(on_turn_white_wins[:min_length])
    on_turn_black_wins = on_turn_white_wins = on_turn_draw = None

    on_turn_tensor = torch.Tensor(len(on_turn_features), 1)
    on_turn_tensor.requires_grad_(False)
    torch.cat(on_turn_features, out=on_turn_tensor)

    labels_tensor = torch.LongTensor(labels)
    labels_tensor.requires_grad_(False)
    labels = None

    position_train, position_test, on_turn_train, on_turn_test, \
        labels_train, labels_test = train_test_split(position_features_tensor,
                                                     on_turn_tensor,
                                                     labels_tensor,
                                                     test_size=0.1 if len(labels_tensor) < BATCH_SIZE*10 else BATCH_SIZE/len(labels_tensor),
                                                     # Size of test data <= batch size (That test data fits on gpu completely)
                                                     random_state=42)

    labels_tensor = position_features_tensor = on_turn_tensor = None

    position_train = torch.FloatTensor(position_train)
    position_test = torch.FloatTensor(position_test)
    on_turn_train = torch.FloatTensor(on_turn_train)
    on_turn_train.resize_([len(on_turn_train), 1])
    on_turn_test = torch.FloatTensor(on_turn_test)
    on_turn_test.resize_([len(on_turn_test), 1])
    labels_train = torch.LongTensor(labels_train)   # .reshape(-1, 1)
    labels_test = torch.LongTensor(labels_test)     # .reshape(-1, 1)

    val_size = 10000 if len(labels_train) >= 10000 else len(labels_train)
    validation_set = position_train[0:val_size]
    validation_set = validation_set.to(device)
    validation_turns = on_turn_train[0:val_size]
    validation_turns = validation_turns.to(device)
    validation_labels = labels_train[0:val_size]
    validation_labels = validation_labels.to(device)

    train_dataset = ChessANN2Dataset(position_train, on_turn_train, labels_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    number_of_batches = len(train_loader)

    end_time = time.time()
    print(f'\nPreparing data needed {time_string(time.time() - start_time)}. \n'
          f'{len(labels_train) + len(labels_test)} data points for training available.')

    # ===== Training loop =====
    print("Starting training...")
    losses = []
    percentages = []
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    start_time = time.time()

    for i in range(NR_EPOCHS):
        if i == 10:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        elif i == 30:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        elif i == 50:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        elif i == 65:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        elif i == 75:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.000005)
        elif i == 85:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
        elif i == 95:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)

        total_loss = 0

        for positions, on_turns, labels in train_loader:
            pred = model(positions.to(device), on_turns.to(device))
            loss = criterion(pred, labels.to(device))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = None

        losses.append(total_loss / number_of_batches)

        with torch.no_grad():
            pred = model(validation_set, validation_turns)
            correct = 0
            for idx in range(val_size):
                if torch.argmax(pred[idx]).float() == validation_labels[idx].item():
                    correct += 1
            accuracy = 100 * correct / val_size
            percentages.append(accuracy)
            print(f'epoch: {i:3}  loss: {total_loss/number_of_batches:11.8f}  accuracy: {accuracy:3.2f}%')

    print(f'Training needed {time_string(time.time() - start_time)}')
    print("Validating with test data...")
    torch.cuda.empty_cache()
    correct = 0
    with torch.no_grad():
        position_test = position_test.to(device)
        on_turn_test = on_turn_test.to(device)
        labels_test = labels_test.to(device)
        pred = model(position_test, on_turn_test)
        loss = criterion(pred, labels_test)

        for idx in range(len(position_test)):
            if torch.argmax(pred[idx]).float() == labels_test[idx].item():
                correct += 1
    print(f'On Validation data: loss: {loss.item():11.8f}  accuracy: {100 * correct / len(position_test):3.2f}%')

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
        torch.save(model.state_dict(), "../../models/ChessANNv2" + name)
