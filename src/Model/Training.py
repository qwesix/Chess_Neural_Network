import time
import multiprocessing

import torch
import torch.nn as nn
from torch.backends import cudnn

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tinydb import TinyDB

import Model
from Model import Model
from CustomDataset import CustomDataset
from Utils import print_gpu_information, print_process_bar, time_string, seed_everything

# ===== PREFERENCES =====
DATABASE_PATH = "../../database/chess_db_sample.json"
USE_GPU = True
BATCH_SIZE = 256
NR_EPOCHS = 4
TEST_SIZE = 0.3  # between 0 and 1
GLOBAL_SEED = 42

seed_everything(GLOBAL_SEED)     # set the seed for everything to 42 to guarantee reproducibility
sns.set_style("darkgrid")


def games_to_list(tinydb_entries):
    """
    Takes a list with games (tinyDB entries) and returns a list with the positions of the games.
    :return: lists consisting of tuples (position epd, player on turn, result)
    """
    positions_list = []
    for game in tinydb_entries:
        result = game["result"] + 1
        # entry["result"] in {-1, 0, 1} but result is categorical label -> result in {0, 1, 2}
        positions = game["states"]

        half_move_clock = 0
        for position in positions:
            positions_list.append((position, torch.Tensor([half_move_clock]), result))
            half_move_clock = (half_move_clock + 1) % 2

    return positions_list


def positions_list_to_tensors(positions_list: list):
    """
    Takes a list created by the method 'games_to_list'
    :param positions_list: list created by the method 'games_to_list'
    """
    length = len(positions_list)
    positions_tensor = torch.zeros([length, 2, 8, 8])   # 2 = nr of channels => black / white
    player_on_turn_tensor = torch.zeros([length, 1])
    result_tensor = torch.zeros([length])

    counter = 0
    for position in positions_list:
        Model.process_and_add_to_tensor(position[0], positions_tensor, counter)
        player_on_turn_tensor[counter][0] = position[1]     # TODO: Testen ob = auch durch += ersetzt werden kann
        result_tensor[counter] = position[2]
        counter += 1


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

    # ===== Get features and labels from database =====
    db = TinyDB(DATABASE_PATH)
    table = db.table('default_table')
    start_time = time.time()
    data = table.all()
    db.close()
    end_time = time.time()

    nr_of_games = len(data)
    print(f'Collecting data from db needed {time_string(time.time() - start_time)}. '
          f'{nr_of_games} games available.')

    # Typically the samples get split but here we split between complete games so that the positions in the
    # test and training samples are from different games. (So they don't correlate)
    games_for_train, games_for_test = train_test_split(data, test_size=TEST_SIZE, random_state=GLOBAL_SEED)

    train_positions = games_to_list(games_for_train)
    test_positions = games_to_list(games_for_test)

    train_dataset = CustomDataset(train_positions)
    test_dataset = CustomDataset(test_positions)
    nr_train_samples = len(train_dataset)
    # sample = train_dataset.__getitem__(0)
    # print(sample)

    train_loader = torch.utils.data.DataLoader(train_dataset
                                        , batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset
                                        , batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    number_of_batches = len(train_loader)
    # sample = next(iter(train_dataloader))
    # print(type(sample))
    # print(sample)

    # ===== Training loop =====
    print("Starting training...")
    losses = []
    percentages = []
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.02)

    start_time = time.time()

    # Train:
    for i in range(NR_EPOCHS):
        total_loss = 0
        correct = 0

        for positions, on_turns, labels in train_loader:
            pred = model(positions.to(device), on_turns.to(device))
            loss = criterion(pred, labels.to(device))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for idx in range(len(pred)):
                if torch.argmax(pred[idx]).float() == labels[idx].item():
                    correct += 1

        pred = None
        losses.append(total_loss / number_of_batches)

        accuracy = 100 * correct / nr_train_samples
        percentages.append(accuracy)
        print(f'epoch: {i:3}  loss: {total_loss / number_of_batches:11.8f}  accuracy: {accuracy:3.2f}%')

    # Validate:
    train_loader = None
    train_dataset = None

    with torch.no_grad():
        model.eval()
        total_loss = 0
        correct = 0

        for positions, on_turns, labels in test_loader:
            pred = model(positions.to(device), on_turns.to(device))
            loss = criterion(pred, labels.to(device))
            total_loss += loss.item()

            for idx in range(len(pred)):
                if torch.argmax(pred[idx]).float() == labels[idx].item():
                    correct += 1

        pred = None
        print(f'On Validation data: loss: {total_loss:11.8f}  accuracy: {100 * correct / len(test_dataset):3.2f}%')

    # plot the collected data:
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
