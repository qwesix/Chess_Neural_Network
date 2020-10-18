from tinydb import TinyDB, Query
import torch
import chess.pgn

import sys
import os
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import time


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

STATES = mp.Queue()


def process_epd(epd_: str) -> torch.Tensor:
    tensor = torch.zeros([2, 8, 8])

    # 2 channels -> for every color one
    # figures encoded like in channel encode
    rows = epd_.split(" ")[0].split("/")
    for i in range(8):
        row = list(rows[i])

        j = 0
        while j < 8:
            if row[j] in channel_encoder:
                encoded = channel_encoder[row[j]]
                tensor[encoded[0]][i][j] = encoded[1]
            else:
                j += int(row[j])

            j += 1

    return tensor


def process_file(path):
    pgn = open(path)
    print("Processing file ", path, "...")

    while (game := chess.pgn.read_game(pgn)) is not None:
        white = game.headers["White"]
        black = game.headers["Black"]

        result_ = game.headers["Result"]
        result_encoded = 0
        if result_ == "1-0":  # white won
            result_encoded = 1
        elif result_ == "0-1":
            result_encoded = -1

        board = game.board()

        for move in game.mainline_moves():
            board.push(move)
            epd = game.board().epd().split(" ")
            board_state = epd[0]
            on_turn = epd[1]  # 'w' or 'b'
            board_tensor = process_epd(game.board().epd())

            STATES.put({'white': white,
                        'black': black,
                        'result': result_encoded,
                        'state': board_state,
                        'tensor': board_tensor.tolist(),
                        'on_turn': on_turn
                        })

    print("Successfully processed ", path)


def add_to_database(db_: TinyDB, states_: mp.Queue):
    """
    Constantly inserts the game states created by the other processes into the TinyDB
    :param states_: Queue with the data collected by the processes
    :param db_: The TinyDB to add the data in.
    """
    index = 0
    while not states_.empty():
        print("Data to db")
        data = states_.get()
        data["index"] = index
        index += 1
        db_.insert(data)

    return index


if __name__ == '__main__':
    # pgn_folder = sys.argv[1]
    # database_path = sys.argv[2]

    pgn_folder = "../pgn/"
    database_path = "../database/chess_db.json"

    paths = list()
    for x in os.scandir(pgn_folder):
        paths.append(x.path)

    db = TinyDB(database_path)
    states = mp.Queue()

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map_async(process_file, paths[0:2])
        results.get()

    nr_examples_added = add_to_database(db, states)
    print(f"Examples in the database: {len(db)} ({nr_examples_added} newly added)")
