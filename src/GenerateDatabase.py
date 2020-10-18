from tinydb import TinyDB
import torch
import chess.pgn

import os
import multiprocessing as mp
import functools
import operator


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
        while j < 8:
            if row[j] in channel_encoder:
                encoded = channel_encoder[row[j]]
                tensor[encoded[0]][i][j] = encoded[1]
            else:
                j += int(row[j])

            j += 1

    return tensor


def process_file(path) -> list:
    pgn = open(path, encoding="iso-8859-15")
    print("Processing file ", path)
    states_ = list()

    while (game := chess.pgn.read_game(pgn)) is not None:
        try:
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

                states_.append({'white': white,
                                'black': black,
                                'result': result_encoded,
                                'state': board_state,
                                'tensor': board_tensor.tolist(),
                                'on_turn': on_turn
                                })

        except Exception:
            print("Something gone wrong!")

        print("Successfully processed ", path)
    return states_


def add_to_database(db_, states_: list) -> int:
    """
    Constantly inserts the game states created by the other processes into the TinyDB
    :param states_: List with the states collected by the processes
    :param db_: The TinyDB to add the states in.
    """
    index = 0
    for state in states:
        state["index"] = index
        index += 1
    db_.insert_multiple(states)

    return index


if __name__ == '__main__':
    # pgn_folder = sys.argv[1]
    # database_path = sys.argv[2]

    pgn_folder = "../pgn/"
    database_path = "../database/chess_db.json"

    paths = list()
    for x in os.scandir(pgn_folder):
        paths.append(x.path)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map_async(process_file, paths)
        results = results.get()

    states = functools.reduce(operator.iconcat, results, [])

    print("Add collected data to database...")
    db = TinyDB(database_path)
    db.drop_tables()
    table = db.table('default_table', cache_size=50000)
    nr_examples_added = add_to_database(table, states)
    print(f"Examples in the database: {len(table)} ({nr_examples_added} newly added)")
    db.close()

