from tinydb import TinyDB
import chess.pgn

import os
import multiprocessing as mp
import functools
import operator


def process_file(path) -> list:
    pgn = open(path, encoding="iso-8859-15")
    print("Processing file ", path)
    games = list()
    # saves for every game an entry, that holds meta information and a list with all game states

    game = chess.pgn.read_game(pgn)
    while game is not None:
        try:
            white = game.headers["White"]
            black = game.headers["Black"]

            result_ = game.headers["Result"]
            result_encoded = 0
            if result_ == "1-0":    # white won
                result_encoded = 1
            elif result_ == "0-1":  # black won
                result_encoded = -1

            board = game.board()
            states_ = list()

            for move in game.mainline_moves():
                board.push(move)
                states_.append(board.epd())

            new_entry = {'white': white,
                         'black': black,
                         'result': result_encoded,
                         'states': states_,
                         }
            games.append(new_entry)

        except Exception:
            print("Something gone wrong! File: ", path)

        game = chess.pgn.read_game(pgn)

    print("Successfully processed ", path)
    return games


def add_to_database(db_, states_: list) -> int:
    """
    Constantly inserts the game states created by the other processes into the TinyDB
    :param states_: List with the states collected by the processes
    :param db_: The TinyDB to add the states in.
    """
    index = 0
    for state in states_:
        state["index"] = index
        index += 1
    db_.insert_multiple(states_)

    return index


if __name__ == '__main__':
    # pgn_folder = sys.argv[1]
    # database_path = sys.argv[2]

    pgn_folder = "../pgn/"
    database_path = "../../database/chess_db_big.json"

    paths = list()
    for x in os.scandir(pgn_folder):
        paths.append(x.path)

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map_async(process_file, paths, chunksize=2)
        results = results.get()

    states = functools.reduce(operator.iconcat, results, [])

    print("Add collected data to database...")
    db = TinyDB(database_path)
    db.drop_tables()
    table = db.table('default_table', cache_size=5000)
    nr_examples_added = add_to_database(table, states)
    print(f"Examples in the database: {len(table)} ({nr_examples_added} newly added)")
    db.close()

