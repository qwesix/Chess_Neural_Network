from tinydb import TinyDB, Query
import tinydb
import torch
import chess.pgn
import sys
import os


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


if __name__ == '__main__':
    pgn_folder = sys.argv[1]
    database_path = sys.argv[2]

    db = TinyDB(database_path)

    nr_games = 0
    index = len(db)
    for file in os.scandir(pgn_folder):
        pgn = open(file.path)
        print("Processing file ", file.path, "...")

        try:
            while (game := chess.pgn.read_game(pgn)) is not None:
                # game = chess.pgn.read_game(pgn)
                white = game.headers["White"]
                black = game.headers["Black"]

                result = game.headers["Result"]
                result_encoded = 0
                if result == "1-0":  # white won
                    result_encoded = 1
                elif result == "0-1":
                    result_encoded = -1

                epd = game.board().epd().split(" ")
                board_state = epd[0]
                on_turn = epd[1]    # 'w' or 'b'
                board_tensor = process_epd(game.board().epd())

                db.insert({'index': index,
                           'white': white,
                           'black': black,
                           'result': result_encoded,
                           'state': board_state,
                           'tensor': board_tensor.tolist(),
                           'on_turn': on_turn
                           })
                nr_games += 1
                index += 1

        except ValueError:
            print("Exception in game. Skip to next one.")

    print(f"Games in the database: {index} ({nr_games} newly added)")
