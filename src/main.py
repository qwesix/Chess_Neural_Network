import re

import chess

from PlayingEngine import PlayingEngine
from ValueFunction import ChessANNValueFunction


uci_re = re.compile('[a-h][0-8][a-h][0-8]', re.IGNORECASE)


def get_player_move(board_: chess.Board) -> chess.Move:
    inp = input("Enter your move >>> ")
    while not uci_re.match(inp) or not board.is_legal(chess.Move.from_uci(inp)):
        inp = input("Illegal move! Enter move >>> ")

    return chess.Move.from_uci(inp)


def play_against_black(board_: chess.Board):
    computer = PlayingEngine(ChessANNValueFunction("../models/v1.pt"), 2, PlayingEngine.BLACK)

    white_on_turn = True
    while not board.is_game_over():
        if board.is_game_over():
            print("Result: ", board.result())
            return

        if white_on_turn:
            move = get_player_move(board_)
            board.push(move)
            print(board)

        else:
            move = computer.min_max_search(board)
            board.push(move)
            print(board)

        white_on_turn = not white_on_turn


def play_against_white(board_: chess.Board):
    computer = PlayingEngine(ChessANNValueFunction("../models/v1.pt"), 2, PlayingEngine.WHITE)

    white_on_turn = True
    while not board.is_game_over():
        if board.is_game_over():
            print("Result: ", board.result())
            return

        if white_on_turn:
            move = computer.min_max_parallel(board)
            board.push(move)
            print(board)

        else:
            move = get_player_move(board_)
            board.push(move)
            print(board)

        white_on_turn = not white_on_turn


if __name__ == '__main__':
    player_color = input("Choose color (w/b): ")
    while player_color != "w" and player_color != "b":
        player_color = input("Choose color (w/b): ")

    board = chess.Board()
    print(board)

    if player_color == "w":
        play_against_black(board)
    else:
        play_against_white(board)
