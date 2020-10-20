import chess.pgn
import torch

if __name__ == '__main__':
    # board = chess.Board()
    pgn = open("../pgn/Alexhine Grischuk.pgn", encoding="utf-8")
    # pgn = open("../pgn/Magnus-Carlsen_vs_Jeffery-Xiong_2020.01.13.pgn", encoding="utf-8")
    game = chess.pgn.read_game(pgn)

    print(game.headers["Event"])
    print("White:  ", game.headers["White"])
    print("Black:  ", game.headers["Black"])
    print("Result: ", game.headers["Result"])

    board = game.board()
    print(board)

    print(game.board().epd())

    # for move in game.mainline_moves():
    #     board.push(move)
    #     print(board.epd())

    board_before = game.board().__str__()
    board = game.board()

    print([x for x in board.legal_moves])

    # make two moves
    board.push([x for x in board.legal_moves][0])
    board.push([x for x in board.legal_moves][0])

    # redo the moves
    board.pop()
    board.pop()

    if board.__str__() == board_before:
        print("equal")

    print(board.is_game_over())

