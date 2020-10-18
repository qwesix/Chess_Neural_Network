from own import Board
from own.Pieces import *

if __name__ == '__main__':
    board = Board.Board()
    board.print()

    successful = board.move_algebraic("e2e4")
    print(successful)
    board.print()

    successful = board.move_algebraic("g7g5")
    print(successful)
    board.print()

    successful = board.move_algebraic("e1e2")
    print(successful)
    board.print()

    print(horizontal_free(board.field, 1, 5, 7))
    print(horizontal_free(board.field, 4, 0, 4))
    print(horizontal_free(board.field, 4, 0, 7))
