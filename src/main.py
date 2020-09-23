import Board
import re

if __name__ == '__main__':
    board = Board.Board()
    board.print()

    successful = board.move_algebraic("a2a3")
    print(successful)
    board.print()

    successful = board.move_algebraic("g7g5")
    print(successful)
    board.print()
