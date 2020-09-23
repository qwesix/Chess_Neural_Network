import Board
import re

if __name__ == '__main__':
    board = Board.Board()
    board.print()

    board.move_algebraic("a2a3")
    board.print()

    board.move_algebraic("g7g5")
    board.print()