import Board
import re

if __name__ == '__main__':
    board = Board.Board()
    board.move_algebraic("g1f3")

    # board.print()
    print(board)

    notation = re.compile("^[a-h][0-8][a-h][0-8]$")
    if notation.match("f1g5"):
        print("match!")
    if notation.match("3gd2"):
        print("wrong match!")
    if notation.match("a5c8"):
        print("match!")