import re

class Board:
    def __init__(self):
        self.board = [
            []
        ]
        self.decoder = {
            "a": 0, "b": 1, "c": 2, "d": 3,
            "e": 4, "f": 5, "g": 6, "h": 7
        }
        self.letters = "abcdefgh"
        self.notation = re.compile("^[a-h][0-8][a-h][0-8]$")

    def move_algebraic(self, move_str: str):
        if len(move_str) != 4:
            return "Wrong length!"

        if not self.notation.match(move_str):
            return "Syntax violated!"

        self.move((self.decoder[move_str[0]], 8 - int(move_str[1])), (self.decoder[move_str[2]], 8 - int(move_str[3])))
        return True

    def move(self, old_pos: (int, int), new_pos: (int, int)):
        print(old_pos)
        print(new_pos)
        pass


class Pieces:
    empty = 0
    wKing = 1
    wQueen = 2
    wRook = 3
    wBishop = 4
    wKnight = 5
    wPawn = 6
    bKing = -1
    bQueen = -2
    bRook = -3
    bBishop = -4
    bKnight = -5
    bPawn = -6