import re
from Pieces import Pieces as P

class Board:
    def __init__(self):
        self.board = [
            [P.wRook, P.wKnight, P.wBishop, P.wQueen, P.wKing, P.wBishop, P.wKnight, P.wRook],
            [P.wPawn, P.wPawn, P.wPawn, P.wPawn, P.wPawn, P.wPawn, P.wPawn, P.wPawn],

            [P.empty, P.empty, P.empty, P.empty, P.empty, P.empty, P.empty, P.empty],
            [P.empty, P.empty, P.empty, P.empty, P.empty, P.empty, P.empty, P.empty],
            [P.empty, P.empty, P.empty, P.empty, P.empty, P.empty, P.empty, P.empty],
            [P.empty, P.empty, P.empty, P.empty, P.empty, P.empty, P.empty, P.empty],

            [P.bPawn, P.bPawn, P.bPawn, P.bPawn, P.bPawn, P.bPawn, P.bPawn, P.bPawn],
            [P.bRook, P.bKnight, P.bBishop, P.bQueen, P.bKing, P.bBishop, P.bKnight, P.bRook]
        ]
        self.decoder = {
            "a": 0, "b": 1, "c": 2, "d": 3,
            "e": 4, "f": 5, "g": 6, "h": 7
        }
        self.letters = "abcdefgh"
        self.notation = re.compile("^[a-h][0-8][a-h][0-8]$")
        self.rules = Rules()

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
        self.rules.move_possible(self.board, old_pos, new_pos)

    def print(self):
        for nr in range(8):
            row = self.board[7 - nr]
            to_print = ""
            for piece in row:
                to_print += P.switcher[piece]
                to_print += " "
            print(to_print)

    def __str__(self):
        string = ""
        for nr in range(8):
            row = self.board[7 - nr]
            to_print = ""
            for piece in row:
                to_print += P.switcher[piece]
                to_print += " "
            string += to_print
            if nr < 7:
                string += "\n"

        return string



class Rules:
    def __init__(self):
        pass

    def move_possible(self, board: list, move_from: (int, int), move_to: (int, int)) -> bool:
        print("Board: ", board[move_from[0]][move_from[1]])
