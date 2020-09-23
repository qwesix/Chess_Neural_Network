import re
from Pieces import *


class Board:
    def __init__(self):
        self.board = [
            [Rook(WHITE), Knight(WHITE), Bishop(WHITE), Queen(WHITE), King(WHITE), Bishop(WHITE), Knight(WHITE), Rook(WHITE)],
            [Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE)],

            [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],
            [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],
            [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],
            [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],

            [Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK)],
            [Rook(BLACK), Knight(BLACK), Bishop(BLACK), Queen(BLACK), King(BLACK), Bishop(BLACK), Knight(BLACK), Rook(BLACK)]
        ]
        assert len(self.board) == 8
        for row in self.board:
            assert len(row) == 8

        self.decoder = {
            "a": 0, "b": 1, "c": 2, "d": 3,
            "e": 4, "f": 5, "g": 6, "h": 7
        }
        self.letters = "abcdefgh"
        self.notation = re.compile("^[a-h][1-8][a-h][1-8]$")

    def move_algebraic(self, move_str: str):
        if len(move_str) != 4:
            return "Wrong length!"

        if not self.notation.match(move_str):
            return "Syntax violated!"

        self.move((self.decoder[move_str[0]], 8 - int(move_str[1])), (self.decoder[move_str[2]], 8 - int(move_str[3])))
        return True

    def move(self, old_pos: (int, int), new_pos: (int, int)) -> bool:
        print(old_pos)
        print(new_pos)
        old_x, old_y = old_pos
        new_x, new_y = new_pos
        if self.board[old_x][old_y] == FREE:
            return False

        if self.board[old_x][old_y].move_possible(self.board, old_pos, new_pos):
            self.board[new_x][new_y] = self.board[old_x][old_y]
            self.board[old_x][old_y] = FREE


    def print(self):
        print(self.__str__())

    def __str__(self):
        string = ""
        for nr in range(8):
            row = self.board[7 - nr]
            to_print = ""
            for piece in row:
                to_print += "[ ]" if piece == FREE else piece.__str__()
                to_print += " "
            string += to_print
            if nr < 7:
                string += "\n"

        return string
