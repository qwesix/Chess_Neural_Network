import re
from own.Pieces import *


class Board:
    def __init__(self):
        self.field = [
            [Rook(WHITE), Knight(WHITE), Bishop(WHITE), Queen(WHITE), King(WHITE), Bishop(WHITE), Knight(WHITE), Rook(WHITE)],
            [Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE), Pawn(WHITE)],

            [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],
            [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],
            [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],
            [FREE, FREE, FREE, FREE, FREE, FREE, FREE, FREE],

            [Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK), Pawn(BLACK)],
            [Rook(BLACK), Knight(BLACK), Bishop(BLACK), Queen(BLACK), King(BLACK), Bishop(BLACK), Knight(BLACK), Rook(BLACK)]
        ]
        assert len(self.field) == 8
        for row in self.field:
            assert len(row) == 8

        self.decoder = {
            "a": 0, "b": 1, "c": 2, "d": 3,
            "e": 4, "f": 5, "g": 6, "h": 7
        }
        self.letters = "abcdefgh"
        self.notation = re.compile("^[a-h][1-8][a-h][1-8]$")

        self.on_turn = WHITE

    def move_algebraic(self, move_str: str):
        """
        The string must have the size 4 and consists of letter+number of olf position followed by letter+number of new
        position. e.g. "a1a4" or "h7d7"
        :param move_str: The string that holds the information of the move to do.
        :return: True if the move was successful, otherwise a string with a error message.
        """
        if len(move_str) != 4:
            return "Wrong length!"

        if not self.notation.match(move_str):
            return "Syntax violated!"

        return self.move((self.decoder[move_str[0]], int(move_str[1]) - 1),
                         (self.decoder[move_str[2]], int(move_str[3]) - 1))

    def move(self, old_pos: (int, int), new_pos: (int, int)):
        """
        Makes move from old_pos to new_pos. The first specifies the position on the alphabetical scale (a=0, ..., h=7)
        and the second specifies the position on the numeric scale (0 to 7).
        :param old_pos: The position the piece is standing now.
        :param new_pos: The position to move to.
        :return: True if the move was successful, otherwise a string with a error message.
        """
        old_x, old_y = old_pos
        new_x, new_y = new_pos
        if self.field[old_y][old_x] == FREE:
            return "There is no piece! (" + str(old_x) + ", " + str(old_y) + ")"

        if self.field[old_y][old_x].color != self.on_turn:
            return ("White" if self.on_turn == WHITE else "Black") + " is on turn!"

        # turn x and y around for the piece:
        if self.field[old_y][old_x].move_possible(self.field, (old_y, old_x), (new_y, new_x)):
            if self.field[new_y][new_x] is King:
                self.field[new_y][new_x].was_moved = True

            self.field[new_y][new_x] = self.field[old_y][old_x]
            self.field[old_y][old_x] = FREE
            self.on_turn = (self.on_turn + 1) % 2
            return True

        return "Move not possible!"

    def print(self, show_labeling=True):
        string = ""
        for nr in range(8):
            row = self.field[7 - nr]
            if show_labeling:
                string += str(8 - nr) + " "

            to_print = ""
            for piece in row:
                to_print += "[ ]" if piece == FREE else piece.__str__()
                to_print += " "
            string += to_print
            if nr < 7:
                string += "\n"

        if show_labeling:
            string += "\n   a   b   c   d   e   f   g   h"

        print(string)
        print("On turn: ", "white" if self.on_turn == WHITE else "black", "\n")

    def __str__(self):
        string = ""
        for nr in range(8):
            row = self.field[7 - nr]
            to_print = ""
            for piece in row:
                to_print += "[ ]" if piece == FREE else piece.__str__()
                to_print += " "
            string += to_print
            if nr < 7:
                string += "\n"

        return string
