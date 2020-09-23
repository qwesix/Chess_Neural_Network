
WHITE = 0
BLACK = 1

FREE = 0


class King:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        if abs(old_pos[0] - new_pos[0]) < 1 and abs(old_pos[1] - new_pos[1]) < 1:
            return True
        # return False
        return True

    def __str__(self):
        return "wKi" if self.color == WHITE else "bKi"


class Queen:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        return True

    def __str__(self):
        return "wQu" if self.color == WHITE else "bQu"


class Rook:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        return True


    def __str__(self):
        return "wRo" if self.color == WHITE else "bRo"


class Bishop:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        return True

    def __str__(self):
        return "wBi" if self.color == WHITE else "bBi"


class Knight:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        return True

    def __str__(self):
        return "wKn" if self.color == WHITE else "bKn"


class Pawn:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        return True

    def __str__(self):
        return "wPa" if self.color == WHITE else "bPa"
