
WHITE = 0
BLACK = 1
FREE = 0


class King:
    def __init__(self, color: int):
        self.color = color
        self.was_moved = False

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        if old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1]:   # same field -> not possible
            return False

        in_range = abs(old_pos[0] - new_pos[0]) < 2 and abs(old_pos[1] - new_pos[1]) < 2

        if in_range and (board[new_pos[0]][new_pos[1]] == FREE or board[new_pos[0]][new_pos[1]].color != self.color):
            return True

        return False

    def __str__(self):
        return "wKi" if self.color == WHITE else "bKi"


class Queen:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        if old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1]:
            return False
        if not (board[new_pos[0]][new_pos[1]] == FREE or board[new_pos[0]][new_pos[1]].color != self.color):
            # target field not free
            return False

        return True

    def __str__(self):
        return "wQu" if self.color == WHITE else "bQu"


class Rook:
    def __init__(self, color: int):
        self.color = color
        self.was_moved = False

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        if old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1]:
            return False
        if not (board[new_pos[0]][new_pos[1]] == FREE or board[new_pos[0]][new_pos[1]].color != self.color):
            # target field not free
            return False

        if old_pos[0] == new_pos[0] and horizontal_free(board, old_pos[0], old_pos[1], new_pos[1]):
            # horizontal move possible
            return True

        if old_pos[1] == new_pos[1] and vertical_free(board, old_pos[1], old_pos[0], new_pos[0]):
            # vertical move possible
            return True

        return False


    def __str__(self):
        return "wRo" if self.color == WHITE else "bRo"


class Bishop:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        if old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1]:
            return False
        if not (board[new_pos[0]][new_pos[1]] == FREE or board[new_pos[0]][new_pos[1]].color != self.color):
            # target field not free
            return False

        return True

    def __str__(self):
        return "wBi" if self.color == WHITE else "bBi"


class Knight:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        if old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1]:
            return False
        if not (board[new_pos[0]][new_pos[1]] == FREE or board[new_pos[0]][new_pos[1]].color != self.color):
            # target field not free
            return False

        return True

    def __str__(self):
        return "wKn" if self.color == WHITE else "bKn"


class Pawn:
    def __init__(self, color: int):
        self.color = color

    def move_possible(self, board: list, old_pos: (int, int), new_pos: (int, int)):
        if old_pos[0] == new_pos[0] and old_pos[1] == new_pos[1]:
            return False
        if not (board[new_pos[0]][new_pos[1]] == FREE or board[new_pos[0]][new_pos[1]].color != self.color):
            # target field not free
            return False

        return True

    def __str__(self):
        return "wPa" if self.color == WHITE else "bPa"


def horizontal_free(field: list, line_nr: int, start_nr: int, end_nr: int) -> bool:
    if abs(start_nr - end_nr) < 2:
        # return if field are the same or neighbours -> then there are no non-free fields between them
        return True

    if start_nr > end_nr:
        temp = start_nr
        start_nr = end_nr
        end_nr = temp

    # the start and end field are not counted
    for i in range(start_nr + 1, end_nr):
        if field[line_nr][i] != FREE:
            return False

    return True


def vertical_free(field: list, column_nr: int, start_nr: int, end_nr: int) -> bool:
    if abs(start_nr - end_nr) < 2:
        # return if field are the same or neighbours -> then there are no non-free fields between them
        return True

    if start_nr > end_nr:
        temp = start_nr
        start_nr = end_nr
        end_nr = temp

    # the start and end field are not counted
    for i in range(start_nr + 1, end_nr):
        if field[i][column_nr] != FREE:
            return False

    return True
