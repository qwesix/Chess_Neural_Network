import Board


class UI:
    def update(self, board):
        pass

    def make_turn(self):
        pass


class TerminalUI(UI):
    def __init__(self):
        self.board = Board()

    def update(self, board):
        self.board = board
