class UI:
    def update(self, board):
        pass

    def get_black_turn(self):
        pass

    def get_white_turn(self):
        pass



class TerminalUI(UI):
    def __init__(self):
        self.board = Board()

    def update(self, board):
        self.board = board
