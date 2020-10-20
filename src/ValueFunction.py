from abc import abstractmethod

import chess


class ValueFunction:
    """
        Child classes of this class implement functions that
        calculate the value of a board for the use in game trees
    """

    WHITE = 0
    BLACK = 1

    @abstractmethod
    def evaluate_position(self, board: chess.Board, color=WHITE) -> float:
        raise NotImplementedError


class DumbValueFunction(ValueFunction):
    def evaluate_position(self, board: chess.Board, color=ValueFunction.WHITE) -> float:
        positions = board.epd().split(" ")[0]
        white_pieces = 0
        black_pieces = 0
        for c in positions:
            if c.isalpha():
                if c.isupper():
                    white_pieces += 1
                else:
                    black_pieces += 1

        return white_pieces - black_pieces if color == ValueFunction.WHITE else black_pieces - white_pieces
