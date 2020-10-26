from abc import abstractmethod

import torch
import chess

from ChessANN import ChessANN


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
        white_pieces, black_pieces = 0, 0
        for c in positions:
            if c.isalpha():
                if c.isupper():
                    white_pieces += 1
                else:
                    black_pieces += 1

        return white_pieces - black_pieces if color == ValueFunction.WHITE else black_pieces - white_pieces


class ChessANNValueFunction(ValueFunction):
    def __init__(self, model_path: str):
        self.model = ChessANN()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def evaluate_position(self, board: chess.Board, color=ValueFunction.WHITE) -> float:
        if board.is_game_over():
            # no need to evaluate with neural net
            result = board.result()
            if result == '1-0':     # white won
                return 100 if color == ValueFunction.WHITE else -100

            elif result == "0-1":   # black won
                return 100 if color == ValueFunction.BLACK else -100

            else:
                return 0

        with torch.no_grad():
            in_features = ChessANN.process_epd(board.epd())
            nn_out = self.model.forward(in_features)

        # nn_out = nn_out.tolist()
        b, chance_for_draw, w = nn_out[0].item(), nn_out[1].item(), nn_out[2].item()

        win_chance_player = nn_out[0] if color == ValueFunction.BLACK else nn_out[2]
        win_chance_player = b * (color == ValueFunction.BLACK) + w * (color == ValueFunction.WHITE)
        # win_chance_enemy = nn_out[2] if color == ValueFunction.BLACK else nn_out[0]
        win_chance_enemy = b * (color != ValueFunction.BLACK) + w * (color != ValueFunction.WHITE)
        # chance_for_draw = nn_out[1]

        return 3*win_chance_player + chance_for_draw - 3*win_chance_enemy
        # return win_chance_player + chance_for_draw  # chance for not loosing
