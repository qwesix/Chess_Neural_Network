import sys

import chess

from ValueFunction import ValueFunction, DumbValueFunction


class PlayingEngine:
    WHITE = 0
    BLACK = 1

    def __init__(self, value_function: ValueFunction, search_depth: int, player_color=WHITE):
        self.search_depth = search_depth
        self.player_color = player_color
        self.value_function = value_function

    def min_max_search(self, board: chess.Board) -> chess.Move:
        """
        Tries to find an optimal move via mini maxi algorithm and returns that move.
        :param board:
        """
        result, move = self.maximize(board.copy(stack=False), 0)
        print("Value: ", result)
        print("Make move: ", move)

        return move

    def minimize(self, board: chess.Board, depth: int, move_done=None) -> (float, chess.Move):
        if depth > self.search_depth:
            return self.value_function.evaluate_position(board, self.player_color), move_done

        else:
            min_val = 10000
            best_move = None

            for move in board.legal_moves:
                temp = board.copy(stack=False)
                temp.push(move)
                val, mov = self.maximize(temp, depth + 1, move)

                if val < min_val:
                    min_val = val
                    best_move = move

            return min_val, best_move

    def maximize(self, board: chess.Board, depth: int, move_done=None) -> (float, chess.Move):
        if depth > self.search_depth or board.is_game_over():
            return self.value_function.evaluate_position(board, self.player_color), move_done

        else:
            max_val = -10000
            best_move = None

            for move in board.legal_moves:
                temp = board.copy(stack=False)
                temp.push(move)
                val, mov = self.minimize(temp, depth + 1, move)
                if val > max_val:
                    max_val = val
                    best_move = move

            return max_val, best_move


if __name__ == '__main__':
    board = chess.Board()
    print(board)
    black = PlayingEngine(DumbValueFunction(), 2, PlayingEngine.BLACK)

    for i in range(8):
        inp = input("Enter your move >>> ")
        board.push_uci(inp)
        print(board)

        move = black.min_max_search(board)
        board.push(move)
        print(board)
