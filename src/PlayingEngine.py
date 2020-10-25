import time
import multiprocessing as mp

import chess

from Training import time_string
from ValueFunction import ValueFunction, DumbValueFunction, ChessANNValueFunction


class PlayingEngine:
    WHITE = 0
    BLACK = 1

    def __init__(self, value_function: ValueFunction, search_depth: int, player_color=WHITE):
        self.search_depth = search_depth
        self.player_color = player_color
        self.value_function = value_function

    def min_max_search(self, board: chess.Board, use_alpha_beta=False, mute=False) -> chess.Move:
        """
        Tries to find an optimal move via mini maxi algorithm and returns that move.
        :param mute: No text output if True
        :param use_alpha_beta: If true the algorithm uses alpha-beta-search with pruning
        :param board:
        """

        start_time = time.time()
        best_move = None
        best_result = -10000

        if use_alpha_beta:
            for move in board.legal_moves:
                temp = board.copy(stack=False)
                temp.push(move)
                result = self._alpha_beta_min_(temp, 1, alpha=-10000, beta=10000)
                # Normally maximize must be called, but this method replaces maximise at the beginning
                if result > best_result:
                    best_result = result
                    best_move = move
        else:
            for move in board.legal_moves:
                temp = board.copy(stack=False)
                temp.push(move)
                result = self._minimize_(temp, 1)
                # Because of that we call minimize
                # Normally maximize must be called, but this method replaces maximise at the beginning
                # Because of that we call minimize
                if result > best_result:
                    best_result = result
                    best_move = move

        end_time = time.time()

        if not mute:
            print(f"Value: {best_result:.4f} Computation time: {time_string(end_time-start_time)}")
            print("Make move: ", best_move)

        return best_move

    def min_max_parallel(self, board: chess.Board, use_alpha_beta=True, mute=False, number_workers=mp.cpu_count()):
        """
        Tries to find an optimal move via mini maxi algorithm and returns that move.
        This method uses multiple processes
        :param mute: No text output if True
        :param number_workers: number of parallel processes that are utilized. Default is mp.cpu_count(), but sometimes
                               it helps to make the number smaller.
        :param use_alpha_beta: If true the algorithm uses alpha-beta-search with pruning
        :param board: The current board.
        """
        start_time = time.time()

        legal_moves = board.legal_moves
        nr_of_moves = legal_moves.count()
        epds = [board.epd()] * nr_of_moves
        use_ab = [use_alpha_beta] * nr_of_moves

        with mp.Pool(number_workers) as pool:
            results = pool.starmap_async(self.multiprocess_starting_point, zip(legal_moves, epds, use_ab))
            results = results.get()

        max_value = -10000
        best_move = None
        for result, move in results:
            if result > max_value:
                max_value = result
                best_move = move

        end_time = time.time()

        if not mute:
            print(f"Value: {max_value:.4f} Computation time: {time_string(end_time - start_time)}")
            print("Make move: ", best_move)

        return best_move

    def _minimize_(self, board: chess.Board, depth: int) -> (float, chess.Move):
        if depth > self.search_depth or board.is_game_over():
            return self.value_function.evaluate_position(board, self.player_color)

        else:
            min_val = 10000

            for move in board.legal_moves:
                # temp = board.copy(stack=False)
                # temp.push(move)
                board.push(move)
                val = self._maximize_(board, depth + 1)

                board.pop()     # redo last move

                if val < min_val:
                    min_val = val

            return min_val

    def _maximize_(self, board: chess.Board, depth: int) -> (float, chess.Move):
        if depth > self.search_depth or board.is_game_over():
            return self.value_function.evaluate_position(board, self.player_color)

        else:
            max_val = -10000

            for move in board.legal_moves:
                # temp = board.copy(stack=False)
                # temp.push(move)
                board.push(move)
                val = self._minimize_(board, depth + 1)

                board.pop()  # redo last move

                if val > max_val:
                    max_val = val

            return max_val

    def _alpha_beta_min_(self, board: chess.Board, depth: int, alpha: int, beta: int) -> (float, chess.Move):
        # alpha = current maximum; beta = current minimum

        if depth > self.search_depth or board.is_game_over():
            return self.value_function.evaluate_position(board, self.player_color)

        else:
            min_val = 10000

            for move in board.legal_moves:
                # temp = board.copy(stack=False)
                # temp.push(move)

                board.push(move)
                val = self._alpha_beta_max_(board, depth + 1, alpha, beta)

                board.pop()

                if val < min_val:
                    min_val = val
                if min_val <= alpha:
                    return min_val

            return min_val

    def _alpha_beta_max_(self, board: chess.Board, depth: int, alpha: int, beta: int) -> (float, chess.Move):
        if depth > self.search_depth or board.is_game_over():
            return self.value_function.evaluate_position(board, self.player_color)

        else:
            max_val = -10000

            for move in board.legal_moves:
                # temp = board.copy(stack=False)
                # temp.push(move)

                board.push(move)
                val = self._alpha_beta_min_(board, depth + 1, alpha, beta)

                board.pop()

                if val > max_val:
                    max_val = val
                if max_val > beta:
                    return max_val

            return max_val

    def multiprocess_starting_point(self, move, epd, use_alpha_beta):
        board = chess.Board(epd)
        board.push(move)

        if use_alpha_beta:
            result = self._alpha_beta_min_(board, 1, -10000, 10000)
        else:
            result = self._minimize_(board, 1)

        return result, move
