import sys
import time

import chess

from PlayingEngine import PlayingEngine
from ValueFunction import DumbValueFunction, ChessANNValueFunction


def print_process_bar(i, max_, post_text=""):
    n_bar = 40   # size of progress bar
    j = i / max_
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  Benchmarking {post_text}...")
    sys.stdout.flush()


def bench(function, repetitions: int, name: str, *args):
    timings = []
    print_process_bar(0, repetitions, name)
    for i in range(repetitions):
        start = time.time()

        function(*args)

        end = time.time()
        timings.append(end - start)
        print_process_bar(i+1, repetitions, name)

    sys.stdout.write('\r')
    sys.stdout.write(f"Avg {(name+':'):25} {sum(timings) / len(timings):.4f} sec.\n")
    sys.stdout.flush()


if __name__ == '__main__':
    nr_repetitions = 5
    board = chess.Board()
    # engine = PlayingEngine(ChessANNValueFunction("../models/v6.pt"), search_depth=2, player_color=PlayingEngine.WHITE)
    engine = PlayingEngine(DumbValueFunction(), search_depth=3, player_color=PlayingEngine.WHITE)

    bench(engine.min_max_search, nr_repetitions, "mini max", board, False, True)
    bench(engine.min_max_parallel, nr_repetitions, "mini max /w mp", board, False, True)

    bench(engine.min_max_search, nr_repetitions, "alpha beta", board, True, True)
    bench(engine.min_max_parallel, nr_repetitions, "alpha beta /w mp", board, False, True)
