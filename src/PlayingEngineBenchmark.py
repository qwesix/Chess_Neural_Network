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

    # print_process_bar(repetitions, repetitions, name)

    sys.stdout.write('\r')
    sys.stdout.write(f"Avg {(name+':'):25} {sum(timings) / len(timings):.4f} sec.\n")
    sys.stdout.flush()


if __name__ == '__main__':
    # a = 3
    # b = 2
    # x = (a < b) * 3 + (a > b) * 5
    # print(x)

    nr_repetitions = 5
    board = chess.Board()
    # engine = PlayingEngine(ChessANNValueFunction("../models/v4.pt"), search_depth=2, player_color=PlayingEngine.WHITE)
    engine = PlayingEngine(DumbValueFunction(), search_depth=3, player_color=PlayingEngine.WHITE)

    bench(engine.min_max_search, nr_repetitions, "mini max", board, False, True)
    bench(engine.min_max_parallel, nr_repetitions, "mini max /w mp", board, False, True)

    bench(engine.min_max_search, nr_repetitions, "alpha beta", board, True, True)
    bench(engine.min_max_parallel, nr_repetitions, "alpha beta /w mp", board, False, True)

# With copying:
# Avg mini max:                 21.8467 sec.
# Avg mini max /w mp:           10.2749 sec.
# Avg alpha beta:               22.0127 sec.
# Avg alpha beta /w mp:         10.4167 sec.

# With popping:
# Avg mini max:                 20.5053 sec.
# Avg mini max /w mp:           9.9952 sec.
# Avg alpha beta:               21.1375 sec.
# Avg alpha beta /w mp:         9.8984 sec.

# More inline if instead of branching
# Avg mini max:                 21.2172 sec.
# Avg mini max /w mp:           10.1174 sec.
# Avg alpha beta:               21.6791 sec.
# Avg alpha beta /w mp:         9.9274 sec.

# Using branch-less programming techniques
# vg mini max:                 20.8554 sec.
# Avg mini max /w mp:           9.9916 sec.
# Avg alpha beta:               20.6568 sec.
# Avg alpha beta /w mp:         10.1584 sec.

# use built-in min and max functions:
# Avg mini max:                 20.9947 sec.
# Avg mini max /w mp:           10.0512 sec.
# Avg alpha beta:               21.1346 sec.
# Avg alpha beta /w mp:         11.1406 sec.
