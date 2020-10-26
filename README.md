# Artificial Neural Network for Chess AI with PyTorch
I trained a Convolutional Neural Net (CNN) to precdict the winner of a game for a given chess position (black wins, white wins or draw). 
Records of the games of chess grandmasters were used for training.
They are playing very well but not optimal, therefore the data is sometimes contradictory and the network can't predict the correct outcome with 100% accuracy.

In practice the network reached an accuracy of over 50% on unknown games. That's an advantage over guessing. It's probably possible to increase the performance of
the network by adding one or more additional linear layers or playin around with the convolutional layers. But that would also increase the computation time needed
to propagate through the network.

The programm uses an alpha beta algorithm that uses multiple processes and is perfectly scalable on many processing cores. 
The alpha beta algorithm uses the CNN as evaluation function. It simulates a few turns ahead and evaluates the positions, then tries to maximize it's winning probability.

# Elo-Rating: ???

# Performance
In one word: bad. I use the [python-chess](https://python-chess.readthedocs.io/en/latest/) Board class and it's slow. 
On a 4-core Processor (i7-4790) simulating 4 moves ahead needs up to 20 minutes. (Without using a neural network) 
In future I want to improve the performance by exchanging the the chess engine or using C++.

# Play a game
There is no Graphical User Interface now, it's only possible to play on console. Just start the main script.

# Requirements
- [python-chess](https://python-chess.readthedocs.io/en/latest/)
- [pytorch](pytorch.org)
- TinyDB, scikit-learn, seaborn (Just for Training)
