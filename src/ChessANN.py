import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


P_DROPOUT = 0.25


class ChessANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.hidden1 = nn.Linear(3200, 4000)
        self.flat_bn1 = nn.BatchNorm1d(4000)

        self.hidden2 = nn.Linear(4000, 1000)
        self.flat_bn2 = nn.BatchNorm1d(1000)

        self.hidden3 = nn.Linear(1000, 50)
        self.flat_bn3 = nn.BatchNorm1d(50)

        self.hidden4 = nn.Linear(50, 3)

        self.flat_dropout = nn.Dropout(P_DROPOUT)
        self.dropout2d = nn.Dropout2d(P_DROPOUT)

    def forward(self, input_features, train=False) -> torch.Tensor:
        x = self.conv1(input_features)
        x = F.relu(x)
        x = self.conv1_bn(x)
        x = self.dropout2d(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_bn(x)
        x = self.dropout2d(x)

        x = self.conv3(x)
        x = F.relu(x)

        if train:
            x = torch.flatten(x, start_dim=1)  # From multidimensional to one dimensional
        else:
            x = torch.flatten(x)  # From multidimensional to one dimensional

        x = F.relu(self.hidden1(x))
        x = self.flat_dropout(x)

        x = F.relu(self.hidden2(x))
        x = self.flat_dropout(x)

        x = F.relu(self.hidden3(x))
        x = self.flat_dropout(x)

        x = F.softmax(self.hidden4(x), dim=-1)

        return x


if __name__ == '__main__':
    # tests if an example tensor propagates correctly through the network
    tensor = torch.zeros([5, 2, 8, 8])
    print(tensor)
    ANN = ChessANN()
    y = ANN.forward(tensor, train=True)
    print(y)
