import torch
import torch.nn as nn
import torch.nn.functional as F


P_DROPOUT = 0.2
# TODO: Überanpassung? Weniger Schichten!
# TODO: Multiprocessing im Training!


class ChessANN(nn.Module):
    # Stores for every piece the channel it gets saved in and the value:
    channel_encoder = {
        'K': [0, 0],
        'Q': [0, 1],
        'R': [0, 2],
        'N': [0, 3],
        'B': [0, 4],
        'P': [0, 5],

        'k': [1, 0],
        'q': [1, 1],
        'r': [1, 2],
        'n': [1, 3],
        'b': [1, 4],
        'p': [1, 5]
    }

    def __init__(self):
        super().__init__()
        self.conv0_bn = nn.BatchNorm2d(2)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=2, padding=1)

        # self.flat_bn1 = nn.BatchNorm1d(1936)
        self.hidden1 = nn.Linear(1936, 100)

        # self.flat_bn2 = nn.BatchNorm1d(2000)
        # self.hidden2 = nn.Linear(2000, 50)

        # self.flat_bn3 = nn.BatchNorm1d(100)
        self.hidden3 = nn.Linear(100, 3)

        self.hidden4 = nn.Linear(3, 3)

        self.flat_dropout = nn.Dropout(P_DROPOUT)
        self.dropout2d = nn.Dropout2d(P_DROPOUT)

    def forward(self, x, train=False) -> torch.Tensor:
        x = self.conv0_bn(x)

        x = self.conv1(x)
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

        # x = self.flat_bn1(x)
        x = self.flat_dropout(x)
        x = F.relu(self.hidden1(x))

        # x = self.flat_bn2(x)
        # x = F.relu(self.hidden2(x))

        # x = self.flat_bn3(x)
        x = self.flat_dropout(x)
        x = F.relu(self.hidden3(x))

        x = self.flat_dropout(x)
        x = F.softmax(self.hidden4(x), dim=-1)

        return x

    def process_epd(self, epd_: str) -> torch.Tensor:
        tensor = torch.zeros([2, 8, 8])

        # 2 channels -> for every color one
        # figures encoded like in channel encode
        rows = epd_.split(" ")[0].split("/")
        for i in range(8):
            row = list(rows[i])

            j = 0
            pos = 0
            while j < 8:
                if row[pos] in self.channel_encoder:
                    encoded = self.channel_encoder[row[pos]]
                    tensor[encoded[0]][i][j] = encoded[1]
                else:
                    j += int(row[pos])

                j += 1
                pos += 1

        return tensor


if __name__ == '__main__':
    # tests if an example tensor propagates correctly through the network
    torch.manual_seed(42)
    tensor = torch.zeros([5, 2, 8, 8])
    # print(tensor)
    ANN = ChessANN()
    y = ANN.forward(tensor, train=True)
    print(y)
