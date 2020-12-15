import torch


class CustomDataset(torch.utils.data.Dataset):
    """ Special dataset that is used for ChessANNv2 and delivers two input tensors in stead of one. """

    def __init__(self, conv_input, player_input, result):
        self.conv_input = conv_input
        self.player_input = player_input
        self.result = result

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        return self.conv_input[idx], self.player_input[idx], self.result[idx]
