import torch
import Model
from functools import lru_cache


class CustomDataset(torch.utils.data.Dataset):
    """ Special dataset that is used for ChessANNv2 and delivers two input tensors in stead of one. """

    def __init__(self, positions_list: list):
        self.positions = positions_list

    def __len__(self):
        return len(self.positions)

    @lru_cache(1000000)
    def __getitem__(self, idx):
        sample = self.positions[idx]
        # position, on turn, label
        return Model.process_epd(sample[0]), sample[1], sample[2]
