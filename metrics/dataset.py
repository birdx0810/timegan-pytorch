import numpy as np
import torch

class FeaturePredictionDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    - idx (int): the index of the feature to be predicted
    """
    def __init__(self, data, time, idx):
        no, seq_len, dim = data.shape
        self.X = torch.FloatTensor(
            np.concatenate(
                (data[:, :, :idx], data[:, :, (idx+1):]), 
                axis=2
            )
        )
        self.T = torch.LongTensor(time)
        self.Y = torch.FloatTensor(np.reshape(data[:, :, idx], [no, seq_len, 1]))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]

class OneStepPredictionDataset(torch.utils.data.Dataset):
    r"""The dataset for predicting the feature of `idx` given the other features
    Args:
    - data (np.ndarray): the dataset to be trained on (B x S x F)
    """
    def __init__(self, data, time):
        self.X = torch.FloatTensor(data[:, :-1, :])
        self.T = torch.LongTensor([t-1 if t == 100 else t for t in time])
        self.Y = torch.FloatTensor(data[:, 1:, :])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx], self.Y[idx]