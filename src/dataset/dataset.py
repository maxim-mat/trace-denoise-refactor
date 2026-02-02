import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
import torch.nn.functional as F


def collate_traces_batch(
    batch,
    final_channels,
    padding_value=0,
    one_hot_labels=False,
):
    # final_channels should be at least the initial number of classes
    # if final_channels is more than the initial number of classes, the padding value must be set to final_channels - 1
    # such that one-hot encoding works

    labels = [x[0] for x in batch]
    data = [x[1] for x in batch]

    if final_channels < data[0].shape[-1]:
        raise ValueError(f"final_channels must be at least the initial number of classes, but got {final_channels} < {data[0].shape[-1]}")

    if final_channels > padding_value:
        padding_value = final_channels - 1
    elif padding_value > final_channels:
        raise ValueError(f"padding value must be less than or equal to final channels, but got {padding_value} > {final_channels}")

    padded_labels = pad_sequence(labels, batch_first=True, padding_value=padding_value)
    if one_hot_labels:
        padded_labels = one_hot(padded_labels, num_classes=final_channels)

    padded_data = pad_sequence(data, batch_first=True, padding_value=0.0)
    pad_size = final_channels - padded_data.shape[-1]
    if pad_size > 0:
        padded_data = F.pad(padded_data, (0, pad_size, 0, 0), value=0.0)

    return padded_labels, padded_data


class TracesDataset(Dataset):
    # expected format of data: (B, C, L) or (C, L)
    # expected format of labels: (L,) or (B, L)
    # stored shapes are (L,) and (L, C)
    def __init__(self, labels, data):
        if len(data) != len(labels):
            raise ValueError("data and labels must be of same length")
        self._labels = [torch.as_tensor(item) for item in labels]
        self._data = [torch.as_tensor(item) for item in data]
        
        if self._labels[0].ndim > 2:
            raise ValueError(f"expected labels to have 1 or 2 dims but got {self._labels[0].ndim}, \
                               shape: {self._labels[0].shape}")
        if self._data[0].ndim > 3:
            raise ValueError(f"expected data to have 2 or 3 dims but got {self._data[0].ndim}, \
                               shape: {self._data[0].shape}")

        if self._data[0].ndim == 3:
            self._data = [x.squeeze(0).T for x in self._data]
        if self._labels[0].ndim == 2:
            self._labels = [x.squeeze(0) for x in self._labels]
        
        self.ch_dim = 1
        self.len_dim = 0

        self.n_classes = self._data[0].shape[self.ch_dim]
        self.max_sequence_length = max(x.shape[self.len_dim] for x in self._data)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return self._labels[idx], self._data[idx]
