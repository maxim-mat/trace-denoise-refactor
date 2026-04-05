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
    target_length=None,
):
    # final_channels should be at least the initial number of classes
    # if final_channels is more than the initial number of classes, the padding value must be set to final_channels - 1
    # such that one-hot encoding works
    # target_length: if specified, pad all sequences to this length (must be >= max sequence length in batch)

    labels = [x[0] for x in batch]
    data = [x[1] for x in batch]
    num_classes = data[0].shape[-1]

    if final_channels < num_classes:
        raise ValueError(f"final_channels must be at least the initial number of classes, but got {final_channels} < {data[0].shape[-1]}")

    if final_channels > padding_value:
        padding_value = final_channels - 1
    elif padding_value > final_channels:
        raise ValueError(f"padding value must be less than or equal to final channels, but got {padding_value} > {final_channels}")

    padded_labels = pad_sequence(labels, batch_first=True, padding_value=padding_value)
    if one_hot_labels:
        padded_labels = one_hot(padded_labels, num_classes=final_channels)

    padded_data = pad_sequence(data, batch_first=True, padding_value=0.0)
    
    # Extend to target_length if specified
    if target_length is not None and padded_data.shape[1] < target_length:
        length_pad = target_length - padded_data.shape[1]
        padded_data = F.pad(padded_data, (0, 0, 0, length_pad), value=0.0)
        padded_labels = F.pad(padded_labels, (0, length_pad), value=padding_value)
    
    # Extend channel dimension to final_channels if needed
    pad_size = final_channels - padded_data.shape[-1]
    if pad_size > 0:
        padded_data = F.pad(padded_data, (0, pad_size, 0, 0), value=0.0)

    return padded_labels, padded_data


def collate_traces_batch_probabilistic(
    batch,
    final_channels,
    padding_value=0,
    one_hot_labels=False,
    target_length=None,
):
    # For probabilistic data tensors of shape (L_i, C) where each position is a probability distribution over C classes.
    # This function adds a padding indicator channel:
    # - Padded positions: values at indexes < C are 0, value at index C (padding indicator) is 1
    # - Unpadded positions: values at indexes < C are untouched, value at index C (padding indicator) is 0
    # final_channels should be at least the initial number of classes + 1 (for the padding indicator)
    # target_length: if specified, pad all sequences to this length (must be >= max sequence length in batch)

    labels = [x[0] for x in batch]
    data = [x[1] for x in batch]

    num_classes = data[0].shape[-1]

    if final_channels < num_classes + 1:
        raise ValueError(f"final_channels must be at least num_classes + 1 for padding indicator, but got {final_channels} < {num_classes + 1}")

    if final_channels > padding_value:
        padding_value = final_channels - 1
    elif padding_value > final_channels:
        raise ValueError(f"padding value must be less than or equal to final channels, but got {padding_value} > {final_channels}")

    padded_labels = pad_sequence(labels, batch_first=True, padding_value=padding_value)
    if one_hot_labels:
        padded_labels = one_hot(padded_labels, num_classes=final_channels)

    # Store original lengths before padding
    lengths = [x.shape[0] for x in data]

    # Add padding indicator channel (initialized to 0) to each data tensor
    data_with_indicator = [
        torch.cat([x, torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=-1) 
        for x in data
    ]

    # Pad sequences along length dimension (padding value 0.0)
    padded_data = pad_sequence(data_with_indicator, batch_first=True, padding_value=0.0)

    # Determine final length (use target_length if specified, otherwise max_len from pad_sequence)
    max_len = padded_data.shape[1]
    final_len = max(max_len, target_length) if target_length is not None else max_len

    # Extend to target_length if specified
    if target_length is not None and max_len < target_length:
        length_pad = target_length - max_len
        padded_data = F.pad(padded_data, (0, 0, 0, length_pad), value=0.0)
        padded_labels = F.pad(padded_labels, (0, length_pad), value=padding_value)

    # Set padding indicator to 1 for all padded positions (both from pad_sequence and target_length extension)
    for i, length in enumerate(lengths):
        if length < final_len:
            padded_data[i, length:, num_classes] = 1.0

    # Extend channel dimension to final_channels if needed
    pad_size = final_channels - padded_data.shape[-1]
    if pad_size > 0:
        padded_data = F.pad(padded_data, (0, pad_size, 0, 0), value=0.0)

    return padded_labels, padded_data


def collate_traces_mask(batch, one_hot_labels=False):
    """Collate with a boolean padding mask instead of a sentinel padding value.

    Returns:
        padded_labels: (B, L) long or (B, L, C) float if one_hot_labels
        padded_data:   (B, L, C) float
        padding_mask:  (B, L) bool — True for real positions, False for padding
    """
    labels = [x[0] for x in batch]
    data = [x[1] for x in batch]
    num_classes = data[0].shape[-1]

    # Record original lengths before padding
    lengths = torch.tensor([x.shape[0] for x in labels])

    # Pad sequences — fill value is irrelevant, the mask is the source of truth
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    padded_data = pad_sequence(data, batch_first=True, padding_value=0.0)

    # Boolean mask: True where content is real, False where padded
    max_len = padded_labels.shape[1]
    padding_mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    if one_hot_labels:
        padded_labels = one_hot(padded_labels, num_classes=num_classes)

    return padded_labels, padded_data, padding_mask


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
