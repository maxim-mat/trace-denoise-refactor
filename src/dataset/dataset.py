import torch
from torch.utils.data import Dataset


def collate_traces_batch(
    batch,
    num_tokens,
    padding_value=0,
    one_hot_labels=False,
):
    labels, data = zip(*batch)
    labels = [torch.as_tensor(item) for item in labels]
    data = list(data)

    label_tokens = labels
    if labels[0].ndim == 2:
        label_tokens = [torch.argmax(item, dim=-1) for item in labels]

    if one_hot_labels:
        one_hot = []
        for item in label_tokens:
            item = item.long()
            one_hot.append(torch.nn.functional.one_hot(item, num_classes=num_tokens).float())
        padded_labels, label_lengths, label_mask = _pad_one_hot(
            one_hot,
            num_tokens=num_tokens,
        )
    else:
        padded_labels, label_lengths, label_mask = _pad_tokens(label_tokens, pad_token_id=0)

    if all(item is None for item in data):
        return {
            "labels": padded_labels,
            "data": None,
            "label_lengths": label_lengths,
            "data_lengths": None,
            "label_mask": label_mask,
            "data_mask": None,
            "label_tokens": _pad_tokens(label_tokens, pad_token_id=0)[0],
            "pad_token_id": 0,
            "num_tokens": num_tokens,
        }

    if any(item is None for item in data):
        raise ValueError("data must be provided for all batch items or for none")

    data = [torch.as_tensor(item) for item in data]
    data_lengths = torch.tensor([item.shape[0] for item in data], dtype=torch.long)
    max_len = int(data_lengths.max().item())
    feat_dim = data[0].shape[1]
    padded_data = torch.zeros((len(data), max_len, feat_dim), dtype=data[0].dtype)
    for i, item in enumerate(data):
        padded_data[i, : item.shape[0]] = item
    data_mask = torch.arange(max_len).unsqueeze(0) < data_lengths.unsqueeze(1)

    return {
        "labels": padded_labels,
        "data": padded_data,
        "label_lengths": label_lengths,
        "data_lengths": data_lengths,
        "label_mask": label_mask,
        "data_mask": data_mask,
        "label_tokens": _pad_tokens(label_tokens, pad_token_id=0)[0],
        "pad_token_id": 0,
        "num_tokens": num_tokens,
    }


class TracesDataset(Dataset):
    # expected format of data: (B, C, L) or (C, L)
    # expected format of labels: (L,) or (B, L)
    def __init__(self, labels, data=None):
        if data is not None and len(data) != len(labels):
            raise ValueError("data and labels must be of same length")
        self._labels = [torch.as_tensor(item) for item in labels]
        self._data = [torch.as_tensor(item) for item in data] if data is not None else None
        
        if self._data[0].ndim == 3:
            self.ch_dim = 1
            self.len_dim = 2
        elif self._data[0].ndim == 2:
            self.ch_dim = 0
            self.len_dim = 1
        else:
            raise ValueError(f"expected labels to have 2 or 3 dims but got {self._labels[0].ndim}, \
                               shape: {self._labels[0].shape}")

        self.n_classes = self._data[0].shape[self.ch_dim]
        self.max_sequence_length = max(x.shape[self.len_dim] for x in self._data)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return self._labels[idx], self._data[idx] if self._data is not None else None





def _pad_one_hot(seqs, num_tokens):
    lengths = torch.tensor([item.shape[0] for item in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(seqs), max_len, num_tokens), dtype=seqs[0].dtype)
    for i, item in enumerate(seqs):
        padded[i, : item.shape[0]] = item
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, lengths, mask


def _pad_tokens(seqs, pad_token_id=0):
    lengths = torch.tensor([item.shape[0] for item in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.full((len(seqs), max_len), pad_token_id, dtype=seqs[0].dtype)
    for i, item in enumerate(seqs):
        padded[i, : item.shape[0]] = item
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, lengths, mask
