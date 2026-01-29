import logging
import pickle as pkl
from typing import Any, Mapping, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from src.dataset.dataset import TracesDataset, collate_traces_batch

try:
    import pytorch_lightning as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


logger = logging.getLogger(__name__)


def _get(cfg: Mapping[str, Any], key: str, default=None):
    return cfg.get(key, default)


def _load_dataset(cfg: Mapping[str, Any]) -> TracesDataset:
    data_path = _get(cfg, "data_path")
    if data_path is None:
        raise ValueError("config is missing required field: data_path")
    with open(data_path, "rb") as f:
        payload = pkl.load(f)
    labels = payload.get("target")
    data = payload.get("stochastic")
    if labels is None:
        raise ValueError("dataset payload missing 'target' key")
    return TracesDataset(labels=labels, data=data)


def _split_dataset(dataset: TracesDataset, cfg: Mapping[str, Any]):
    train_percent = float(_get(cfg, "train_percent", 0.8))
    seed = int(_get(cfg, "seed", 0))
    train_size = int(len(dataset) * train_percent)
    val_size = len(dataset) - train_size
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def _build_dataloaders(cfg: Mapping[str, Any]):
    dataset = _load_dataset(cfg)
    train_ds, val_ds = _split_dataset(dataset, cfg)
    batch_size = int(_get(cfg, "batch_size", 8))
    num_workers = int(_get(cfg, "num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_traces_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_traces_batch,
    )
    return train_loader, val_loader


def train(cfg: Mapping[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Entry point for training. This is a Lightning-ready stub: data loading is wired,
    while the LightningModule and Trainer setup can be added next.
    """
    if pl is None:
        raise RuntimeError("pytorch_lightning is required for training")

    seed = int(_get(cfg, "seed", 0))
    pl.seed_everything(seed, workers=True)

    train_loader, val_loader = _build_dataloaders(cfg)
    sample_batch = next(iter(train_loader))
    logger.info(
        "Loaded dataset: train=%d, val=%d, sample_labels=%s, sample_data=%s",
        len(train_loader.dataset),
        len(val_loader.dataset),
        tuple(sample_batch["labels"].shape),
        None if sample_batch["data"] is None else tuple(sample_batch["data"].shape),
    )

    logger.info("Training scaffold ready. Add LightningModule and Trainer next.")
    return train_loader, val_loader
