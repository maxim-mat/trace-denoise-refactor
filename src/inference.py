import logging
import pickle as pkl
from typing import Any, Mapping

import torch

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


def run_inference(cfg: Mapping[str, Any]):
    """
    Entry point for inference. This is a Lightning-ready stub and should be wired to
    model loading + sampling once the LightningModule is in place.
    """
    if pl is None:
        raise RuntimeError("pytorch_lightning is required for inference")

    dataset = _load_dataset(cfg)
    batch_size = int(_get(cfg, "batch_size", 8))
    num_workers = int(_get(cfg, "num_workers", 0))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_traces_batch,
    )
    sample_batch = next(iter(loader))
    logger.info(
        "Inference scaffold ready. Loaded %d samples, sample_labels=%s, sample_data=%s",
        len(dataset),
        tuple(sample_batch["labels"].shape),
        None if sample_batch["data"] is None else tuple(sample_batch["data"].shape),
    )
    logger.info("Wire this to model checkpoint loading and sampling next.")
