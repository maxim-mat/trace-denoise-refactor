"""
Flexible metrics system for diffusion model training.

Supports configurable metrics via YAML config with proper handling of:
- Missing classes in batches (common with rare activities)
- Distributed training
- Automatic device placement
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Literal, Any, Callable
from dataclasses import dataclass, field
from scipy.stats import wasserstein_distance

import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
)


# Registry of available metrics
METRIC_REGISTRY: Dict[str, Callable[..., torchmetrics.Metric]] = {}


def register_metric(name: str):
    """Decorator to register a metric factory function."""
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator


@register_metric("accuracy")
def create_accuracy(num_classes: int, **kwargs) -> torchmetrics.Metric:
    return MulticlassAccuracy(num_classes=num_classes, average="micro")


@register_metric("accuracy_macro")
def create_accuracy_macro(num_classes: int, **kwargs) -> torchmetrics.Metric:
    return MulticlassAccuracy(num_classes=num_classes, average="macro")


@register_metric("precision")
def create_precision(num_classes: int, **kwargs) -> torchmetrics.Metric:
    return MulticlassPrecision(num_classes=num_classes, average="macro")


@register_metric("recall")
def create_recall(num_classes: int, **kwargs) -> torchmetrics.Metric:
    return MulticlassRecall(num_classes=num_classes, average="macro")


@register_metric("f1")
def create_f1(num_classes: int, **kwargs) -> torchmetrics.Metric:
    return MulticlassF1Score(num_classes=num_classes, average="macro")


@register_metric("auroc")
def create_auroc(num_classes: int, **kwargs) -> torchmetrics.Metric:
    # average="macro" and thresholds=None for memory efficiency
    # ignore_index can be set if there's a padding class
    return MulticlassAUROC(
        num_classes=num_classes, 
        average="macro",
        thresholds=None,  # Compute exact AUROC
    )


@register_metric("auroc_weighted")
def create_auroc_weighted(num_classes: int, **kwargs) -> torchmetrics.Metric:
    return MulticlassAUROC(
        num_classes=num_classes,
        average="weighted",
        thresholds=None,
    )


@register_metric("confusion_matrix")
def create_confusion_matrix(num_classes: int, **kwargs) -> torchmetrics.Metric:
    return MulticlassConfusionMatrix(num_classes=num_classes, normalize="true")


class WassersteinDistance(torchmetrics.Metric):
    """
    Computes the mean Wasserstein-1 distance between predicted probability 
    distributions and deterministic targets across the sequence dimension.
    """
    full_state_update: bool = False

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Metric states for global reduction across multiple GPUs
        self.add_state("total_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        """
        Args:
            logits: Predicted raw outputs of shape (B, C, L)
            target: Ground truth class indices of shape (B, L)
        """
        # 1. Convert logits to probabilities and then to CDF
        # We move to (B, L, C) for easier alignment with target one-hot
        probs = torch.softmax(logits, dim=1).permute(0, 2, 1)
        pred_cdf = torch.cumsum(probs, dim=-1)

        # 2. Convert integer targets to one-hot, then to CDF
        target_one_hot = torch.nn.functional.one_hot(
            target, num_classes=self.num_classes
        ).float()
        target_cdf = torch.cumsum(target_one_hot, dim=-1)

        # 3. Compute L1 distance between CDFs
        # Summing over the class dimension gives W1 per (batch, step)
        w1_per_step = torch.sum(torch.abs(pred_cdf - target_cdf), dim=-1)

        # 4. Handle Masking and State Updates
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            # Only sum distances for non-padded elements
            self.total_distance += w1_per_step[mask].sum()
            self.total_samples += mask.sum()
        else:
            self.total_distance += w1_per_step.sum()
            self.total_samples += target.numel()

    def compute(self) -> torch.Tensor:
        # Final global mean calculation
        return self.total_distance / self.total_samples.clamp(min=1)


@register_metric("wasserstein")
def create_wasserstein(ignore_index: Optional[int] = None, **kwargs) -> torchmetrics.Metric:
    return WassersteinDistance(ignore_index=ignore_index)


def create_metric_collection(
    metric_names: List[str],
    num_classes: int,
    prefix: str = "",
    **kwargs,
) -> MetricCollection:
    """
    Create a MetricCollection from a list of metric names.
    
    Args:
        metric_names: List of metric names (must be in METRIC_REGISTRY)
        num_classes: Number of classes for classification metrics
        prefix: Prefix for metric names (e.g., "train_", "val_")
        **kwargs: Additional kwargs passed to metric factories
        
    Returns:
        MetricCollection with all requested metrics
    """
    metrics = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            available = list(METRIC_REGISTRY.keys())
            raise ValueError(
                f"Unknown metric: {name}. Available metrics: {available}"
            )
        metrics[name] = METRIC_REGISTRY[name](num_classes=num_classes, **kwargs)
    
    return MetricCollection(metrics, prefix=prefix)


def get_available_metrics() -> List[str]:
    """Return list of available metric names."""
    return list(METRIC_REGISTRY.keys())
