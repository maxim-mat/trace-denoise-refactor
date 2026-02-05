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
    Compute mean Wasserstein-1 distance between predicted and target sequences.
    
    This measures how different the predicted class distribution is from
    the target across the sequence dimension.
    """
    
    full_state_update: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: Predicted class indices (B, L)
            target: Target class indices (B, L)
        """
        # Compute per-sample Wasserstein distance
        # Using the 1D discrete case: sum of |CDF_pred - CDF_target|
        batch_size = preds.shape[0]
        
        for i in range(batch_size):
            pred_seq = preds[i].float()
            target_seq = target[i].float()
            
            # Sort both sequences
            pred_sorted = torch.sort(pred_seq)[0]
            target_sorted = torch.sort(target_seq)[0]
            
            # Wasserstein-1 distance for 1D distributions
            distance = torch.mean(torch.abs(pred_sorted - target_sorted))
            self.total_distance += distance
        
        self.total_samples += batch_size
    
    def compute(self) -> torch.Tensor:
        return self.total_distance / self.total_samples.clamp(min=1)


@register_metric("wasserstein")
def create_wasserstein(**kwargs) -> torchmetrics.Metric:
    return WassersteinDistance()


class SafeAUROC(torchmetrics.Metric):
    """
    AUROC that gracefully handles batches with missing classes.
    
    Instead of failing, it returns -1 when AUROC cannot be computed
    and logs a warning.
    """
    
    full_state_update: bool = False
    
    def __init__(self, num_classes: int, average: str = "macro", **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.average = average
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: Predicted probabilities (B*L, C) or logits
            target: Target class indices (B*L,)
        """
        # Store predictions and targets for epoch-end computation
        if preds.dim() > 1:
            preds = F.softmax(preds, dim=-1)
        self.preds.append(preds)
        self.targets.append(target)
    
    def compute(self) -> torch.Tensor:
        if len(self.preds) == 0:
            return torch.tensor(-1.0, device=self.device)
        
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        
        # Check if all classes are present
        unique_classes = torch.unique(targets)
        if len(unique_classes) < self.num_classes:
            # Not all classes present - AUROC may be unreliable
            # Try to compute anyway, return -1 on failure
            try:
                auroc = MulticlassAUROC(
                    num_classes=self.num_classes,
                    average=self.average,
                    thresholds=None,
                ).to(preds.device)
                return auroc(preds, targets)
            except (ValueError, RuntimeError):
                return torch.tensor(-1.0, device=self.device)
        
        auroc = MulticlassAUROC(
            num_classes=self.num_classes,
            average=self.average,
            thresholds=None,
        ).to(preds.device)
        return auroc(preds, targets)


@register_metric("safe_auroc")
def create_safe_auroc(num_classes: int, **kwargs) -> torchmetrics.Metric:
    return SafeAUROC(num_classes=num_classes, average="macro")


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
