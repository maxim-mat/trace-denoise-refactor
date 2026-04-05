import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Literal, Optional, Dict, Any, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.config import Config
from omegaconf import OmegaConf

from .base_diffusion import BaseDiffusion
from src.modules.learnable_hybrid_loss import LearnableHybridLoss
from src.modules.hybrid_loss import HybridLoss
from src.utils.metrics import create_metric_collection
from src.utils.setup_utils import create_denoiser, create_diffusion


class DiffusionLightningModule(L.LightningModule):
    """
    Lightning module that wraps a denoiser with a diffusion process.
    
    Handles the forward diffusion (noising) during training and 
    reverse diffusion (sampling) during inference.
    
    Supports denoisers with multiple outputs following the convention that
    the PRIMARY diffusion output is ALWAYS FIRST. Auxiliary outputs (e.g.,
    matrix predictions) follow.
    
    IMPORTANT: Metrics are computed on the FULL reverse diffusion output,
    not on single-step denoising predictions. This means:
    - Training loss: computed on single-step predictions (efficient)
    - Validation/Test metrics: computed after running full reverse diffusion (expensive)
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        diffusion: BaseDiffusion,
        eval_diffusion: BaseDiffusion,
        config: Config,
    ):
        super().__init__()
        self.save_hyperparameters(
            OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
            ignore=['denoiser', 'diffusion', 'eval_diffusion'],
        )
        
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.eval_diffusion = eval_diffusion
        self.optimizer = config.optimizer.method
        self.learning_rate = config.optimizer.learning_rate
        self.weight_decay = config.optimizer.weight_decay
        self.scheduler = config.optimizer.scheduler
        self.warmup_epochs = config.optimizer.warmup_epochs
        self.denoiser_output = config.diffusion.denoiser_output
        self.conditional_dropout = config.model.conditional_dropout
        self.auxilary_dropout = config.model.auxilary_dropout
        self.gamma = config.model.gamma
        self.loss_type = config.model.loss_function
        self.num_classes = config.data.num_classes
        self.verbose_test = config.logging.verbose_test
        self.trajectory_metrics = config.metrics.verbose_trajectory or ["accuracy"]
        self.trajectory_save_every = config.metrics.trajectory_save_every
        self.use_padding_mask = config.data.use_padding_mask
        
        reduction = "none" if self.use_padding_mask else "mean"
        ignore_index = config.data.padding_value if not self.use_padding_mask else None
        self.loss_fn = self._create_loss_fn(self.loss_type, reduction=reduction, ignore_index=ignore_index)
        self.ignore_index = config.data.padding_value
        self.log_samples_every_n = config.logging.log_samples_every_n
        
        self._setup_metrics(
            num_classes=self.num_classes,
            val_metrics=config.metrics.val or [],
            test_metrics=config.metrics.test or [],
        )

        self.trajectory_results: List[Dict[str, Any]] = []

    @classmethod
    def load_from_experiment(cls, checkpoint_path: str, map_location=None):
        """
        Reconstructs the model skeleton and pours in the checkpoint weights,
        automatically restoring graph buffers and flow matrices.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        hparams = checkpoint.get("hyper_parameters")
        
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(hparams)

        denoiser = create_denoiser(cfg, flow_matrix=None, graph_data=None)
        diffusion = create_diffusion(cfg.diffusion.sampler, cfg)
        eval_diffusion = create_diffusion("ddim", cfg) if cfg.diffusion.eval_use_ddim else diffusion

        model = cls.load_from_checkpoint(
            checkpoint_path,
            denoiser=denoiser,
            diffusion=diffusion,
            eval_diffusion=eval_diffusion,
            map_location=map_location
        )
        return model

    def _create_loss_fn(self, loss_type: str, reduction: str = "mean", ignore_index: int = None) -> nn.Module:
        """Create a loss function from type string.
        
        Args:
            loss_type: Type of loss function.
            reduction: 'mean' for legacy path, 'none' for mask-based path.
            ignore_index: If set, passed to CrossEntropyLoss at construction time.
        """
        ce_kwargs = {"reduction": reduction}
        if ignore_index is not None:
            ce_kwargs["ignore_index"] = ignore_index

        if loss_type == "mse":
            return nn.MSELoss(reduction=reduction)
        elif loss_type == "l1":
            return nn.L1Loss(reduction=reduction)
        elif loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(**ce_kwargs)
        elif loss_type == "hybrid":
            return HybridLoss(nn.CrossEntropyLoss(**ce_kwargs), nn.BCEWithLogitsLoss(), self.gamma)
        elif loss_type == "learnable_hybrid":
            return LearnableHybridLoss(nn.CrossEntropyLoss(**ce_kwargs), nn.BCEWithLogitsLoss(), self.gamma)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _masked_loss(self, raw_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply padding mask to per-element loss and return mean over real positions.
        
        Args:
            raw_loss: Per-element loss tensor (B, L) from a loss with reduction='none'.
            mask: Boolean mask (B, L), True for real positions.
        """
        return (raw_loss * mask).sum() / mask.sum().clamp(min=1)
    
    def _setup_metrics(
        self,
        num_classes: Optional[int],
        val_metrics: List[str],
        test_metrics: List[str],
    ):
        """Initialize metric collections for validation and test only."""
        self.val_metrics = None
        self.test_metrics = None
        
        if val_metrics:
            self.val_metrics = create_metric_collection(
                val_metrics, num_classes=num_classes, prefix="val/", ignore_index=self.ignore_index
            )
        if test_metrics:
            self.test_metrics = create_metric_collection(
                test_metrics, num_classes=num_classes, prefix="test/", ignore_index=self.ignore_index
            )

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=0.01 * self.learning_rate,
            )
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=50,
            )
        elif self.scheduler == "none":
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def enable_verbose_test(
        self,
        metrics: Optional[List[str]] = None,
        save_every: int = 1,
    ):
        """Enable verbose test mode for trajectory analysis during inference."""
        self.verbose_test = True
        if metrics is not None:
            self.trajectory_metrics = metrics
        self.trajectory_save_every = save_every
        self.trajectory_results = []
    
    def disable_verbose_test(self):
        """Disable verbose test mode."""
        self.verbose_test = False
    
    def get_trajectory_dataframe(self):
        """Convert trajectory results to a pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for get_trajectory_dataframe()")
        
        rows = []
        for result in self.trajectory_results:
            batch_idx = result["batch_idx"]
            for step_data in result["trajectory"]:
                row = {"batch_idx": batch_idx, **step_data}
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Forward pass through denoiser."""
        return self.denoiser(x, t, y)

    def _unpack_batch(self, batch):
        """Unpack batch into (data, labels, mask).
        
        Returns:
            x: data tensor (B, L, C)
            y: labels tensor
            mask: padding mask (B, L) bool or None
        """
        if self.use_padding_mask:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None
        return x, y, mask

    def _compute_loss(self, denoiser_out, target, mask=None):
        """Compute loss with optional mask-based reduction.
        
        In the mask path (reduction='none'), applies the mask to per-element loss.
        In the legacy path (reduction='mean'), ignore_index is already baked into the loss.
        """
        if self.use_padding_mask and mask is not None:
            if self.loss_type in {"hybrid", "learnable_hybrid"}:
                loss = self.loss_fn(denoiser_out, target, padding_mask=mask)
            else:
                raw_loss = self.loss_fn(denoiser_out, target)
                loss = self._masked_loss(raw_loss, mask)
        else:
            loss = self.loss_fn(denoiser_out, target)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step with forward diffusion (Algorithm 1).
        
        Computes single-step denoising loss. Supports multi-output denoisers
        by extracting primary output for loss computation.
        
        Expected batch format: (labels, data) or (labels, data, mask). shapes are (B, L, C)
        """
        x, y, mask = self._unpack_batch(batch)
        x = x.permute(0, 2, 1).float()  # (B, C, L)
        y = y.permute(0, 2, 1).float()
        
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_t, noise = self.diffusion.noise_data(x, t)
        
        if torch.rand(1).item() < self.conditional_dropout:
            y = None

        use_aux = torch.rand(1).item() > self.auxilary_dropout
        
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x

        if self.loss_type in {"hybrid", "learnable_hybrid"}:
            target = (target, self.denoiser.gt_flow_matrix)
        
        denoiser_out = self.denoiser(x_t, t, y, use_aux)
        loss = self._compute_loss(denoiser_out, target, mask)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.loss_type == "hybrid":
            self.log("mixture_ratio", self.gamma, prog_bar=False)
        elif self.loss_type == "learnable_hybrid":
            self.log("mixture_ratio", torch.sigmoid(self.gamma), prog_bar=False)
        
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm before optimizer step."""
        grad_norm = self._compute_grad_norm()
        if grad_norm is not None:
            self.log("grad_norm", grad_norm, prog_bar=False)
    
    def _compute_grad_norm(self, norm_type: float = 2.0) -> Optional[torch.Tensor]:
        """Compute the total gradient norm across all parameters."""
        parameters = [p for p in self.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return None
        
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
        return total_norm
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        - Every epoch: Compute single-step loss (cheap, for monitoring)
        - Every N epochs: Run full reverse diffusion and compute metrics (expensive)
        """
        x, y, mask = self._unpack_batch(batch)
        x = x.permute(0, 2, 1).float()  # (B, C, L)
        
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_t, noise = self.diffusion.noise_data(x, t)
        
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x

        if self.loss_type in {"hybrid", "learnable_hybrid"}:
            target = (target, self.denoiser.gt_flow_matrix)
        
        denoiser_out = self.denoiser(x_t, t, y)
        loss = self._compute_loss(denoiser_out, target, mask)
        
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step with full reverse diffusion for metrics.
        
        When verbose_test is enabled, also evaluates metrics at each point
        along the reverse diffusion trajectory for analysis.
        """
        x, y, mask = self._unpack_batch(batch)
        x = x.permute(0, 2, 1).float()  # (B, C, L)
        
        x_pred = self.eval_diffusion.sample(self.denosier, y.shape[0], (y.shape[1], y.shape[2]), y, self.denoiser_output)
        self.test_metrics.update(x_pred, x)

        return {
            "preds": x_pred.detach(), 
            "targets": x.detach(),
            "batch_idx": batch_idx
        }

    def predict_step(self, batch, batch_idx):
        """
        Predict step.
        
        - Every n batches: Log a sample of the predictions and targets
        """
        x, y, mask = self._unpack_batch(batch)
        x_pred = self.eval_diffusion.sample(self.denosier, y.shape[0], (y.shape[1], y.shape[2]), y, self.denoiser_output)

        return torch.argmax(torch.softmax(x_pred, dim=1), dim=1)
        
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Only log a sample of batches to avoid slowing down testing
        if batch_idx % self.log_samples_every_n != 0:
            return

        preds = outputs["preds"]    # (B, C, L) logits/probs
        targets = outputs["targets"] # (B, C, L) or indices
        
        # Get hard indices for the gallery
        pred_indices = torch.argmax(preds, dim=1).cpu().numpy()
        target_indices = torch.argmax(targets, dim=1).cpu().numpy()

        # Access the loggers
        if self.logger:
            for logger in self.loggers:
                if isinstance(logger, L.pytorch.loggers.WandbLogger):
                    self._log_wandb_gallery(logger, pred_indices, target_indices, batch_idx)
                elif isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                    self._log_tensorboard_vis(logger, pred_indices, batch_idx)

    def _log_wandb_gallery(self, logger, pred_indices, target_indices, batch_idx):
        import wandb
        columns = ["id", "ground_truth", "prediction", "segmentation_vis"]
        table = wandb.Table(columns=columns)

        for i in range(min(len(pred_indices), 5)): # Log 5 samples per batch
            # Assuming you have a visualize_segmentation function
            vis_img = self.visualize_segmentation(pred_indices[i]) 
            
            table.add_data(
                f"b{batch_idx}_s{i}",
                pred_indices[i].tolist(),
                target_indices[i].tolist(),
                wandb.Image(vis_img)
            )
        logger.experiment.log({"test/output_gallery": table})

    def _log_tensorboard_vis(self, logger, pred_indices, batch_idx):
        # Log the first image as a simple qualitative check
        vis_img = self.visualize_segmentation(pred_indices[0])
        # Convert HWC to CHW for TensorBoard
        vis_tensor = torch.from_numpy(vis_img).permute(2, 0, 1)
        logger.experiment.add_image(f"test_vis/{batch_idx}", vis_tensor, self.global_step)

    def _create_heatmap_figure(self, cm: np.ndarray):
        """
        Creates a matplotlib figure representing the confusion matrix.
        cm: Normalized confusion matrix (C, C)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use activity names for labels if available, otherwise indices
        labels = self.hparams.get("activity_names", range(self.num_classes))
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=".2f", 
            cmap="Blues", 
            xticklabels=labels, 
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_title(f"Confusion Matrix - Epoch {self.current_epoch}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        plt.tight_layout()
        
        return fig

    def _log_confusion_matrix(self, conf_matrix):
        # Convert tensor to numpy for easier visualization
        cm = conf_matrix.cpu().numpy()
        
        if self.logger:
            for logger in self.loggers:
                # W&B has a dedicated plot for this
                if isinstance(logger, L.pytorch.loggers.WandbLogger):
                    import wandb
                    logger.experiment.log({
                        "test/conf_matrix": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=None, # You can pass labels if you have them
                            preds=None,
                            cm_template=cm,
                            class_names=self.activity_names # Provided in your MSc data
                        )
                    })
                
                # TensorBoard requires an image (standard heatmap)
                elif isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                    fig = self._create_heatmap_figure(cm) # Use matplotlib to create a figure
                    logger.experiment.add_figure("test/conf_matrix", fig, self.global_step)
                    plt.close(fig)
    
    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        if self.test_metrics is not None:
            metrics = self.test_metrics.compute()
            conf_matrix = metrics.pop("confusion_matrix")
            self._log_confusion_matrix(conf_matrix)
            self.log_dict(metrics, prog_bar=True, sync_dist=True)
            self.test_metrics.reset()
        
        if self.verbose_test and self.trajectory_results:
            self._log_trajectory_summary()
    
    def _log_trajectory_summary(self):
        """Aggregate and log trajectory metrics across all test batches."""
        if not self.trajectory_results:
            return
        
        all_trajectories = [r["trajectory"] for r in self.trajectory_results]
        timesteps = [step["timestep"] for step in all_trajectories[0]]
        
        aggregated = {}
        for step_idx, t in enumerate(timesteps):
            step_metrics = {}
            for traj in all_trajectories:
                if step_idx < len(traj):
                    for key, value in traj[step_idx].items():
                        if key not in ["timestep", "step"]:
                            if key not in step_metrics:
                                step_metrics[key] = []
                            step_metrics[key].append(value)
            
            for key, values in step_metrics.items():
                metric_key = f"traj_t{t}/{key}"
                if metric_key not in aggregated:
                    aggregated[metric_key] = sum(values) / len(values)
        
        if aggregated:
            self.log_dict(aggregated, sync_dist=True)
        
        # Log final timestep metrics prominently
        if all_trajectories and all_trajectories[0]:
            final_step = all_trajectories[0][-1]
            final_metrics = {}
            for key in final_step:
                if key not in ["timestep", "step"]:
                    values = [t[-1][key] for t in all_trajectories if t and key in t[-1]]
                    if values:
                        final_metrics[f"traj_final/{key}"] = sum(values) / len(values)
            if final_metrics:
                self.log_dict(final_metrics, prog_bar=True, sync_dist=True)
    
    def evaluate_trajectory(
        self,
        ground_truth: torch.Tensor,
        labels: torch.Tensor,
        conditioning: torch.Tensor,
        metric_names: List[str],
        save_every: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate metrics at every point along the reverse diffusion trajectory.
        """
        if self.num_classes is None:
            raise ValueError("num_classes must be set to evaluate trajectory")
        
        results = []
        shape = (ground_truth.shape[1], ground_truth.shape[2])
        batch_size = ground_truth.shape[0]
        
        metrics = create_metric_collection(
            metric_names,
            num_classes=self.num_classes,
            prefix="traj/",
        )
        metrics = metrics.to(self.device)
        
        for i, (t, x_t) in enumerate(self.sample_generator(
            batch_size=batch_size,
            shape=shape,
            y=conditioning,
            use_eval_diffusion=True,
        )):
            if i % save_every != 0:
                continue
            
            metrics.reset()
            self._update_metrics(x_t, ground_truth, labels, metrics, "traj")
            
            step_metrics = metrics.compute()
            results.append({
                "timestep": t,
                "step": i,
                **{k.replace("traj/", ""): v.item() if hasattr(v, 'item') else v 
                   for k, v in step_metrics.items()}
            })
        
        return results
    
    @classmethod
    def load_from_checkpoint_with_denoiser(
        cls,
        checkpoint_path: str,
        denoiser: nn.Module,
        map_location=None,
        **kwargs,
    ):
        """
        Load model from checkpoint with a provided denoiser.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        hparams = checkpoint.get("hyper_parameters", {})
        model = cls(denoiser=denoiser, **hparams, **kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        return model
