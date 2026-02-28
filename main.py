#!/usr/bin/env python3
"""
Trace Denoising with Diffusion Models - Lightning Implementation

Usage:
    # Training
    python main.py --config configs/train.yaml
    
    # Training with overrides
    python main.py --config configs/train.yaml trainer.max_epochs=50 optimizer.learning_rate=1e-3
    
    # Resume training
    python main.py --config configs/train.yaml resume_from=outputs/checkpoints/last.ckpt
    
    # Inference
    python main.py --config configs/inference.yaml
    python main.py --config configs/train.yaml mode=inference checkpoint_path=outputs/checkpoints/best.ckpt
"""
import logging
import sys

from omegaconf import OmegaConf

from src.train import train
from src.inference import run_inference
from src.utils.initialization import load_config, parse_args


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("Loading config from: %s", args.config)
    cfg = load_config(args.config, args.overrides)
    
    logger.info("Config:\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    
    if args.train:
        train(cfg)
    else:
        run_inference(cfg)

if __name__ == "__main__":
    main()
