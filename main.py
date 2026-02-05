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

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from src.utils.config import Config
from src.train import train
from src.inference import run_inference


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path, overrides: list[str]) -> Config:
    """
    Load configuration from YAML file with CLI overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: List of dotlist overrides (e.g., ["trainer.max_epochs=50"])
        
    Returns:
        Structured Config object
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML config
    file_cfg = OmegaConf.load(config_path)
    
    # Create structured config with defaults
    schema = OmegaConf.structured(Config)
    
    # Merge: schema (defaults) <- file config <- CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(schema, file_cfg, cli_cfg)
    else:
        cfg = OmegaConf.merge(schema, file_cfg)
    
    # Convert to structured Config object for type safety
    return OmegaConf.to_object(cfg)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trace Denoising with Diffusion Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in dotlist format (e.g., trainer.max_epochs=50)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("Loading config from: %s", args.config)
    cfg = load_config(args.config, args.overrides)
    
    logger.info("Mode: %s", cfg.mode)
    logger.info("Config:\n%s", OmegaConf.to_yaml(OmegaConf.structured(cfg)))
    
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "inference":
        run_inference(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
