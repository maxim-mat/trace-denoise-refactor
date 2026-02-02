import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
from omegaconf import OmegaConf

from src.train import train
from src.inference import run_inference


logger = logging.getLogger(__name__)


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    if OmegaConf is not None and path.suffix.lower() in {".yaml", ".yml"}:
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)

    with open(path, "r") as f:
        return json.load(f)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--mode", choices=["train", "inference"], default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    cfg = _load_config(Path(args.config))
    mode = args.mode or cfg.get("mode", "train")

    if mode == "train":
        train(cfg)
    elif mode == "inference":
        run_inference(cfg)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
