import argparse
import json
import logging
import os
import pickle as pkl
from utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default=r"config.json", help="configuration path")
    parser.add_argument('--save_path', default=None, help="path for checkpoints and results")
    parser.add_argument('--resume', default=0, type=int, help="resume previous run")
    args = parser.parse_args()
    return args


def initialize():
    args = parse_args()
    with open(args.cfg_path, "r") as f:
        cfg_json = json.load(f)
        cfg = Config(**cfg_json)
    with open(cfg.data_path, "rb") as f:
        dataset = pkl.load(f)
    if args.save_path is None:
        args.save_path = cfg.summary_path
    for path in (args.save_path, cfg.summary_path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(cfg.summary_path, "cfg.json"), "w") as f:
        json.dump(cfg_json, f)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return args, cfg, dataset, logger
