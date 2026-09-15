"""Shared helpers used across prepare/train/evaluate/serve stages."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """Load the params.yaml file used to drive every DVC stage."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Make DVC pipeline runs reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> torch.device:
    """Fall back to CPU automatically if CUDA was requested but unavailable."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def collate_fn(batch):
    """torchvision detection models expect a list of images + list of targets,
    not a stacked tensor -- so we override the default collate."""
    return tuple(zip(*batch))
