"""Shared helpers for the post-training pathway notebooks."""
import os, random
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
