"""Definitions shared by the numpy PPO (ppo.py) and its torch parts (rollout_torch.py,
ppo_torch.py), plus the run-tracking helper. No torch import here: ppo.py loads the torch
modules lazily.
"""

import subprocess
from pathlib import Path

import numpy as np

from .sim import DONE_NAMES

LOG2PI = float(np.log(2 * np.pi))  # Gaussian log-likelihood constant
N_DONE = len(DONE_NAMES)  # done kinds counted per rollout (bincount length)

# Privileged critic inputs (not visible to the policy), in this order:
#   age     car age / max_age
#   queue   cars queued in the env / PRIV_QUEUE_SCALE
#   cars    occupied driving slots in the env / n_slots
#   demand  demand of the env * PRIV_DEMAND_SCALE
PRIV_FEATURES = ("age", "queue", "cars", "demand")
N_PRIV = len(PRIV_FEATURES)
PRIV_QUEUE_SCALE = 10.0
PRIV_DEMAND_SCALE = 5.0

ROOT = Path(__file__).resolve().parents[1]


def git_info(root: Path = ROOT) -> dict:
    """{"commit": HEAD hash, "dirty": uncommitted changes to tracked files} of the repo, for
    run tracking (args.json, checkpoint extra); None values if git is unavailable."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8",
                              check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
