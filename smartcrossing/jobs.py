"""Small helpers shared by the measurement commands (evalset, select, deadlock sweep, cloud, ppo eval):
a process pool map with the ``spawn`` start method (Windows has no fork; the same on every OS) and appending
JSON lines (utf-8, LF)."""

import json
import multiprocessing as mp
from pathlib import Path


def pool_map(fn, jobs, workers: int = 1) -> list:
    """[fn(job) for job in jobs], in order; workers > 1: a spawn process pool (fn must be a top-level function)."""
    jobs = list(jobs)
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            return pool.map(fn, jobs, chunksize=1)
    return [fn(j) for j in jobs]


def append_jsonl(path, rows) -> None:
    """Append one ``json.dumps(row)`` line per row to ``path`` (parent folders are created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
