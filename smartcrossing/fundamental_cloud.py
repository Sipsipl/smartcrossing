"""Fundamental-diagram scatter cloud (``cli cloud``, results/fundamental_cloud.json): single measured windows of
driving cars on the map vs exits per minute, per driver and demand, in the default world on held-out traffic seeds.

    poetry run python -m smartcrossing.cli cloud --driver brain=policy:results/round7_best.npz --driver light \
        --driver reservation --out results/fundamental_cloud.json --workers 4

``cloud_windows`` is the window definition; the start page's live dot uses the same one (site/src/js/fdwindow.js,
checked by tests/test_site_js.py).
"""

import time

import numpy as np

from .deadlock import WARMUP
from .drivers import make_driver
from .jobs import pool_map
from .protocol import REPORT_SEEDS, SWEEP_DEMANDS
from .sim import Sim, SimConfig

CLOUD_WINDOW_S, CLOUD_STRIDE_S, CLOUD_SECONDS, CLOUD_ENVS, CLOUD_SEED = 60.0, 10.0, 270.0, 6, REPORT_SEEDS[0]  # held-out: seeds 4000..4006
CLOUD_PER_DEMAND = 60  # points kept per (driver, demand) after thinning


def cloud_windows(cars: np.ndarray, exits: np.ndarray, dt: float, stride_s: float = CLOUD_STRIDE_S):
    """The cloud's window definition (the start page's live dot uses the same, js/fdwindow.js): cars (T, B) =
    driving cars on the map before step i, exits (T + 1, B) = cumulative exits after i steps. A CLOUD_WINDOW_S
    window ending every stride_s, only windows lying entirely after the WARMUP from the empty start (so the first
    one ends at WARMUP + CLOUD_WINDOW_S). Returns [(end time s, density (B,), throughput per min (B,)), ...]:
    density = mean cars over the window, throughput = exits in the window per minute."""
    T = len(cars)
    w, stride = int(round(CLOUD_WINDOW_S / dt)), int(round(stride_s / dt))
    first = int(round(WARMUP / dt)) + w
    return [(end * dt, cars[end - w:end].mean(0), (exits[end] - exits[end - w]) * (60.0 / CLOUD_WINDOW_S))
            for end in range(first, T + 1, stride)]


def cloud_job(job) -> list[list[float]]:
    """One cloud cell (spec, cfg_d, demand, seed, envs, seconds) -> [[density, throughput, demand], ...] per env
    and window (cloud_windows)."""
    spec, cfg_d, demand, seed, envs, seconds = job
    cfg = SimConfig(**{**(cfg_d or {}), "demand": demand})
    sim = Sim(cfg, envs, seed=seed)
    drv = make_driver(spec, sim)
    T = int(round(seconds / cfg.dt))
    cars = np.zeros((T, envs))
    exits = np.zeros((T + 1, envs))
    for i in range(T):
        cars[i] = sim.alive.sum(1)
        sim.step(drv.act())
        exits[i + 1] = sim.stats.exits
    out = []
    for _, dens, thr in cloud_windows(cars, exits, cfg.dt):
        out += [[round(float(d), 1), round(float(t), 1), demand] for d, t in zip(dens, thr)]
    return out


def cloud(drivers: dict, demands=SWEEP_DEMANDS, envs=CLOUD_ENVS, seconds=CLOUD_SECONDS, seed=CLOUD_SEED,
          per_demand=CLOUD_PER_DEMAND, cfg_d=None, workers=1, log=print) -> dict:
    """Fundamental-diagram scatter cloud: {key: spec} -> {"meta", "drivers": {key: {"points": [[density,
    throughput, demand], ...]}}}; each cell thinned evenly to ``per_demand`` points."""
    jobs = [(spec, cfg_d, dm, seed + k, envs, seconds) for spec in drivers.values() for k, dm in enumerate(demands)]
    t0 = time.time()
    cells = pool_map(cloud_job, jobs, workers)
    out = {"meta": {
        "density": "mean number of driving cars on the map over the window (as the live dot on the start page)",
        "throughput": f"exits in the window per minute ({CLOUD_WINDOW_S:g}-s sliding window, one every "
                      f"{CLOUD_STRIDE_S:g} s, windows entirely after the {WARMUP:g} s warm-up)",
        "points": "[density, throughput, demand]; per driver and demand evenly thinned",
        "protocol": f"{envs} envs x {seconds:g} s per demand {list(demands)}, default world, traffic seed "
                    f"{seed} + k for the k-th demand ({seed}..{seed + len(demands) - 1})"},
        "drivers": {}}
    for ki, key in enumerate(drivers):
        pts = []
        for cell in cells[ki * len(demands):(ki + 1) * len(demands)]:
            if len(cell) > per_demand:
                cell = [cell[j] for j in np.linspace(0, len(cell) - 1, per_demand).round().astype(int)]
            pts += cell
        out["drivers"][key] = {"spec": drivers[key], "points": pts}
        log(f"cloud {key}: {len(pts)} points")
    log(f"cloud: {len(jobs)} cells in {time.time() - t0:.0f} s")
    return out
