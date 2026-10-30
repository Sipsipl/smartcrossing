"""Shared test setup. CUDA test cases are opt-in: set SC_TEST_CUDA=1 (they allocate GPU memory and can stall a
running training). Without it they are skipped; CPU torch cases always run."""
import os

import pytest

CUDA_REASON = "cuda tests are opt-in: set SC_TEST_CUDA=1 (needs a free GPU, not during a training run)"
CUDA_ENABLED = os.environ.get("SC_TEST_CUDA", "") == "1"

# decorator for cuda-only tests, parametrize list for tests that run on cpu and (opt-in) cuda
CUDA = pytest.mark.cuda
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a GPU; runs only with SC_TEST_CUDA=1")


def pytest_collection_modifyitems(config, items):
    skip_off = pytest.mark.skip(reason=CUDA_REASON)
    skip_none = pytest.mark.skip(reason="cuda not available")
    avail = None
    for item in items:
        if "cuda" not in item.keywords:
            continue
        if not CUDA_ENABLED:
            item.add_marker(skip_off)
            continue
        if avail is None:
            import torch
            avail = torch.cuda.is_available()
        if not avail:
            item.add_marker(skip_none)


def exit_spawn_scene(cfg, heading_sign=-1.0, v=8.0):
    """One env, demand 0: a car queued on arm 0 (heading north) and a car 10 m ahead of the spawn line on the
    other half of the road (outside the spawn corridor). heading_sign -1: it drives south, out of the map along
    arm 0 (an exiting car: blocks under the old spawn rule, not under cfg.spawn_exit_static); +1: it heads
    north into the arm (a queued car: blocks under both rules if standing)."""
    import dataclasses

    import numpy as np

    from smartcrossing.sim import STRAIGHT, Sim

    sim = Sim(dataclasses.replace(cfg, demand=0.0), 1, seed=0)
    sim.alive[0, 0] = True
    sim.x[0, 0], sim.y[0, 0] = -cfg.W / 2, -cfg.L + 2.5 + 10.0
    sim.th[0, 0], sim.v[0, 0] = heading_sign * np.pi / 2, v
    sim.arm[0, 0], sim.man[0, 0], sim.uid[0, 0] = (2 if heading_sign < 0 else 0), STRAIGHT, 0
    sim.next_uid = 1
    sim._prev_dist[0, 0] = sim.dist_to_gate()[0, 0]
    sim._qbuf[0, 0, 0] = (0.5, STRAIGHT, 0)
    sim.queue[0, 0] = 1
    return sim
