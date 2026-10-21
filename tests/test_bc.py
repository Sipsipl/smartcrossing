"""Behaviour cloning (bc.py, ppo --init-bc): MLP teacher -> set-encoder actor or wider MLP actor."""

import json
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import policy_set
from smartcrossing.policy import default_in_scale, init_params, layer_sizes
from smartcrossing.sim import SimConfig, obs_dim

ROOT = Path(__file__).resolve().parents[1]


def _arch(cfg, pool, **kw):
    return policy_set.make_arch(cfg, hidden=(16, 12), phi=(10, 8), pool=pool, heads=2, dk=4, dv=3, **kw)


def test_behaviour_cloning_into_set_actor():
    """bc.py: the teacher's states and pre-tanh outputs are recorded (student K > teacher K works),
    and fitting reduces the imitation error."""
    pytest.importorskip("torch")
    from smartcrossing.bc import collect, fit
    from smartcrossing.policy import load_in_scale, load_npz, mlp_linear

    teacher = ROOT / "results" / "round2_best.npz"
    cfg = SimConfig(n_slots=16, n_neighbours=6)
    x, y = collect(str(teacher), cfg, envs=4, seconds=15.0, demand=(0.3, 0.4))
    assert x.shape[1] == obs_dim(cfg) and len(x) == len(y) > 100
    params, sizes, _ = load_npz(teacher)
    n_t = sizes[0]
    np.testing.assert_allclose(mlp_linear(params, sizes, x[:, :n_t] * load_in_scale(teacher)), y, atol=1e-12)
    arch = _arch(cfg, "attn")
    p0 = policy_set.init_params(arch, 0, 0.01)
    scale = default_in_scale(cfg)
    mse0 = float(np.mean((policy_set.forward(p0, arch, x * scale) - y) ** 2))
    p, mse = fit(arch, p0, x * scale, y, steps=150, batch=512, lr=3e-3)
    assert mse < 0.6 * mse0
    assert float(np.mean((policy_set.forward(p, arch, x * scale) - y) ** 2)) == pytest.approx(mse, rel=1e-3)


def test_behaviour_cloning_into_mlp_actor(tmp_path):
    """bc.py also clones into an MLP of other hidden sizes (ppo --arch mlp --hidden 16,16 --init-bc);
    ppo --max-minutes stops after the first iteration past the limit and still writes the final checkpoint."""
    pytest.importorskip("torch")
    from smartcrossing import ppo
    from smartcrossing.bc import collect, fit

    teacher = ROOT / "results" / "round2_best.npz"
    cfg = SimConfig(n_slots=16)
    x, y = collect(str(teacher), cfg, envs=4, seconds=15.0, demand=(0.3, 0.4))
    sizes = layer_sizes(cfg, (16, 16))
    p0 = init_params(sizes, np.random.default_rng(0), out_scale=0.01)
    scale = default_in_scale(cfg)
    from smartcrossing.ppo_torch import mlp_forward
    import torch

    def err(p):
        return float(torch.mean((mlp_forward(torch.as_tensor(p), sizes, torch.as_tensor(x * scale))[0]
                                 - torch.as_tensor(y)) ** 2))

    mse0 = err(p0)
    p, mse = fit(sizes, p0, x * scale, y, steps=200, batch=512, lr=3e-3)
    assert mse < 0.6 * mse0 and err(p) == pytest.approx(mse, rel=1e-3)
    runs = tmp_path / "runs"
    ppo.main(["--name", "m", "--iters", "50", "--max-minutes", "1e-6", "--runs-dir", str(runs), "--workers", "1",
              "--envs", "2", "--horizon", "16", "--demand-start", "0.3", "--hidden", "16,16", "--minibatches", "2", "--epochs", "1",
              "--eval-every", "0", "--init-bc", str(teacher), "--bc-steps", "5", "--bc-seconds", "3",
              "--cfg", json.dumps({"n_slots": 8})])
    log = (runs / "m" / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1 and (runs / "m" / "current.npz").exists()
    assert (runs / "m" / "ckpt_0000.npz").exists()  # the final checkpoint, also with --eval-every 0
