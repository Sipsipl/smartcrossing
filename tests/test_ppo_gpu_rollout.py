"""Parity of the torch rollout (rollout_torch.TorchRollout) with ppo.Worker.rollout.

float64, numpy traffic streams and Worker's numpy action noise: the batches (obs, priv,
actions, log-probs, values, advantages, returns, done counts, rewards) must agree over
three rollouts incl. an episode reset. cuda parts are skipped without cuda.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smartcrossing.policy import default_in_scale, layer_sizes  # noqa: E402
from smartcrossing.ppo import MLP, Worker  # noqa: E402
from smartcrossing.ppo_common import N_PRIV  # noqa: E402
from smartcrossing.rollout_torch import TorchRollout  # noqa: E402
from smartcrossing.sim import SimConfig, act_dim, obs_dim  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")


def _setup(B=3, T=60, ep=100, seed=0):
    cfg = SimConfig(n_slots=8, r_time=0.02)
    in_scale = default_in_scale(cfg)
    a_sizes = layer_sizes(cfg, (16, 16))
    c_sizes = [obs_dim(cfg) + N_PRIV, 16, 16, 1]
    A = act_dim(cfg)
    actor, critic = MLP(a_sizes, 0, 0.01), MLP(c_sizes, 1)
    actor.p[-A] = np.arctanh(0.3)
    kw = dict(cfg_d=dict(cfg.__dict__), n_envs=B, seed=seed, horizon=T, episode_steps=ep, gamma=0.99, lam=0.95,
              in_scale=in_scale, act_mask=np.ones(A))
    return kw, actor, critic, np.full(A, -0.5), a_sizes, c_sizes


@pytest.mark.parametrize("device", DEVICES)
def test_torch_rollout_matches_worker(device):
    # cuda (eager, launch-bound): shorter rollouts, still with an episode reset in the third
    kw, actor, critic, logstd, a_sizes, c_sizes = _setup() if device == "cpu" else _setup(T=40, ep=70)
    w = Worker(**kw)
    ro = TorchRollout(**kw, actor_sizes=a_sizes, critic_sizes=c_sizes, device=device, dtype=torch.float64,
                      rng="numpy", noise="numpy")
    kinds = np.zeros(6, np.int64)
    for k in range(3):  # T + T steps, then a reset (T < episode <= 2 T)
        lo, hi = (0.3, 0.5) if k < 2 else (0.4, 0.6)
        ref = w.rollout(actor.p, a_sizes, logstd, critic.p, c_sizes, lo, hi)
        ro.set_params(actor.p, logstd, critic.p)
        got = ro.rollout(lo, hi)
        assert got["car_steps"] == ref["car_steps"] > 0
        np.testing.assert_array_equal(got["kinds"], ref["kinds"])
        for key in ("obs", "priv", "u", "logp", "adv", "ret", "val"):
            g = got[key].cpu().numpy()
            assert g.shape == ref[key].shape and g.dtype == ref[key].dtype, key
            np.testing.assert_allclose(g, ref[key], rtol=1e-6, atol=1e-6, err_msg=key)
        for key in ("own_rew", "queued", "demand_mean"):
            assert got[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-12), key
        kinds += ref["kinds"]
        # perturb the policy so later rollouts differ
        actor.p += 0.01 * np.sin(np.arange(len(actor.p)))
    assert w.n_resets == ro.n_resets == 2
    assert kinds[1:].sum() > 5, kinds  # contains removals (done handling exercised)


@CUDA
def test_cuda_graph_rollout_matches_eager():
    kw, actor, critic, logstd, a_sizes, c_sizes = _setup(B=16, T=40, ep=1000)
    outs = []
    for graph in (True, False):
        ro = TorchRollout(**kw, actor_sizes=a_sizes, critic_sizes=c_sizes, device="cuda", dtype=torch.float64,
                          graph=graph)
        ro.set_params(actor.p, logstd, critic.p)
        if not graph:  # the capture ran 3 warm-up steps
            ro._reset(0.3, 0.5)
            ro.advance(3)
        outs.append([ro.rollout(0.3, 0.5), ro.rollout(0.3, 0.5)])
    for g, e in zip(*outs):
        assert g["car_steps"] == e["car_steps"] > 0
        np.testing.assert_array_equal(g["kinds"], e["kinds"])
        for key in ("obs", "u", "adv", "ret"):
            torch.testing.assert_close(g[key], e[key], rtol=0, atol=1e-6)


@CUDA
def test_cuda_graph_float32_reset_and_shapes():
    kw, actor, critic, logstd, a_sizes, c_sizes = _setup(B=64, T=32, ep=40)
    ro = TorchRollout(**kw, actor_sizes=a_sizes, critic_sizes=c_sizes, device="cuda", dtype=torch.float32)
    ro.set_params(actor.p, logstd, critic.p)
    for k in range(3):  # second rollout starts a new episode in place (graph kept)
        r = ro.rollout(0.3, 0.5)
        n = r["car_steps"]
        assert n > 100 and all(len(r[key]) == n for key in ("obs", "priv", "u", "logp", "adv", "ret", "val"))
        assert all(torch.isfinite(r[key]).all() for key in ("obs", "adv", "ret"))
        torch.testing.assert_close(r["ret"], r["adv"] + r["val"], rtol=0, atol=1e-5)
    assert ro.n_resets == 2 and ro._replay is not None
