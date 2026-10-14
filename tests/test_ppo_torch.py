import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smartcrossing.policy import default_in_scale, layer_sizes  # noqa: E402
from smartcrossing.ppo import MLP, Adam, Worker, ppo_update  # noqa: E402
from smartcrossing.ppo_common import N_PRIV  # noqa: E402
from smartcrossing.ppo_torch import TorchPPO  # noqa: E402
from smartcrossing.sim import SimConfig, act_dim, obs_dim  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def batch():
    cfg = SimConfig(n_slots=8, r_time=0.02)
    in_scale = default_in_scale(cfg)
    a_sizes = layer_sizes(cfg, (16, 16))
    c_sizes = [obs_dim(cfg) + N_PRIV, 16, 16, 1]
    A = act_dim(cfg)
    act_mask = np.ones(A)
    w = Worker(dict(cfg.__dict__), 3, 0, 60, 1000, 0.99, 0.95, in_scale, act_mask)
    actor, critic = MLP(a_sizes, 0, 0.01), MLP(c_sizes, 1)
    logstd = np.full(A, -0.5)
    data = w.rollout(actor.p, a_sizes, logstd, critic.p, c_sizes, 0.3, 0.5)
    opt_a, opt_c = Adam(len(actor.p), 3e-4), Adam(len(critic.p), 1e-3)
    # one numpy update first so Adam moments / step count are non-trivial
    a = SimpleNamespace(epochs=1, minibatches=4, clip=0.2, ent=0.0, target_kl=0.0, max_grad=1.0, lr_std=3e-3,
                        logstd_min=-2.5)
    ppo_update(actor, logstd, critic, opt_a, opt_c, data, a, act_mask, in_scale, np.random.default_rng(9))
    return dict(actor=actor, critic=critic, logstd=logstd, opt_a=opt_a, opt_c=opt_c, data=data,
                act_mask=act_mask, in_scale=in_scale)


def _run(b, fn, a, seed):
    st = {k: copy.deepcopy(b[k]) for k in ("actor", "critic", "logstd", "opt_a", "opt_c")}
    out = fn(st["actor"], st["logstd"], st["critic"], st["opt_a"], st["opt_c"], b["data"], a, b["act_mask"],
             b["in_scale"], np.random.default_rng(seed))
    return st, out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ent,max_grad", [(0.0, 1.0), (0.01, 0.05)])  # second: grad clipping active
def test_torch_update_matches_numpy_float64(batch, device, ent, max_grad):
    a = SimpleNamespace(epochs=1, minibatches=8, clip=0.2, ent=ent, target_kl=0.0, max_grad=max_grad,
                        lr_std=3e-3, logstd_min=-2.5)
    n_st, n_out = _run(batch, ppo_update, a, 3)
    t_st, t_out = _run(batch, TorchPPO(device, "float64").update, a, 3)
    assert n_out["epochs_done"] == t_out["epochs_done"] == 1
    for k in ("kl", "clipfrac", "vloss", "pg"):
        assert t_out[k] == pytest.approx(n_out[k], rel=1e-9, abs=1e-12), k
    tol = dict(rtol=0, atol=1e-12)
    np.testing.assert_allclose(t_st["actor"].p, n_st["actor"].p, **tol)
    np.testing.assert_allclose(t_st["critic"].p, n_st["critic"].p, **tol)
    np.testing.assert_allclose(t_st["logstd"], n_st["logstd"], **tol)
    for o in ("opt_a", "opt_c"):
        assert t_st[o].t == n_st[o].t
        np.testing.assert_allclose(t_st[o].m, n_st[o].m, **tol)
        np.testing.assert_allclose(t_st[o].v, n_st[o].v, rtol=1e-9, atol=1e-20)
    # the update really moved the parameters
    assert np.abs(n_st["actor"].p - batch["actor"].p).max() > 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_torch_update_float32_multi_epoch_with_early_stop(batch, device):
    a = SimpleNamespace(epochs=3, minibatches=4, clip=0.2, ent=0.0, target_kl=0.03, max_grad=1.0, lr_std=3e-3,
                        logstd_min=-2.5)
    n_st, n_out = _run(batch, ppo_update, a, 5)
    t_st, t_out = _run(batch, TorchPPO(device, "float32").update, a, 5)
    assert t_out["epochs_done"] == n_out["epochs_done"]
    np.testing.assert_allclose(t_st["actor"].p, n_st["actor"].p, rtol=0, atol=1e-4)
    np.testing.assert_allclose(t_st["critic"].p, n_st["critic"].p, rtol=0, atol=1e-4)
    assert t_out["vloss"] == pytest.approx(n_out["vloss"], rel=1e-3)
