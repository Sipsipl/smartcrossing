import numpy as np

from smartcrossing.policy import PolicyDriver, default_in_scale, forward, layer_sizes, save_npz
from smartcrossing.ppo import MLP, N_PRIV, Worker, mlp_forward
from smartcrossing.sim import Sim, SimConfig, act_dim, obs_dim


def test_mlp_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 6, 3], rng)
    x = rng.normal(size=(4, 5))
    gout = rng.normal(size=(4, 3))
    y, hs = net.forward(x, keep=True)
    g = net.backward(hs, gout)
    num = np.zeros_like(net.p)
    for i in range(len(net.p)):
        old = net.p[i]
        net.p[i] = old + 1e-6
        fp = (net.forward(x) * gout).sum()
        net.p[i] = old - 1e-6
        fm = (net.forward(x) * gout).sum()
        net.p[i] = old
        num[i] = (fp - fm) / 2e-6
    np.testing.assert_allclose(g, num, atol=1e-6)
    np.testing.assert_allclose(mlp_forward(net.p, net.sizes, x), y)


def test_export_is_tanh_of_mean(tmp_path):
    cfg = SimConfig(n_slots=8, demand=0.5)
    sizes = layer_sizes(cfg, (16, 16))
    net = MLP(sizes, 1)
    scale = default_in_scale(cfg)
    save_npz(tmp_path / "p.npz", net.p, sizes, cfg, scale)
    sim = Sim(cfg, 2, seed=0)
    for _ in range(40):
        sim.step(np.zeros((2, 8, act_dim(cfg))))
    drv = PolicyDriver.load(tmp_path / "p.npz", sim)
    obs = sim.observe()
    np.testing.assert_allclose(drv.act(), np.tanh(net.forward(obs * scale)))
    np.testing.assert_allclose(forward(net.p, obs, sizes, scale), np.tanh(net.forward(obs * scale)))


def test_worker_rollout_trajectories():
    cfg = SimConfig(n_slots=8, r_time=0.02)
    in_scale = default_in_scale(cfg)
    a_sizes = layer_sizes(cfg, (16, 16))
    c_sizes = [obs_dim(cfg) + N_PRIV, 16, 16, 1]
    w = Worker(dict(cfg.__dict__), 3, 0, 60, 1000, 0.99, 0.95, in_scale, np.ones(act_dim(cfg)))
    actor, critic = MLP(a_sizes, 0, 0.01), MLP(c_sizes, 1)
    r = w.rollout(actor.p, a_sizes, np.full(act_dim(cfg), -0.5), critic.p, c_sizes, 0.3, 0.5)
    n = r["car_steps"]
    assert n > 50 and all(len(r[k]) == n for k in ("obs", "priv", "u", "logp", "adv", "ret", "val"))
    assert np.isfinite(r["adv"]).all() and r["kinds"][1:].sum() > 0
    np.testing.assert_allclose(r["ret"], r["adv"] + r["val"], atol=1e-5)
    # the second call continues the same episode (cars keep their uids)
    alive_before = w.sim.alive.sum()
    r2 = w.rollout(actor.p, a_sizes, np.full(act_dim(cfg), -0.5), critic.p, c_sizes, 0.3, 0.5)
    assert alive_before > 0 and r2["car_steps"] > 0 and w.n_resets == 1
