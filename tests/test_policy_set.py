"""Set encoder over the neighbours (policy_set.py, Roadmap B): numpy / torch forward parity,
permutation invariance over neighbour slots, presence mask, checkpoint round trip (old MLP
checkpoints unchanged), more neighbour slots (SimConfig.n_neighbours), the torch update and
rollout with set networks, and ppo.main --arch set."""

import json
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import policy_set
from smartcrossing.policy import (PolicyDriver, default_in_scale, forward, init_params, layer_sizes, load_arch,
                                  save_npz)
from smartcrossing.ppo_common import N_PRIV
from smartcrossing.sim import OBS_DIM, Sim, SimConfig, act_dim, obs_dim

ROOT = Path(__file__).resolve().parents[1]
POOLS = policy_set.POOLS


def _traffic_obs(cfg, steps=80, B=2, seed=1):
    """Real observations of a busy crossing (alive rows only), scaled like the policy sees them."""
    sim = Sim(cfg, B, seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(steps):
        a = rng.uniform(-1, 1, (B, cfg.n_slots, act_dim(cfg)))
        a[..., 0] = 0.4
        a[..., 1] *= 0.1
        sim.step(a)
        if t % 10 == 9:
            rows.append(sim.observe()[sim.alive])
    return np.concatenate(rows) * default_in_scale(cfg)


def _arch(cfg, pool, **kw):
    return policy_set.make_arch(cfg, hidden=(16, 12), phi=(10, 8), pool=pool, heads=2, dk=4, dv=3, **kw)


def _nb_slice(arch):
    E, K, F = arch["n_ego"], arch["K"], arch["F"]
    return slice(E, E + K * F)


@pytest.fixture(scope="module")
def obs8():
    cfg = SimConfig(n_slots=24, demand=0.6, n_neighbours=8)
    x = _traffic_obs(cfg)
    n_present = x[:, OBS_DIM + 5 :: 10].sum(1)
    assert (n_present == 0).any() and (n_present >= 3).any() and len(x) > 50
    return cfg, x


@pytest.mark.parametrize("pool", POOLS)
def test_permutation_invariance_and_mask(obs8, pool):
    cfg, x = obs8
    arch = _arch(cfg, pool)
    p = policy_set.init_params(arch, 0, out_scale=1.0)
    y = policy_set.forward(p, arch, x)
    assert y.shape == (len(x), act_dim(cfg)) and np.isfinite(y).all()
    E, K, F = arch["n_ego"], arch["K"], arch["F"]
    nb = x[:, _nb_slice(arch)].reshape(len(x), K, F)
    rng = np.random.default_rng(0)
    for _ in range(3):  # shuffle the slot order per row
        perm = np.argsort(rng.random((len(x), K)), 1)
        xs = x.copy()
        xs[:, _nb_slice(arch)] = np.take_along_axis(nb, perm[..., None], 1).reshape(len(x), -1)
        np.testing.assert_allclose(policy_set.forward(p, arch, xs), y, rtol=0, atol=1e-12)
    # absent slots are ignored, whatever their features say
    junk = x.copy()
    blk = junk[:, _nb_slice(arch)].reshape(len(x), K, F)
    absent = blk[..., arch["present"]] == 0
    blk[absent] = rng.normal(size=(absent.sum(), F))
    blk[..., arch["present"]] *= ~absent
    junk[:, _nb_slice(arch)] = blk.reshape(len(x), -1)
    np.testing.assert_allclose(policy_set.forward(p, arch, junk), y, rtol=0, atol=1e-12)
    # the neighbours matter (not a constant pooling)
    alone = x.copy()
    alone[:, _nb_slice(arch)] = 0.0
    has = nb[..., arch["present"]].any(1)
    assert np.abs(policy_set.forward(p, arch, alone)[has] - y[has]).max() > 1e-3


def test_more_slots_same_weights(obs8):
    """Weights do not depend on K: K=4 and K=8 nets share params; with only the nearest 4 present
    in the K=8 obs, both give the same output."""
    cfg8, x8 = obs8
    cfg4 = SimConfig(n_slots=24, n_neighbours=4)
    a8, a4 = _arch(cfg8, "attn"), _arch(cfg4, "attn")
    assert policy_set.n_params(a8) == policy_set.n_params(a4)
    p = policy_set.init_params(a8, 2, out_scale=1.0)
    F = a8["F"]
    x4 = x8[:, : OBS_DIM + 4 * F]
    x8c = x8.copy()
    x8c[:, OBS_DIM + 4 * F :] = 0.0
    np.testing.assert_allclose(policy_set.forward(p, a8, x8c), policy_set.forward(p, a4, x4), rtol=0, atol=1e-12)


def test_torch_forward_parity(obs8):
    torch = pytest.importorskip("torch")
    from smartcrossing.ppo_torch import set_forward

    cfg, x = obs8
    for pool in POOLS:
        for n_extra in (0, N_PRIV):
            arch = _arch(cfg, pool, n_extra=n_extra, n_out=1 if n_extra else None)
            xi = np.concatenate([x, np.random.default_rng(1).normal(size=(len(x), n_extra))], 1)
            p = policy_set.init_params(arch, 3, out_scale=1.0)
            ref = policy_set.forward(p, arch, xi)
            got = set_forward(torch.as_tensor(p), arch, torch.as_tensor(xi)).numpy()
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12, err_msg=f"{pool} extra={n_extra}")
            # batched leading dims (B, N, D) as in the rollout
            got3 = set_forward(torch.as_tensor(p), arch, torch.as_tensor(xi[:12].reshape(3, 4, -1))).numpy()
            np.testing.assert_allclose(got3.reshape(12, -1), ref[:12], rtol=0, atol=1e-12)


def test_torch_autograd_matches_finite_differences(obs8):
    torch = pytest.importorskip("torch")
    from smartcrossing.ppo_torch import net_backward, net_forward

    cfg, x = obs8
    arch = _arch(cfg, "attn")
    p = torch.as_tensor(policy_set.init_params(arch, 4, out_scale=1.0))
    xs = torch.as_tensor(x[:40])
    gout = torch.as_tensor(np.random.default_rng(5).normal(size=(40, arch["n_out"])))
    out, ctx = net_forward(p, None, xs, arch)
    g = net_backward(p, None, ctx, gout, arch).numpy()
    rng = np.random.default_rng(6)
    for j in rng.choice(len(p), 25, replace=False):
        e = np.zeros(len(p))
        e[j] = 1e-6
        fp = (policy_set.forward(p.numpy() + e, arch, x[:40]) * gout.numpy()).sum()
        fm = (policy_set.forward(p.numpy() - e, arch, x[:40]) * gout.numpy()).sum()
        assert g[j] == pytest.approx((fp - fm) / 2e-6, rel=1e-5, abs=1e-8), j


def test_checkpoint_roundtrip_and_mixed_groups(tmp_path):
    cfg = SimConfig(n_slots=16, demand=0.6, n_neighbours=6)
    arch = _arch(cfg, "attn")
    sizes = layer_sizes(cfg, (16, 12))
    p_set = policy_set.init_params(arch, 1, out_scale=1.0)
    p_mlp = init_params(sizes, 2, out_scale=1.0)
    scale = default_in_scale(cfg)
    save_npz(tmp_path / "set.npz", p_set, sizes, cfg, in_scale=scale, arch=arch)
    save_npz(tmp_path / "mlp.npz", p_mlp, sizes, cfg, in_scale=scale)
    assert load_arch(tmp_path / "set.npz") == arch and load_arch(tmp_path / "mlp.npz") is None
    with np.load(tmp_path / "mlp.npz") as z:  # MLP checkpoints keep the old file layout
        assert "arch" not in z.files
    sim = Sim(SimConfig(n_slots=16, demand=0.6), 2, seed=3, group_probs=[0.5, 0.5])
    drv = PolicyDriver.load(f"{tmp_path / 'set.npz'}+{tmp_path / 'mlp.npz'}", sim)
    assert sim.cfg.n_neighbours == 6  # the checkpoints' interface
    for _ in range(60):  # fill the crossing (fixed actions: the random brains would stop or crash)
        sim.step(np.tile([0.4, 0.0, 0.0], (2, 16, 1)))
    obs = sim.observe()
    a = drv.act()
    g0 = sim.alive & (sim.group == 0)
    g1 = sim.alive & (sim.group == 1)
    assert g0.any() and g1.any()
    np.testing.assert_allclose(a[g0], np.tanh(policy_set.forward(p_set, arch, obs[g0] * scale)), atol=1e-12)
    np.testing.assert_allclose(a[g1], forward(p_mlp, obs[g1], sizes, scale), atol=1e-12)
    # mismatching interface (an MLP of K=4 with a K=6 set brain) is refused
    save_npz(tmp_path / "mlp4.npz", init_params(layer_sizes(SimConfig(), (8,)), 0), layer_sizes(SimConfig(), (8,)),
             SimConfig(), in_scale=default_in_scale(SimConfig()))
    with pytest.raises(ValueError, match="interfaces"):
        PolicyDriver.load(f"{tmp_path / 'set.npz'}+{tmp_path / 'mlp4.npz'}", Sim(SimConfig(n_slots=8), 1))


def test_old_checkpoints_unchanged():
    """Committed MLP checkpoints load as before (no arch, K = 4) and their export has layers."""
    from smartcrossing.policy import export_json, load_npz

    mlp_paths = [p for p in sorted((ROOT / "results").glob("*.npz")) if load_arch(p) is None]
    assert len(mlp_paths) >= 3
    for path in mlp_paths[:3]:
        params, sizes, _ = load_npz(path)
        sim = Sim(SimConfig(n_slots=8), 1, seed=0)
        drv = PolicyDriver.load(path, sim)
        assert drv.archs == [None] and sim.cfg.n_neighbours == 4
        assert "layers" in export_json(path) and "arch" not in export_json(path)


def test_torch_rollout_and_update_with_set_networks():
    """TorchRollout with set actor + critic matches ppo.Worker (float64, numpy streams); the torch
    update moves the set params and keeps the numpy objects as the source of truth."""
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from smartcrossing.ppo import Adam, SetNet, Worker
    from smartcrossing.ppo_torch import TorchPPO
    from smartcrossing.rollout_torch import TorchRollout

    cfg = SimConfig(n_slots=8, r_time=0.02, n_neighbours=6)
    in_scale = default_in_scale(cfg)
    A = act_dim(cfg)
    a_arch = _arch(cfg, "attn")
    c_arch = _arch(cfg, "max", n_extra=N_PRIV, n_out=1)
    a_sizes = layer_sizes(cfg, (16, 12))
    c_sizes = [obs_dim(cfg) + N_PRIV, 16, 12, 1]
    actor, critic = SetNet(a_arch, a_sizes, 0, 0.01), SetNet(c_arch, c_sizes, 1)
    actor.p[-A] = np.arctanh(0.3)
    logstd = np.full(A, -0.5)
    kw = dict(cfg_d=dict(cfg.__dict__), n_envs=3, seed=0, horizon=50, episode_steps=1000, gamma=0.99, lam=0.95,
              in_scale=in_scale, act_mask=np.ones(A), actor_arch=a_arch, critic_arch=c_arch)
    w = Worker(**kw)
    ro = TorchRollout(**kw, actor_sizes=a_sizes, critic_sizes=c_sizes, device="cpu", dtype=torch.float64,
                      rng="numpy", noise="numpy")
    ref = w.rollout(actor.p, a_sizes, logstd, critic.p, c_sizes, 0.4, 0.6)
    ro.set_params(actor.p, logstd, critic.p)
    got = ro.rollout(0.4, 0.6)
    assert got["car_steps"] == ref["car_steps"] > 0
    for key in ("obs", "u", "logp", "adv", "ret", "val"):
        np.testing.assert_allclose(got[key].numpy(), ref[key], rtol=1e-6, atol=1e-6, err_msg=key)
    a = SimpleNamespace(epochs=2, minibatches=4, clip=0.2, ent=0.0, target_kl=0.0, max_grad=1.0, lr_std=3e-3,
                        logstd_min=-2.5)
    opt_a, opt_c = Adam(len(actor.p), 3e-4), Adam(len(critic.p), 1e-3)
    p0, c0 = actor.p.copy(), critic.p.copy()
    out = TorchPPO("cpu", "float64").update(actor, logstd, critic, opt_a, opt_c, got, a, np.ones(A), in_scale,
                                            np.random.default_rng(0))
    assert out["epochs_done"] == 2 and np.isfinite(out["vloss"])
    assert np.abs(actor.p - p0).max() > 1e-5 and np.abs(critic.p - c0).max() > 1e-5
    assert opt_a.t == opt_c.t == 8


def test_ppo_main_arch_set(tmp_path):
    pytest.importorskip("torch")
    from smartcrossing import ppo

    runs = tmp_path / "runs"
    common = ["--workers", "1", "--envs", "2", "--horizon", "16", "--hidden", "8,8", "--minibatches", "2", "--epochs",
              "1", "--eval-every", "0", "--demand-start", "0.3", "--arch", "set", "--set-phi", "8", "--set-heads", "2",
              "--set-dk", "4", "--cfg", json.dumps({"n_slots": 8, "n_neighbours": 6})]
    with pytest.raises(SystemExit, match="torch update"):
        ppo.main(["--name", "bad", "--iters", "1", "--runs-dir", str(runs), *common])
    ppo.main(["--name", "s", "--iters", "3", "--runs-dir", str(runs), "--update-device", "cpu", *common])
    ppo.main(["--name", "t", "--iters", "2", "--runs-dir", str(runs), "--rollout-device", "cpu", *common])
    # behaviour-cloning warm start from the MLP brain (K 4 -> 6)
    ppo.main(["--name", "b", "--iters", "1", "--runs-dir", str(runs), "--rollout-device", "cpu", "--init-bc",
              str(ROOT / "results" / "round2_best.npz"), "--bc-steps", "5", "--bc-seconds", "3", *common])
    for name in ("s", "t", "b"):
        run = runs / name
        arch = load_arch(run / "current.npz")
        assert arch["type"] == "set" and arch["K"] == 6 and arch["phi"] == [8]
        args = json.loads((run / "args.json").read_text(encoding="utf-8"))
        assert args["actor_arch"] == arch and args["critic_arch"]["n_extra"] == N_PRIV
        sim = Sim(SimConfig(n_slots=8, demand=0.5), 2, seed=0)
        drv = PolicyDriver.load(run / "current.npz", sim)
        for _ in range(20):
            sim.step(drv.act())
    from smartcrossing.gates import stop_gate

    g, _ = stop_gate(f"policy:{runs / 's' / 'current.npz'}", {})
    assert 0.0 <= g["safe_rate_feasible"] <= 1.0


def test_behaviour_cloning_into_set_actor():
    """bc.py: the teacher's states and pre-tanh outputs are recorded (student K > teacher K works),
    and fitting reduces the imitation error."""
    pytest.importorskip("torch")
    from smartcrossing.bc import collect, fit, teacher_mu
    from smartcrossing.policy import load_in_scale, load_npz

    teacher = ROOT / "results" / "round2_best.npz"
    cfg = SimConfig(n_slots=16, n_neighbours=6)
    x, y = collect(str(teacher), cfg, envs=4, seconds=15.0, demand=(0.3, 0.4))
    assert x.shape[1] == obs_dim(cfg) and len(x) == len(y) > 100
    params, sizes, _ = load_npz(teacher)
    n_t = sizes[0]
    np.testing.assert_allclose(teacher_mu(params, sizes, x[:, :n_t] * load_in_scale(teacher)), y, atol=1e-12)
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
