"""GRU memory of the set encoder (policy_set ``mem``): the upgraded round-7 brain drives exactly like the stateless
one, numpy / torch parity of the recurrent forward over sequences with resets, the per-car state of PolicyDriver
(reset when a slot gets a new car), and the recurrent PPO path: TorchRollout's stored log-probs equal the update's
recomputation (sequence minibatches, truncated BPTT) at unchanged parameters. cuda cases only with SC_TEST_CUDA=1."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from smartcrossing import policy_set
from smartcrossing.metrics import summarize
from smartcrossing.policy import PolicyDriver, default_in_scale, layer_sizes, load_arch, load_in_scale, load_npz, save_npz
from smartcrossing.ppo_common import LOG2PI, N_PRIV
from smartcrossing.sim import Sim, SimConfig, act_dim, obs_dim

ROOT = Path(__file__).resolve().parents[1]
ROUND7 = ROOT / "results" / "round7_best.npz"


def small_mem(cfg, seed=0, Hm=6, pool="attn"):
    """A small memory arch with params whose memory matters (mem.out.W and the GRU biases non-zero)."""
    arch = policy_set.make_arch(cfg, hidden=(16, 12), phi=(10, 8), pool=pool, heads=2, dk=4, dv=3, mem=Hm)
    p = policy_set.init_params(arch, seed, out_scale=1.0)
    P = policy_set.unpack(p, arch)
    rng = np.random.default_rng(seed + 100)
    for name in (policy_set.MEM_OUT, "gru.bh", "gru.bx"):
        P[name][...] += rng.normal(0, 0.7, P[name].shape)
    return arch, p


def round7_upgraded(Hm=16, seed=0):
    p, sizes, cfg = load_npz(ROUND7)
    arch = load_arch(ROUND7)
    arch_m = policy_set.with_mem(arch, Hm)
    return p, arch, policy_set.upgrade_params(p, arch, arch_m, seed), arch_m, sizes, cfg


def test_upgrade_is_prefix_and_has_no_effect():
    p, arch, pm, arch_m, _, _ = round7_upgraded()
    assert arch_m["mem"] == {"type": "gru", "size": 16} and "mem" not in arch
    assert len(pm) == policy_set.n_params(arch_m) > len(p) and np.array_equal(pm[: len(p)], p)
    P = policy_set.unpack(pm, arch_m)
    assert not P[policy_set.MEM_OUT].any() and np.all(P["gru.bx"][16:32] == policy_set.UPDATE_GATE_BIAS)
    assert policy_set.out_bias_index(arch_m) == policy_set.out_bias_index(arch) == len(p) - arch["n_out"]
    x = np.random.default_rng(1).normal(size=(50, policy_set.in_dim(arch)))
    x[:, arch["n_ego"] + arch["present"] :: arch["F"]] = np.random.default_rng(2).random((50, arch["K"])) > 0.4
    ref = policy_set.forward(p, arch, x)
    for h in (np.zeros((50, 16)), np.random.default_rng(3).uniform(-1, 1, (50, 16))):
        out, h_new = policy_set.forward_mem(pm, arch_m, x, h)
        np.testing.assert_array_equal(out, ref)  # bit-identical: + 0 * memory
        assert np.abs(h_new).max() > 0.01
    with pytest.raises(ValueError, match="memory"):
        policy_set.forward(pm, arch_m, x)
    with pytest.raises(ValueError, match="cannot upgrade"):
        policy_set.upgrade_params(pm, arch_m, arch_m)


@pytest.mark.parametrize("noise", [None, 0.3])
def test_upgraded_round7_drives_identically(tmp_path, noise):
    """Multi-step rollout (dense traffic, deterministic and with action noise): the upgraded brain gives the same
    actions at every step as round7_best, and the same bench summary."""
    _, _, pm, arch_m, sizes, cfg = round7_upgraded()
    save_npz(tmp_path / "mem.npz", pm, sizes, cfg, load_in_scale(ROUND7), arch=arch_m)
    sims = [Sim(SimConfig(n_slots=40, demand=0.4), 3, seed=5) for _ in range(2)]
    drvs = [PolicyDriver.load(ROUND7, sims[0]), PolicyDriver.load(tmp_path / "mem.npz", sims[1])]
    if noise:
        for d in drvs:
            d.set_noise(np.full(3, noise), np.random.default_rng(7))
    for _ in range(150):
        acts = [d.act() for d in drvs]
        np.testing.assert_array_equal(acts[0], acts[1])
        for s, a in zip(sims, acts):
            s.step(a)
    assert drvs[1].h[0] is not None and np.abs(drvs[1].h[0]).max() > 0.01 and drvs[0].h[0] is None
    assert summarize(sims[0], 15.0) == summarize(sims[1], 15.0)


def _seq_inputs(arch, T=12, R=30, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, R, policy_set.in_dim(arch)))
    x[..., arch["n_ego"] + arch["present"] :: arch["F"]] = rng.random((T, R, arch["K"])) > 0.4
    first = rng.random((T, R)) < 0.15
    valid = rng.random((T, R)) < 0.85
    first &= valid
    h0 = rng.uniform(-1, 1, (R, policy_set.mem_size(arch)))
    return x, first, valid, h0


@pytest.mark.parametrize("pool", policy_set.POOLS)
def test_numpy_torch_recurrent_parity(pool):
    """Step-by-step numpy forward_mem (the driver's pass) == torch set_forward_mem == the update's sequence pass
    (seq_forward_mem: encoder on the valid rows, GRU over T with resets from h0), float64."""
    torch = pytest.importorskip("torch")
    from smartcrossing.ppo_torch import seq_forward_mem, set_forward_mem

    cfg = SimConfig(n_neighbours=5)
    arch, p = small_mem(cfg, 1, pool=pool)
    x, first, valid, h0 = _seq_inputs(arch)
    T, R = first.shape
    h, outs, h_t = h0.copy(), np.zeros((T, R, arch["n_out"])), torch.as_tensor(h0)
    pt = torch.as_tensor(p)
    for t in range(T):
        h = np.where(first[t][:, None], 0.0, h)
        out, hn = policy_set.forward_mem(p, arch, x[t], h)
        out_t, hn_t = set_forward_mem(pt, arch, torch.as_tensor(x[t]), torch.where(torch.as_tensor(first[t])[:, None], 0.0, h_t))
        np.testing.assert_allclose(out_t.numpy(), out, rtol=0, atol=1e-12)
        np.testing.assert_allclose(hn_t.numpy(), hn, rtol=0, atol=1e-12)
        outs[t] = out
        h = np.where(valid[t][:, None], hn, 0.0)
        h_t = torch.where(torch.as_tensor(valid[t])[:, None], hn_t, 0.0)
    # the sequence pass over a subset of the columns (as one minibatch of the update)
    cols = np.array([3, 0, 7, 12, 29, 5])
    tt, cc = np.nonzero(valid[:, cols])
    seq = {"first": torch.as_tensor(first), "valid": torch.as_tensor(valid), "h0": torch.as_tensor(h0)}
    got = seq_forward_mem(pt, arch, torch.as_tensor(x[tt, cols[cc]]), seq, torch.as_tensor(cols), torch.as_tensor(tt),
                          torch.as_tensor(cc))
    np.testing.assert_allclose(got.numpy(), outs[tt, cols[cc]], rtol=0, atol=1e-12)


def test_driver_memory_resets_on_new_car():
    """PolicyDriver keeps one state per car: a slot that gets a new car (new uid; here also in the same step as the
    old car left) starts from 0, the others continue; non-driving slots hold 0; reset_state forgets everything."""
    cfg = SimConfig(n_slots=8, demand=0.6, wreck_time=0.0)
    arch, p = small_mem(cfg, 2)
    sim = Sim(cfg, 4, seed=3)
    scale = default_in_scale(cfg)
    drv = PolicyDriver(sim, p, layer_sizes(cfg, (16, 12)), in_scale=scale, archs=arch)
    reuse = 0
    uid0, al0 = np.full(sim.uid.shape, -1), np.zeros(sim.alive.shape, bool)  # at the previous act()
    for t in range(300):
        H_before = np.zeros((*sim.alive.shape, 6)) if drv.h[0] is None else drv.h[0].copy()
        obs = sim.observe()
        a = drv.act()
        new = sim.alive & ((sim.uid != uid0) | ~al0)
        h_in = np.where(new[..., None], 0.0, H_before)
        mu, h_new = policy_set.forward_mem(p, arch, obs * scale, h_in)
        np.testing.assert_allclose(a[sim.alive], np.tanh(mu)[sim.alive], rtol=0, atol=1e-12)
        np.testing.assert_allclose(drv.h[0][sim.alive], h_new[sim.alive], rtol=0, atol=1e-12)
        assert not drv.h[0][~sim.alive].any()
        reuse += int((sim.alive & al0 & (sim.uid != uid0)).sum())  # removed and refilled in one step
        uid0, al0 = sim.uid.copy(), sim.alive.copy()
        sim.step(a)
    assert reuse > 0
    # a fake new car in a busy slot: its action is the zero-state one, and differs from the continued one
    b, n = np.argwhere(sim.alive)[0]
    obs = sim.observe()
    cont = drv._policy(p, obs[b, n][None], arch, drv.h[0][b, n][None])[0][0]
    sim.uid[b, n] += 10**6
    a = drv.act()
    fresh = np.tanh(policy_set.forward_mem(p, arch, obs[b, n][None] * scale, np.zeros((1, 6)))[0][0])
    np.testing.assert_allclose(a[b, n], fresh, atol=1e-12)
    assert np.abs(fresh - cont).max() > 1e-4
    drv.reset_state()
    assert drv.h[0] is None and drv.uid_prev is None


# ---------------------------------------------------------------- recurrent PPO (torch rollout + update)
def _mem_rollout(device="cpu", B=6, T=40, dtype="float64", graph=None, seed=0):
    torch = pytest.importorskip("torch")
    from smartcrossing.rollout_torch import TorchRollout

    cfg = SimConfig(n_slots=6, wreck_time=0.0, r_time=0.02)
    arch, p = small_mem(cfg, 4)
    A = act_dim(cfg)
    c_sizes = [obs_dim(cfg) + N_PRIV, 16, 1]
    pc = np.random.default_rng(1).normal(0, 0.2, (obs_dim(cfg) + N_PRIV) * 16 + 16 + 16 + 1)
    logstd = np.full(A, -0.7)
    ro = TorchRollout(dict(cfg.__dict__), B, seed, T, 1000, 0.99, 0.95, default_in_scale(cfg), np.ones(A),
                      layer_sizes(cfg, (16, 12)), c_sizes, device=device, dtype=getattr(torch, dtype), graph=graph,
                      actor_arch=arch)
    ro.set_params(p, logstd, pc)
    return ro, cfg, arch, p, pc, c_sizes, logstd


def _logp(arch, p, data, in_scale, logstd, cols=None):
    """Log-probs of the batch rows recomputed by the update's sequence pass (all columns)."""
    import torch

    from smartcrossing.ppo_torch import seq_forward_mem

    seq = data["seq"]
    C = seq["h0"].shape[0]
    x = data["obs"].double() * torch.as_tensor(in_scale)
    mu = seq_forward_mem(torch.as_tensor(p), arch, x, seq, torch.arange(C), seq["row_t"], seq["row_c"])
    ls = torch.as_tensor(logstd)
    return ((-0.5 * ((data["u"].double() - mu) / torch.exp(ls)) ** 2 - ls - 0.5 * LOG2PI).sum(-1))


def test_torch_rollout_logp_equals_update_recompute():
    """KEY test of the recurrent PPO: on cpu float64 the log-probs stored by TorchRollout equal the update's
    recomputation from the sequence batch (ratio 1), over two rollouts (the second starts from a non-zero h0) with
    slot reuse (a car removed and a new one spawned into its slot in the same step); a stateless recomputation
    differs (the memory matters). An update with zero learning rates then sees ratio 1 (kl 0, no clipping)."""
    import torch

    from smartcrossing.ppo_numpy import MLP, Adam, SetNet
    from smartcrossing.ppo_torch import TorchPPO

    ro, cfg, arch, p, pc, c_sizes, logstd = _mem_rollout()
    in_scale = default_in_scale(cfg)
    reuse = 0
    for k in range(2):
        r = ro.rollout(0.6, 0.8)
        seq = r["seq"]
        n, C = r["car_steps"], seq["h0"].shape[0]
        assert seq["row_t"].shape == seq["row_c"].shape == (n,) and seq["first"].shape == seq["valid"].shape
        assert int(seq["valid"].sum()) == n and C <= ro.B * ro.N
        if k == 1:
            assert float(seq["h0"].abs().max()) > 0.01
        reuse += int((seq["first"][1:] & seq["valid"][:-1]).sum())
        lp = _logp(arch, p, r, in_scale, logstd)
        np.testing.assert_allclose(lp.numpy(), r["logp"].double().numpy(), rtol=0, atol=2e-5)
        no_mem = {**r, "seq": {**seq, "first": torch.ones_like(seq["first"])}}
        assert float((_logp(arch, p, no_mem, in_scale, logstd) - r["logp"]).abs().max()) > 1e-2
    assert reuse > 0
    actor = SetNet(arch, layer_sizes(cfg, (16, 12)))
    actor.p = p.copy()
    critic = MLP(c_sizes)
    critic.p = pc.copy()
    a = SimpleNamespace(epochs=2, minibatches=3, clip=0.2, ent=0.0, target_kl=0.0, max_grad=1.0, lr_std=0.0,
                        logstd_min=-2.5)
    ls = logstd.copy()
    out = TorchPPO("cpu", "float64").update(actor, ls, critic, Adam(len(p), 0.0), Adam(len(pc), 0.0), r, a,
                                            np.ones(act_dim(cfg)), in_scale, np.random.default_rng(0))
    assert out["kl"] < 1e-9 and out["clipfrac"] == 0.0 and out["epochs_done"] == 2
    np.testing.assert_array_equal(actor.p, p)
    # with a learning rate the memory weights get a gradient (mem.out.W starts non-zero here)
    actor.p = p.copy()
    TorchPPO("cpu", "float64").update(actor, logstd.copy(), critic, Adam(len(p), 1e-3), Adam(len(pc), 1e-3), r, a,
                                      np.ones(act_dim(cfg)), in_scale, np.random.default_rng(0))
    P0, P1 = policy_set.unpack(p, arch), policy_set.unpack(actor.p, arch)
    assert all(np.abs(P1[k] - P0[k]).max() > 0 for k in ("gru.Wx", "gru.Wh", policy_set.MEM_OUT))


def test_numpy_paths_refuse_memory():
    from smartcrossing.ppo_numpy import SetNet, Worker, ppo_update

    cfg = SimConfig(n_slots=6)
    arch, _ = small_mem(cfg)
    with pytest.raises(ValueError, match="torch rollout"):
        Worker(dict(cfg.__dict__), 1, 0, 8, 100, 0.99, 0.95, default_in_scale(cfg), np.ones(3), actor_arch=arch)
    with pytest.raises(ValueError, match="torch update"):
        ppo_update(SetNet(arch, layer_sizes(cfg)), None, None, None, None, {"adv": []}, None, None, None, None)
    from smartcrossing.bc import bc_actor

    with pytest.raises(ValueError, match="memory"):
        bc_actor(str(ROUND7), cfg, arch, None, None)


@pytest.mark.cuda
def test_cuda_graph_recurrent_rollout_matches_eager():
    """The memory state lives in static tensors updated in place: graph replay == eager steps (float64)."""
    import torch

    outs = []
    for graph in (True, False):
        ro, *_ = _mem_rollout("cuda", B=16, T=40, graph=graph)
        if not graph:  # the capture ran 3 warm-up steps
            ro._reset(0.6, 0.8)
            ro.advance(3)
        outs.append([ro.rollout(0.6, 0.8), ro.rollout(0.6, 0.8)])
    for g, e in zip(*outs):
        assert g["car_steps"] == e["car_steps"] > 0
        for key in ("obs", "u", "logp", "adv"):
            torch.testing.assert_close(g[key], e[key], rtol=0, atol=1e-6)
        for key in ("first", "valid", "row_t", "row_c"):
            assert torch.equal(g["seq"][key], e["seq"][key]), key
        torch.testing.assert_close(g["seq"]["h0"], e["seq"]["h0"], rtol=0, atol=1e-6)
