"""Parity between smartcrossing/sim.py and its torch port smartcrossing/sim_torch.py.

Same seed (TorchSim(rng="numpy") replays sim.py's random draws), same actions: every
step positions, done kinds, uids, rewards, queues and observations must agree
(float64, cpu; and cuda with SC_TEST_CUDA=1, with a shorter run: eager cuda is launch-bound).
torch comes from the optional `gpu` poetry group: skipped without it.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import CUDA, DEVICES  # noqa: E402
from smartcrossing.sim import STAT_NAMES, Sim, SimConfig, act_dim  # noqa: E402
from smartcrossing.sim_torch import TorchSim  # noqa: E402

TOL = 1e-9
RUN = {"cpu": (6, 300), "cuda": (6, 120)}  # (envs, steps); both contain exits, crashes, off-road and wrecks



def _actions(rng, B, N, A):
    a = rng.normal(0.0, 0.5, (B, N, A))
    a[..., 0] += 0.3  # mostly accelerating, mostly straight: exits as well as crashes / off-road
    a[..., 1] *= 0.15
    return a


def _close(name, t, ref, mask=None, tol=TOL):
    got = t.detach().cpu().numpy()
    if mask is not None:
        got, ref = got[mask], ref[mask]
    err = float(np.max(np.abs(got - ref))) if got.size else 0.0
    assert err <= tol, f"{name}: max abs deviation {err:g}"
    return err


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cfg_kw", [
    # current defaults: wrecks, brake light + hazard, one lamp, right-half spawning
    dict(n_slots=24, r_time=0.02, demand_range=(0.2, 0.5)),
    # short wrecks (many appear and vanish), signal noise off
    dict(n_slots=16, demand=0.5, wreck_time=1.0, n_scenarios=2),
    # the legacy interface / world of the old checkpoints
    dict(n_slots=12, demand=0.4, v_reverse=0.0, max_age=12.0, n_scenarios=3, n_signal=2, brake_light=False,
         spawn_right=False, wreck_time=0.0),
    # round 2: crash cost by own impact speed + obstacle scenarios (wrecks and parked cars) in half the envs
    dict(n_slots=24, r_time=0.02, demand_range=(0.2, 0.5), crash_cost_speed="own", obstacle_envs=0.5,
         obstacle_rate=2.0, obstacle_time=3.0, obstacle_max=3,
         wreck_time=5.0),  # (long wrecks block the traffic so much that no obstacle gets hit in the short run)
    # more neighbour slots (set encoder interface); and more slots than cars (padding)
    dict(n_slots=24, demand=0.5, n_neighbours=10, L=30.0),  # short arms: exits within the short cuda run
    dict(n_slots=6, demand=0.5, n_neighbours=10, L=30.0),
])
def test_torch_matches_numpy(device, cfg_kw):
    cfg_kw = dict(cfg_kw)
    S = cfg_kw.pop("n_scenarios", None)
    cfg = SimConfig(**cfg_kw)
    (B, steps), N, A = RUN[device], cfg.n_slots, act_dim(cfg)
    ref = Sim(cfg, B, seed=7, n_scenarios=S)
    ts = TorchSim(cfg, B, seed=7, n_scenarios=S, device=device, dtype=torch.float64, rng="numpy")
    rng = np.random.default_rng(3)
    kinds = np.zeros(6, np.int64)
    worst = 0.0
    n_wreck_seen = n_parked_seen = n_obst_wreck_seen = 0
    for t in range(steps):
        o_ref = ref.observe()
        o_t = ts.observe()
        worst = max(worst, _close(f"obs@{t}", o_t, o_ref))
        act = _actions(rng, B, N, A)
        r_ref = ref.step(act)
        r_t = ts.step(torch.as_tensor(act, device=device))
        al = ref.alive | ref.wreck
        assert np.array_equal(ts.alive.cpu().numpy(), ref.alive), f"alive@{t}"
        assert np.array_equal(ts.wreck.cpu().numpy(), ref.wreck), f"wreck@{t}"
        assert np.array_equal(ts.wreck_left.cpu().numpy(), ref.wreck_left), f"wreck_left@{t}"
        assert np.array_equal(ts.obstacle.cpu().numpy(), ref.obstacle), f"obstacle@{t}"
        assert np.array_equal(ts.parked.cpu().numpy(), ref.parked), f"parked@{t}"
        n_wreck_seen += int(ref.wreck.sum())
        n_parked_seen += int(ref.parked.sum())
        n_obst_wreck_seen += int((ref.obstacle & ~ref.parked).sum())
        assert np.array_equal(ts.done_kind.cpu().numpy(), ref.done_kind), f"done_kind@{t}"
        assert np.array_equal(ts.done_uid.cpu().numpy(), ref.done_uid), f"done_uid@{t}"
        assert np.array_equal(ts.uid.cpu().numpy(), ref.uid), f"uid@{t}"
        assert np.array_equal(ts.queue.cpu().numpy(), ref.queue), f"queue@{t}"
        for name in ("x", "y", "th", "v", "kappa", "acc", "age"):
            worst = max(worst, _close(f"{name}@{t}", getattr(ts, name), getattr(ref, name), al))
        for name in ("arm", "man"):
            assert np.array_equal(getattr(ts, name).cpu().numpy()[al], getattr(ref, name)[al]), f"{name}@{t}"
        worst = max(worst, _close(f"car_reward@{t}", ts.car_reward, ref.car_reward))
        worst = max(worst, _close(f"reward@{t}", r_t, r_ref))
        worst = max(worst, _close(f"prev_dist@{t}", ts._prev_dist, ref._prev_dist, al))
        kinds += np.bincount(ref.done_kind.ravel(), minlength=6)
    for name in STAT_NAMES:
        assert np.array_equal(getattr(ts.stats, name).cpu().numpy(), getattr(ref.stats, name)), name
    # the run must actually contain traffic events
    assert kinds[1] > 0 and kinds[3] > 0 and kinds[4] > 0, kinds
    assert (n_wreck_seen > 0) == (cfg.wreck_time > 0)
    if cfg.obstacle_envs > 0:  # obstacles of both kinds appeared and were hit
        assert ref.stats.obstacles.sum() >= 4 and n_parked_seen > 0 and n_obst_wreck_seen > 0
        assert ref.stats.obstacle_hits.sum() > 0 or steps < 300  # (the short cuda run may have none)
    print(f"[{device}] max abs deviation over {steps} steps: {worst:.3g}; done kinds {kinds.tolist()}")


@CUDA
@pytest.mark.parametrize("extra", [{}, dict(obstacle_envs=0.5, obstacle_rate=2.0, crash_cost_speed="own")])
def test_cuda_graph_matches_eager(extra):
    """The CUDA-graph replay of step() gives the same trajectory as eager stepping (torch rng)."""
    cfg = SimConfig(n_slots=24, demand=0.4, **extra)
    B, N, A = 16, cfg.n_slots, act_dim(cfg)
    sims = [TorchSim(cfg, B, seed=5, device="cuda", rng="torch") for _ in range(2)]
    for s in sims:
        s.fifo_sync = False  # same fixed-capacity FIFO in both
    act = torch.zeros(B, N, A, device="cuda", dtype=torch.float64)
    act[..., 0] = 0.3
    eager, graphed = sims
    replay = graphed.cuda_graph(lambda: graphed.step(act), warmup=3)
    for _ in range(3):  # the warm-up ran 3 steps (capture only records)
        eager.step(act)
    for _ in range(80):
        replay()
        eager.step(act)
    torch.cuda.synchronize()
    assert torch.equal(eager.alive, graphed.alive)
    assert torch.equal(eager.uid, graphed.uid)
    assert torch.allclose(eager.x, graphed.x, atol=1e-9) and torch.allclose(eager.y, graphed.y, atol=1e-9)
    assert int(graphed.stats.spawned.sum()) > 0 and not bool(graphed.fifo_overflow)
    assert torch.equal(eager.obstacle, graphed.obstacle) and torch.equal(eager.stats.obstacles, graphed.stats.obstacles)
    assert (int(graphed.stats.obstacles.sum()) > 0) == bool(extra)


def test_group_probs_and_fifo_growth_match_numpy():
    """Constructor args group_probs / fifo_cap: the groups follow sim.py's draws, and a tiny FIFO
    that must grow keeps the queue order (same spawns as sim.py)."""
    cfg = SimConfig(n_slots=4, demand=1.5)
    B, N, A = 3, cfg.n_slots, act_dim(cfg)
    ref = Sim(cfg, B, seed=4, group_probs=[0.6, 0.4])
    ts = TorchSim(cfg, B, seed=4, rng="numpy", group_probs=[0.6, 0.4], fifo_cap=2)
    act = np.tile([0.5, 0.0, 0.0], (B, N, 1))
    for _ in range(150):
        ref.step(act)
        ts.step(torch.as_tensor(act))
    assert ts._qcap >= 16 and ref.queue.max() > 8 and len(set(ref.group[ref.alive])) == 2
    for name in ("queue", "alive", "uid", "group", "man"):
        assert np.array_equal(getattr(ts, name).numpy(), getattr(ref, name)), name


@pytest.mark.parametrize("demand_range", [None, (0.2, 0.5)])
def test_reset_inplace_is_reproducible(demand_range):
    """reset_inplace(seed) continues exactly like a fresh TorchSim(seed) (incl. the generator)."""
    cfg = SimConfig(n_slots=12, demand=0.4, demand_range=demand_range)
    B, N, A = 4, cfg.n_slots, act_dim(cfg)
    act = torch.zeros(B, N, A, dtype=torch.float64)
    act[..., 0] = 0.3
    reused = TorchSim(cfg, B, seed=1)
    for _ in range(40):
        reused.step(act)
    reused.reset_inplace(5)
    fresh = TorchSim(cfg, B, seed=5)
    for _ in range(60):
        reused.step(act)
        fresh.step(act)
    assert int(fresh.stats.spawned.sum()) > 0
    for name in ("alive", "uid", "x", "y", "queue", "man"):
        assert torch.equal(getattr(reused, name), getattr(fresh, name)), name
