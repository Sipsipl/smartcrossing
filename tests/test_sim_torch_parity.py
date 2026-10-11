"""Parity between smartcrossing/sim.py and its torch port smartcrossing/sim_torch.py.

Same seed (TorchSim(rng="numpy") replays sim.py's random draws), same actions: every
step positions, done kinds, uids, rewards, queues and observations must agree
(float64, cpu; and cuda if available). torch is not a declared dependency: skipped without it.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from smartcrossing.sim import Sim, SimConfig, act_dim  # noqa: E402
from smartcrossing.sim_torch import TorchSim  # noqa: E402

TOL = 1e-9
STEPS = 300

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


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
@pytest.mark.parametrize("cfg_kw", [dict(n_slots=24, r_time=0.02, demand_range=(0.2, 0.5)),
                                    dict(n_slots=12, demand=0.4, v_reverse=0.0, max_age=12.0, n_scenarios=3)])
def test_torch_matches_numpy(device, cfg_kw):
    cfg_kw = dict(cfg_kw)
    S = cfg_kw.pop("n_scenarios", None)
    cfg = SimConfig(**cfg_kw)
    B, N, A = 6, cfg.n_slots, act_dim(cfg)
    ref = Sim(cfg, B, seed=7, n_scenarios=S)
    ts = TorchSim(cfg, B, seed=7, n_scenarios=S, device=device, dtype=torch.float64, rng="numpy")
    rng = np.random.default_rng(3)
    kinds = np.zeros(6, np.int64)
    worst = 0.0
    for t in range(STEPS):
        o_ref = ref.observe()
        o_t = ts.observe()
        worst = max(worst, _close(f"obs@{t}", o_t, o_ref))
        act = _actions(rng, B, N, A)
        r_ref = ref.step(act)
        r_t = ts.step(torch.as_tensor(act, device=device))
        al = ref.alive
        assert np.array_equal(ts.alive.cpu().numpy(), al), f"alive@{t}"
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
    for name in ("arrived", "spawned", "exits", "wrong_exits", "crashes", "offroad", "timeouts"):
        assert np.array_equal(getattr(ts.stats, name).cpu().numpy(), getattr(ref.stats, name)), name
    # the run must actually contain traffic events
    assert kinds[1] > 0 and kinds[3] > 0 and kinds[4] > 0, kinds
    print(f"[{device}] max abs deviation over {STEPS} steps: {worst:.3g}; done kinds {kinds.tolist()}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_cuda_graph_matches_eager():
    """The CUDA-graph replay of step() gives the same trajectory as eager stepping (torch rng)."""
    cfg = SimConfig(n_slots=24, demand=0.4)
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
    for _ in range(150):
        replay()
        eager.step(act)
    torch.cuda.synchronize()
    assert torch.equal(eager.alive, graphed.alive)
    assert torch.equal(eager.uid, graphed.uid)
    assert torch.allclose(eager.x, graphed.x, atol=1e-9) and torch.allclose(eager.y, graphed.y, atol=1e-9)
    assert int(graphed.stats.spawned.sum()) > 0 and not bool(graphed.fifo_overflow)
