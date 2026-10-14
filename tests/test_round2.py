"""Round 2 options: crash cost by own impact speed (variant C), obstacle scenarios (variant B),
the stop gate (gates.py) and the exploration knobs of the warm start (variant A)."""

import json
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import gates, ppo
from smartcrossing.sim import (
    DONE_CRASH,
    NB_FEATURES,
    OBS_DIM,
    STAT_NAMES,
    Sim,
    SimConfig,
    act_dim,
    obstacle_clear,
    obstacle_steps,
)

from test_wrecks import place, zeros  # noqa: E402 (tests/ is on sys.path)

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = ROOT / "results" / "round1_x3.npz"


def rear_end(cfg, v=10.0):
    """A car driving into the back of a standing car (overlap after one step)."""
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 3.5, -20.0, np.pi / 2, v)
    place(sim, 1, 3.5, -20.0 + 4.6 + 0.5, np.pi / 2, 0.0)
    sim.step(zeros(sim))  # (pedal 0: the front car keeps standing)
    return sim


def test_crash_cost_own_speed():
    rel = rear_end(SimConfig(demand=0.0))
    own = rear_end(SimConfig(demand=0.0, crash_cost_speed="own"))
    for sim in (rel, own):
        assert (sim.done_kind[0, :2] == DONE_CRASH).all()
    # relative: both pay base + 0.1 x closing speed; own: the standing car pays only the base cost
    assert rel.car_reward[0, 1] == pytest.approx(-1.5 - 0.1 * 10.0)  # the standing car (no progress)
    assert own.car_reward[0, 1] - rel.car_reward[0, 1] == pytest.approx(0.1 * 10.0)  # no speed term
    assert own.car_reward[0, 0] == pytest.approx(rel.car_reward[0, 0])  # the hitter: same as relative
    assert own.car_reward[0, 0] < -1.5 - 0.5
    with pytest.raises(AssertionError):
        rear_end(SimConfig(demand=0.0, crash_cost_speed="blame"))


def obstacle_cfg(**kw):
    return SimConfig(**{**dict(demand=0.2, obstacle_envs=0.5, obstacle_rate=3.0, obstacle_time=2.0, obstacle_max=2), **kw})


def test_obstacles_appear_only_in_their_envs_and_vanish():
    cfg = obstacle_cfg()
    sim = Sim(cfg, 8, seed=3)
    act = zeros(sim)
    act[..., 0] = 0.3
    life = np.zeros((8, cfg.n_slots), np.int64)
    px = sim.x.copy()
    rewarded = 0.0
    for _ in range(400):
        before = sim.obstacle.copy()
        sim.step(act)
        assert not (sim.obstacle & sim.alive).any() and (sim.wreck[sim.obstacle]).all()
        assert (sim.uid[sim.obstacle] == -1).all() and (sim.v[sim.obstacle] == 0).all()
        assert not sim.obstacle[4:].any()  # obstacle_envs = 0.5 -> envs 0..3
        assert (sim.obstacle.sum(1) <= cfg.obstacle_max).all()
        rewarded += np.abs(sim.car_reward[before & sim.obstacle]).sum()
        same = sim.obstacle & before & (sim.x == px)  # (a slot may get a new obstacle right away)
        life = np.where(same, life + 1, np.where(sim.obstacle, 1, 0))
        px = sim.x.copy()
        assert life.max() <= obstacle_steps(cfg)
    st = sim.stats
    assert st.obstacles[:4].sum() > 10 and st.obstacles[4:].sum() == 0
    assert rewarded == 0.0  # obstacles never get a reward
    assert st.obstacle_hits.sum() <= st.crashes.sum()


def test_obstacle_placement_keeps_clearance():
    """An obstacle never appears where a car could no longer stop in front of it."""
    cfg = obstacle_cfg(demand=0.5, obstacle_rate=10.0)
    sim = Sim(cfg, 6, seed=1)
    act = zeros(sim)
    act[..., 0] = 0.5
    n_new = 0
    for _ in range(300):
        before = sim.obstacle.copy()
        prev_uids = sim.uid.copy()
        sim.step(act)
        for b, n in zip(*np.nonzero(sim.obstacle & ~before)):
            n_new += 1
            # positions after this step's motion are what the placement saw: check against them
            others = sim.alive[b] | (sim.wreck[b] & ~((np.arange(cfg.n_slots) == n)))
            spawned_now = sim.alive[b] & ~np.isin(sim.uid[b], prev_uids[b])  # entered after the placement
            m = others & ~spawned_now
            dx, dy = sim.x[b, n] - sim.x[b][m], sim.y[b, n] - sim.y[b][m]
            vx, vy = sim.v[b][m] * np.cos(sim.th[b][m]), sim.v[b][m] * np.sin(sim.th[b][m])
            assert obstacle_clear(cfg, dx, dy, vx, vy).all()
    assert n_new > 5


def test_parked_has_no_hazard_light():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 3.5, -30.0, np.pi / 2, 5.0)
    for n, (y, parked) in enumerate(((-20.0, True), (-10.0, False)), start=1):
        sim.wreck[0, n] = sim.obstacle[0, n] = True
        sim.parked[0, n] = parked
        sim.wreck_left[0, n] = 100
        sim.x[0, n], sim.y[0, n], sim.th[0, n] = 3.5, y, np.pi / 2
    obs = sim.observe()[0, 0]
    blk = NB_FEATURES + cfg.n_signal
    nb = [obs[OBS_DIM + k * blk: OBS_DIM + (k + 1) * blk] for k in range(2)]
    assert nb[0][5] == 1 and nb[0][8] == 0  # nearest: the parked car (present, no hazard)
    assert nb[1][5] == 1 and nb[1][8] == 1  # the obstacle wreck shows its hazard lights


def test_obstacle_hit_counts_as_crash_not_wreck_hit():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 3.5, -20.0, np.pi / 2, 8.0)
    sim.wreck[0, 1] = sim.obstacle[0, 1] = sim.parked[0, 1] = True
    sim.wreck_left[0, 1] = 100
    sim.x[0, 1], sim.y[0, 1], sim.th[0, 1] = 3.5, -15.0, np.pi / 2
    sim.step(zeros(sim))
    st = sim.stats
    assert st.crashes[0] == 1 and st.obstacle_hits[0] == 1 and st.wreck_hits[0] == 0 and st.obstacles[0] == 0
    assert sim.car_reward[0, 1] == 0.0 and sim.wreck[0, 0] and not sim.obstacle[0, 0]


def test_obstacles_off_draws_nothing():
    """obstacle_envs = 0: no obstacle draws, no state (the bench stays bit-identical)."""
    sim = Sim(SimConfig(demand=0.3), 2, seed=0)
    state = sim.obst_rng.bit_generator.state
    act = zeros(sim)
    for _ in range(50):
        sim.step(act)
    assert sim.obst_rng.bit_generator.state == state and not sim.obstacle.any()
    assert set(STAT_NAMES) >= {"obstacles", "obstacle_hits"}


class Brake:
    """Full braking, straight on: the reference for the stop gate's feasibility."""

    def __init__(self, sim):
        self.sim = sim

    def act(self):
        a = np.zeros((self.sim.B, self.sim.cfg.n_slots, act_dim(self.sim.cfg)))
        a[..., 0] = -1.0
        return a


def test_stop_gate_with_a_braking_driver(monkeypatch):
    from smartcrossing import cli

    monkeypatch.setattr(cli, "make_driver", lambda spec, sim: Brake(sim))
    summary, rows = gates.stop_gate("brake")
    assert summary["cases"] == len(gates.cases()) == 108
    for r in rows:
        assert r["outcome"] == ("stopped" if r["feasible"] else "crash"), r
    assert summary["stop_rate_feasible"] == 1.0 and 0 < summary["feasible"] < summary["cases"]


def test_stop_gate_policy_runs_fast():
    s, _ = gates.stop_gate(f"policy:{ROUND1}")
    assert sum(s[o] for o in gates.OUTCOMES) == s["cases"] and s["wall_s"] < 30


def test_logstd_init_with_warm_start(tmp_path):
    """Variant A: --logstd-init applies with --init-policy, and with --init only via --reset-logstd."""
    a = ppo.parse_args(["--name", "x", "--init-policy", str(ROUND1), "--logstd-init", "-0.7"])
    st = ppo.build_state(a)
    assert np.all(st.logstd == -0.7)
    np.savez(tmp_path / "state.npz", actor=st.actor.p, critic=st.critic.p, logstd=np.full(3, -2.0), ceiling=0.3, iter=5)
    keep = ppo.build_state(ppo.parse_args(["--name", "x", "--init", str(tmp_path / "state.npz"), "--logstd-init", "-0.7"]))
    assert np.all(keep.logstd == -2.0) and keep.it0 == 6
    reset = ppo.build_state(ppo.parse_args(["--name", "x", "--init", str(tmp_path / "state.npz"), "--logstd-init", "-0.7",
                                            "--reset-logstd"]))
    assert np.all(reset.logstd == -0.7)
    np.testing.assert_array_equal(reset.actor.p, st.actor.p)


def test_round2_cfg_roundtrip():
    cfg = SimConfig(**json.loads(json.dumps(dict(SimConfig(crash_cost_speed="own", obstacle_envs=0.25).__dict__))))
    assert cfg.crash_cost_speed == "own" and cfg.obstacle_envs == 0.25
