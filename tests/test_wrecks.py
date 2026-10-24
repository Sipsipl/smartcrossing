"""Next training round (2026-10-11): wrecks, brake light + hazard neighbour features, one lamp,
right-half spawning; and that the old behaviour is still reproducible."""

import json
from pathlib import Path

import numpy as np
import pytest

from smartcrossing.baselines import LaneDriver
from smartcrossing.policy import PolicyDriver
from smartcrossing.policy_compat import interface_cfg, obs_keys, physics_of
from smartcrossing.sim import (
    BRAKE_ACC,
    DONE_CRASH,
    NB_FEATURES,
    NB_FEATURES_LEGACY,
    OBS_DIM,
    STRAIGHT,
    Sim,
    SimConfig,
    act_dim,
    obs_dim,
    spawn_lats,
    wreck_steps,
)

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "results" / "wide_s1ft2_ckpt2650.npz"


def place(sim, n, x, y, th, v=0.0, arm=0, man=STRAIGHT, b=0):
    sim.alive[b, n] = True
    sim.x[b, n], sim.y[b, n], sim.th[b, n], sim.v[b, n] = x, y, th, v
    sim.arm[b, n], sim.man[b, n] = arm, man
    sim.age[b, n] = 0.0
    sim.uid[b, n] = sim.next_uid
    sim.next_uid += 1
    sim._prev_dist[b, n] = sim.dist_to_gate()[b, n]


def zeros(sim):
    return np.zeros((sim.B, sim.cfg.n_slots, act_dim(sim.cfg)))


def nb_block(sim, obs, n, k=0):
    blk = (NB_FEATURES if sim.cfg.brake_light else NB_FEATURES_LEGACY) + sim.cfg.n_signal
    return obs[0, n, OBS_DIM + k * blk: OBS_DIM + (k + 1) * blk]


def test_new_defaults():
    cfg = SimConfig()
    assert cfg.n_signal == 1 and cfg.brake_light and cfg.spawn_right and cfg.wreck_time == SimConfig().wreck_time
    assert obs_dim(cfg) == OBS_DIM + 4 * (NB_FEATURES + 1) and act_dim(cfg) == 3
    assert wreck_steps(cfg) == round(cfg.wreck_time / cfg.dt)
    old = SimConfig(n_signal=2, brake_light=False)
    assert obs_dim(old) == OBS_DIM + 4 * (NB_FEATURES_LEGACY + 2)
    assert obs_keys(cfg)[OBS_DIM:OBS_DIM + 10] == [f"n1_{k}" for k in ("fwd", "left", "same", "cross", "speed", "present",
                                                                     "kappa", "brake", "hazard", "lamp1")]
    assert "n1_acc" in obs_keys(old) and "n1_hazard" not in obs_keys(old)


def crash_pair(cfg, v=5.0):
    """Two cars head-on, overlapping after one step (a crash), plus the sim."""
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0.0, -2.5, np.pi / 2, v)
    place(sim, 1, 0.0, 2.5, -np.pi / 2, v, arm=2)
    sim.step(zeros(sim))
    return sim


def test_wreck_stays_exactly_wreck_time_and_blocks():
    cfg = SimConfig(demand=0.0)
    sim = crash_pair(cfg)
    assert (sim.done_kind[0, :2] == DONE_CRASH).all() and sim.stats.crashes[0] == 2
    assert sim.wreck[0, :2].all() and not sim.alive[0, :2].any()
    pos = (sim.x[0, :2].copy(), sim.y[0, :2].copy(), sim.th[0, :2].copy())
    uids = sim.uid[0, :2].copy()
    assert (uids >= 0).all() and (sim.v[0, :2] == 0).all()
    present = 1  # states after a step in which the wrecks exist (the crash step included)
    while sim.wreck[0, :2].any():
        assert sim.wreck[0, :2].all()
        assert np.array_equal(sim.x[0, :2], pos[0]) and np.array_equal(sim.y[0, :2], pos[1])
        assert np.array_equal(sim.th[0, :2], pos[2]) and np.array_equal(sim.uid[0, :2], uids)
        sim.step(zeros(sim))
        assert sim.car_reward[0, :2].sum() == 0.0 and (sim.done_kind[0, :2] == 0).all()
        present += 1 if sim.wreck[0, :2].any() else 0
    assert present * cfg.dt == pytest.approx(cfg.wreck_time)  # exactly wreck_time / dt states
    assert (sim.uid[0, :2] == -1).all() and sim.stats.crashes[0] == 2  # no recount, no reward


def test_hitting_a_wreck_is_a_crash_for_the_hitter():
    cfg = SimConfig(demand=0.0)
    sim = crash_pair(cfg)
    wx, wy = sim.x[0, 0], sim.y[0, 0]
    # a third car approaches the wreck from the west at 6 m/s
    v = 6.0
    place(sim, 2, wx - 6.0, wy, 0.0, v, arm=3)
    a = zeros(sim)
    a[0, 2, 0] = 0.0  # keep speed (no pedal)
    hit_step = None
    for k in range(20):
        sim.step(a)
        if sim.done_kind[0, 2] == DONE_CRASH:
            hit_step = k
            break
        assert sim.alive[0, 2]
    assert hit_step is not None
    assert sim.car_reward[0, 2] == pytest.approx(cfg.r_crash - cfg.r_crash_speed * v + cfg.r_progress * 0.0, abs=0.2)
    # impact speed = own speed against the standing wreck (the progress term aside)
    prog = sim.car_reward[0, 2] - (cfg.r_crash - cfg.r_crash_speed * v)
    assert abs(prog) < cfg.r_progress * v * cfg.dt * 1.01 + 1e-9
    assert sim.stats.crashes[0] == 3 and sim.stats.wreck_hits[0] == 1  # the old wrecks are not recounted
    assert sim.wreck[0, 2] and sim.wreck_left[0, 2] == wreck_steps(cfg)
    assert sim.wreck_left[0, 0] < wreck_steps(cfg)  # the first wrecks keep their own timer


def test_wrecks_are_visible_with_hazard_lights_and_occupy_slots():
    cfg = SimConfig(demand=0.0, n_slots=4)
    sim = crash_pair(cfg)
    place(sim, 2, -10.0, 0.0, 0.0, 0.0, arm=3)
    obs = sim.observe()
    assert not obs[0, :2].any()  # wrecks get no observation (not driven)
    f = nb_block(sim, obs, 2)
    assert f[5] == 1.0 and f[4] == 0.0 and f[7] == 0.0 and f[8] == 1.0  # present, speed 0, no brake, hazard
    # legacy interface: same neighbour, acceleration feature (0), no hazard feature
    sim.cfg.brake_light = False
    f = nb_block(sim, sim.observe(), 2)
    assert len(f) == NB_FEATURES_LEGACY + cfg.n_signal and f[5] == 1.0 and f[7] == 0.0
    sim.cfg.brake_light = True
    # slots: 2 wrecks + 1 car leave one free slot; a queued car takes it, the next one waits
    sim.queue[0, 1] = 2
    sim._qbuf[0, 1, :2] = (0.5, STRAIGHT, 0)
    sim.step(zeros(sim))
    assert (sim.alive | sim.wreck)[0].sum() == 4 and sim.queue[0, 1] == 1
    for _ in range(wreck_steps(cfg)):
        sim.step(zeros(sim))
    assert not sim.wreck[0, :2].any()  # the first wrecks are gone (the entering car may have crashed into the standing one)


def test_spawn_waits_for_a_wreck_at_the_entry():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    lats = spawn_lats(sim.cfg)
    y0 = -cfg.L + 2.5 + 3.0
    for i, lat in enumerate((lats[0], lats[-1])):  # wrecks across the right half, right after the spawn line
        place(sim, i, lat, y0, np.pi / 2)
    sim._make_wreck(sim.alive.copy())
    sim.queue[0, 0] = 1
    sim._qbuf[0, 0, 0] = (0.5, STRAIGHT, 0)
    for _ in range(wreck_steps(cfg) - 1):
        sim.step(zeros(sim))
        assert sim.queue[0, 0] == 1 and not sim.alive.any()
    sim.step(zeros(sim))  # wrecks gone: the car enters
    assert sim.queue[0, 0] == 0 and sim.alive.sum() == 1


@pytest.mark.parametrize("acc_action, lit", [(-0.1, True), (-0.05, False), (0.5, False)])
def test_brake_light_matches_threshold(acc_action, lit):
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0.0, -30.0, np.pi / 2, 10.0)
    place(sim, 1, 3.0, -40.0, np.pi / 2, 10.0)
    a = zeros(sim)
    a[0, 0, 0] = acc_action  # brake_max 7: -0.1 -> -0.7 m/s^2, -0.05 -> -0.35 m/s^2
    sim.step(a)
    assert (sim.acc[0, 0] < BRAKE_ACC) == lit
    f = nb_block(sim, sim.observe(), 1)
    assert f[7] == float(lit) and f[8] == 0.0


def test_right_half_spawn_default():
    sim = Sim(SimConfig(demand=0.5), 2, seed=1)
    assert spawn_lats(sim.cfg).min() >= 1.0
    for _ in range(200):
        a = zeros(sim)
        a[..., 0] = 1.0
        sim.step(a)
        new = sim.alive & (sim.age == 0.0)
        c, s = np.rint(np.cos(sim.arm * np.pi / 2)), np.rint(np.sin(sim.arm * np.pi / 2))
        assert ((c * sim.x + s * sim.y)[new] >= 1.0 - 1e-9).all()


def test_wreck_time_zero_reproduces_the_old_behaviour():
    """Committed reference (tests/data/legacy_behaviour.json, recorded with the sim before wrecks /
    brake lights / one lamp / right-half spawning, commit 27a54b2): the main brain in its own
    interface and the baselines, wreck_time = 0, full-width spawning."""
    ref = json.loads((ROOT / "tests" / "data" / "legacy_behaviour.json").read_text(encoding="utf-8"))
    for name, r in ref.items():
        legacy = {"wreck_time": 0.0, "n_signal": r["n_signal"]}
        if name == "brain":
            legacy["spawn_right"] = False
        sim = Sim(SimConfig(demand=r["demand"], **legacy), r["B"], seed=r["seed"])
        drv = PolicyDriver.load(BRAIN, sim) if name == "brain" else LaneDriver(sim, name)
        tot = 0.0
        for _ in range(r["steps"]):
            tot += float(sim.step(drv.act()).sum())
        for k, v in r["stats"].items():
            assert getattr(sim.stats, k).tolist() == v, (name, k)
        assert sim.uid.tolist() == r["uid"], name
        for k in ("x", "y", "v"):
            np.testing.assert_allclose(getattr(sim, k), r[k], atol=1e-8, err_msg=f"{name} {k}")
        assert tot == pytest.approx(r["reward_sum"], abs=1e-8)


def test_old_brain_loads_with_its_interface_in_the_new_world():
    sim = Sim(SimConfig(demand=0.3), 2, seed=3)
    drv = PolicyDriver.load(BRAIN, sim)
    assert sim.cfg.n_signal == 2 and not sim.cfg.brake_light  # its own interface
    assert sim.cfg.spawn_right and sim.cfg.wreck_time == SimConfig().wreck_time  # but the current world
    assert drv.sizes[0] == obs_dim(sim.cfg) and sim.sig.shape[-1] == 2
    for _ in range(300):
        sim.step(drv.act())
    assert sim.stats.exits.sum() > 0
    phys = physics_of(drv.train_cfg)
    assert phys["n_signal"] == 2 and phys["brake_light"] is False
    assert interface_cfg(drv.train_cfg).n_signal == 2
    assert physics_of(SimConfig().__dict__)["n_signal"] == 1


@pytest.mark.parametrize("control", ["light", "reservation"])
def test_baselines_stop_behind_a_wreck(control):
    """The rule-based drivers treat wrecks as standing obstacles (car following) and drive on
    once the wreck is gone."""
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    drv = LaneDriver(sim, control)
    for i, lat in enumerate((1.75, 5.25)):  # both right lanes blocked on the south arm
        place(sim, i, lat, -25.0, np.pi / 2)
    sim._make_wreck(sim.alive.copy())
    sim.queue[0, 0] = 1
    sim._qbuf[0, 0, 0] = (0.5, STRAIGHT, 0)
    min_gap = np.inf
    for _ in range(250):
        sim.step(drv.act())
        car = sim.alive[0]
        if car.any() and sim.wreck[0].any():
            min_gap = min(min_gap, float((-25.0 - 2.3) - (sim.y[0][car].max() + 2.3)))
    assert sim.stats.crashes[0] == 0 and sim.stats.spawned[0] == 1
    assert sim.stats.exits[0] == 1  # passed after the wrecks vanished
    assert 0.0 < min_gap < 10.0  # it waited close behind them
