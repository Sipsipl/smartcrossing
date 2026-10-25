"""Obstacle scenarios (round 2 variant B: wrecks / parked cars on a path in some envs), the crash cost
by own impact speed (variant C, kept for old checkpoints) and the round-2 SimConfig fields."""

import json

import numpy as np
import pytest

from smartcrossing.sim import (
    BLOCK_BODIES,
    BLOCK_CURB_MARGIN,
    DISC_OFFSETS,
    DISC_R,
    DONE_CRASH,
    DONE_OFFROAD,
    OBSTACLE_ARM_END,
    NB_FEATURES,
    OBS_DIM,
    STAT_NAMES,
    Sim,
    SimConfig,
    act_dim,
    block_pose,
    corner_dist,
    obstacle_clear,
    obstacle_steps,
)

from test_wrecks import place, zeros  # noqa: E402 (tests/ is on sys.path)


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


@pytest.mark.parametrize("block", [0.0, 1.0])
def test_obstacle_placement_keeps_clearance(block):
    """An obstacle (every body of a blockade) never appears where a car could no longer stop in front of it."""
    cfg = obstacle_cfg(demand=0.5, obstacle_rate=10.0, obstacle_block=block, n_slots=40)
    sim = Sim(cfg, 6, seed=1)
    act = zeros(sim)
    act[..., 0] = 0.5
    n_new = n_blocks = 0
    for _ in range(300):
        before = sim.obstacle.copy()
        prev_uids = sim.uid.copy()
        sim.step(act)
        new = sim.obstacle & ~before
        n_blocks += int((new.sum(1) == BLOCK_BODIES).sum())
        for b, n in zip(*np.nonzero(new)):
            n_new += 1
            # positions after this step's motion are what the placement saw: check against them
            others = sim.alive[b] | (sim.wreck[b] & ~new[b])
            spawned_now = sim.alive[b] & ~np.isin(sim.uid[b], prev_uids[b])  # entered after the placement
            m = others & ~spawned_now
            dx, dy = sim.x[b, n] - sim.x[b][m], sim.y[b, n] - sim.y[b][m]
            vx, vy = sim.v[b][m] * np.cos(sim.th[b][m]), sim.v[b][m] * np.sin(sim.th[b][m])
            assert obstacle_clear(cfg, dx, dy, vx, vy).all()
    assert n_new > 5 and (n_blocks > 2) == (block > 0)


def test_every_obstacle_shows_hazard_light():
    """Owner 2026-10-24: hazard = not a participant; parked-type obstacles show it too."""
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
    assert nb[0][5] == 1 and nb[0][8] == 1  # nearest: the parked-type obstacle (present, hazard lights)
    assert nb[1][5] == 1 and nb[1][8] == 1  # the wreck-type obstacle shows its hazard lights
    assert nb[0][7] == 0 and nb[0][NB_FEATURES:].sum() == 0  # brake light off, lamp 0


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


def test_round2_cfg_roundtrip():
    cfg = SimConfig(**json.loads(json.dumps(dict(SimConfig(crash_cost_speed="own", obstacle_envs=0.25).__dict__))))
    assert cfg.crash_cost_speed == "own" and cfg.obstacle_envs == 0.25


# ---------------------------------------------------------------- blockades (cfg.obstacle_block, blockade scenarios of round 7)
def _block_u(k, r_frac, face):
    u = np.zeros(8)
    u[3], u[5], u[6] = (k + 0.5) / 4, r_frac, face
    return u


def _canonical(k, x, y):
    """World (x, y) on arm k -> canonical frame of arm 0 (lateral, along: -distance to the centre)."""
    a = -k * np.pi / 2
    c, s = np.rint(np.cos(a)), np.rint(np.sin(a))
    return c * x - s * y, s * x + c * y


def test_blockade_leaves_no_passable_gap():
    """Geometric check on every arm, both line distances and facings: across the full road width (both
    directions, curb to curb) every point of the blockade line is within a disc diameter of a body disc
    (a car's disc touches it) or off-road for a car's disc - no gap, not even through the oncoming half."""
    cfg = SimConfig()  # (the training world: W = 7, Rc = 9)
    for k in range(4):
        for r_frac in (0.0, 1.0):
            for face in (0.2, 0.8):
                x, y, th = block_pose(cfg, _block_u(k, r_frac, face))
                assert x.shape == (BLOCK_BODIES,)
                cx, cy = _canonical(k, x, y)
                assert np.allclose(cy, cy[0]) and -cfg.L + OBSTACLE_ARM_END - 1e-9 <= cy[0]
                assert cy[0] <= -(cfg.W + cfg.Rc + BLOCK_CURB_MARGIN) + 1e-9  # where the arm is straight
                assert np.allclose(np.abs(np.cos(th - k * np.pi / 2)), 1.0)  # crosswise to the road
                # body disc centres (canonical lateral); none of the discs is off-road itself
                dx = cx[:, None] + np.cos(th - k * np.pi / 2)[:, None] * DISC_OFFSETS
                assert (corner_dist(dx, cy[0] + 0 * dx, cfg.W, cfg.Rc) >= cfg.Rc + DISC_R - 1e-9).all()
                s = np.linspace(-cfg.W, cfg.W, 2801)  # car disc centres along the line
                touch = (np.abs(s[:, None] - dx.ravel()[None]) < 2 * DISC_R).any(1)
                off = corner_dist(s, cy[0] + 0 * s, cfg.W, cfg.Rc) < cfg.Rc + DISC_R
                assert (touch | off).all()
                gaps = np.diff(np.sort(dx.ravel())) - 2 * DISC_R  # surface gaps between neighbouring discs
                assert gaps.max() < 2 * DISC_R


def test_blockade_stops_every_car_driving_through():
    """Cars driving straight at the blockade at every lateral offset (both halves, curb to curb) crash or go
    off-road: none gets through."""
    cfg = SimConfig(demand=0.0, n_slots=1 + BLOCK_BODIES)
    lats = np.linspace(-(cfg.W - DISC_R - 0.05), cfg.W - DISC_R - 0.05, 41)
    sim = Sim(cfg, len(lats), seed=0)
    x, y, th = block_pose(cfg, _block_u(0, 0.5, 0.8)[None].repeat(len(lats), 0))
    for b, lat in enumerate(lats):
        place(sim, 0, lat, y[b, 0] - 12.0, np.pi / 2, 6.0, b=b)
        for j in range(BLOCK_BODIES):
            n = 1 + j
            sim.wreck[b, n] = sim.obstacle[b, n] = True
            sim.wreck_left[b, n] = 1000
            sim.x[b, n], sim.y[b, n], sim.th[b, n] = x[b, j], y[b, j], th[b, j]
    act = zeros(sim)
    done = np.zeros(len(lats), np.int8)
    for _ in range(60):
        sim.step(act)
        done = np.where(done == 0, sim.done_kind[:, 0], done)
    assert np.isin(done, (DONE_CRASH, DONE_OFFROAD)).all() and (done == DONE_CRASH).sum() >= len(lats) - 2


def test_blockades_in_traffic():
    """obstacle_block = 1: placements are whole blockades (BLOCK_BODIES bodies in one step, one arm, one type,
    same lifetime); the limit counts the bodies standing before a placement."""
    cfg = obstacle_cfg(demand=0.2, obstacle_block=1.0, obstacle_max=2, obstacle_time=3.0, n_slots=40)
    sim = Sim(cfg, 8, seed=2)
    act = zeros(sim)
    act[..., 0] = 0.3
    n_blocks = 0
    px, py = sim.x.copy(), sim.y.copy()
    for _ in range(400):
        before = sim.obstacle.copy()
        sim.step(act)
        new = sim.obstacle & (~before | (sim.x != px) | (sim.y != py))  # (a slot may get a new obstacle right away)
        px, py = sim.x.copy(), sim.y.copy()
        for b in np.nonzero(new.any(1))[0]:
            n = np.nonzero(new[b])[0]
            assert len(n) == BLOCK_BODIES and (sim.obstacle[b] & ~new[b]).sum() < cfg.obstacle_max
            assert len(set(sim.parked[b, n])) == 1 and len(set(sim.wreck_left[b, n])) == 1
            arm = int(np.argmax(np.stack([-sim.y[b, n], sim.x[b, n], sim.y[b, n], -sim.x[b, n]]).mean(1)))
            cx, cy = _canonical(arm, sim.x[b, n], sim.y[b, n])
            assert np.allclose(cy, cy[0]) and np.allclose(np.sort(cx), [-2 * cfg.W / 3, 0.0, 2 * cfg.W / 3])
            n_blocks += 1
        assert not sim.obstacle[4:].any() and (sim.obstacle.sum(1) <= cfg.obstacle_max - 1 + BLOCK_BODIES).all()
    assert n_blocks > 10 and sim.stats.obstacles.sum() == BLOCK_BODIES * n_blocks


def _state_hash(**kw):
    """Hash of a sim state after 200 steps of straight driving in traffic (obstacles on / off)."""
    import hashlib

    cfg = SimConfig(**{**dict(n_slots=24, demand=0.3), **kw})
    sim = Sim(cfg, 6, seed=11)
    act = np.zeros((6, cfg.n_slots, act_dim(cfg)))
    act[..., 0] = 0.3
    for _ in range(200):
        sim.step(act)
    h = hashlib.sha1()
    for name in ("x", "y", "th", "v", "age"):
        h.update(np.round(getattr(sim, name), 6).tobytes())
    for name in ("alive", "wreck", "obstacle", "parked", "wreck_left", "uid", "queue"):
        h.update(np.ascontiguousarray(getattr(sim, name)).astype(np.int64).tobytes())
    for name in STAT_NAMES:
        h.update(getattr(sim.stats, name).tobytes())
    return h.hexdigest()[:16], int(sim.stats.obstacles.sum())


# state hashes of the code before obstacle_block existed (commit 2ac972f): obstacle_block = 0 reproduces them
HASH_OFF = "7393d45ef98e50e5"
HASH_OBST = "5e0ee350730f9b64"
OBST = dict(obstacle_envs=0.5, obstacle_rate=2.0, obstacle_time=3.0, obstacle_max=3)


def test_obstacle_block_off_is_bit_identical():
    assert _state_hash()[0] == HASH_OFF
    assert _state_hash(**OBST)[0] == HASH_OBST == _state_hash(**OBST, obstacle_block=0.0)[0]
    assert _state_hash(**OBST, obstacle_block=0.5)[0] != HASH_OBST
