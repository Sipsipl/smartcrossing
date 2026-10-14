import numpy as np
import pytest

from smartcrossing.baselines import LaneDriver
from smartcrossing.sim import (
    DONE_CRASH,
    DONE_EXIT,
    DONE_NONE,
    DONE_OFFROAD,
    LEFT,
    N_RAYS,
    NARROW,
    OBS_RAYS,
    RIGHT,
    SPAWN_MAX_PER_STEP,
    STRAIGHT,
    Sim,
    SimConfig,
    act_dim,
    obs_dim,
    spawn_lats,
    offroad_point,
    ray_edge_dist,
)


def place(sim, b, n, x, y, th, v=0.0, arm=0, man=STRAIGHT):
    sim.alive[b, n] = True
    sim.x[b, n], sim.y[b, n], sim.th[b, n], sim.v[b, n] = x, y, th, v
    sim.arm[b, n], sim.man[b, n] = arm, man
    sim.age[b, n] = 0.0
    sim.uid[b, n] = sim.next_uid
    sim.next_uid += 1
    sim.stats.spawned[b] += 1
    sim._prev_dist[b, n] = sim.dist_to_gate()[b, n]


def zeros(sim):
    return np.zeros((sim.B, sim.cfg.n_slots, act_dim(sim.cfg)))


def test_spawn_positions_and_headings_all_arms():
    cfg = SimConfig(demand=100.0)  # an arrival on every arm every step
    sim = Sim(cfg, 1, seed=3)
    sim.step(zeros(sim))
    # every arm releases at least one car; more may enter side by side
    assert set(sim.arm[0][sim.alive[0]]) == {0, 1, 2, 3}
    assert sim.alive[0].sum() <= 4 * SPAWN_MAX_PER_STEP
    lim = cfg.W - 1.2
    d = cfg.L - 2.5
    for n in np.nonzero(sim.alive[0])[0]:
        k, x, y, th = sim.arm[0, n], sim.x[0, n], sim.y[0, n], sim.th[0, n]
        # entry point at the end of arm k, heading towards the centre
        expect = {0: (None, -d, np.pi / 2), 1: (d, None, np.pi), 2: (None, d, -np.pi / 2), 3: (-d, None, 0.0)}[k]
        ex, ey, eth = expect
        if ex is None:
            assert y == pytest.approx(ey) and abs(x) <= lim
        else:
            assert x == pytest.approx(ex) and abs(y) <= lim
        assert np.cos(th - eth) == pytest.approx(1.0)
        assert sim.v[0, n] == cfg.spawn_speed
        assert not offroad_point(x, y, cfg.W, cfg.Rc)


@pytest.mark.parametrize("arm", [0, 1, 2, 3])
def test_straight_car_exits_opposite_arm(arm):
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    d = cfg.L - 2.5
    x, y = [(1.0, -d), (d, 1.0), (-1.0, d), (-d, -1.0)][arm]
    place(sim, 0, 0, x, y, np.pi / 2 + arm * np.pi / 2, v=cfg.v_max, arm=arm, man=STRAIGHT)
    for _ in range(200):
        sim.step(zeros(sim))
        if not sim.alive[0].any():
            break
    st = sim.stats
    assert (st.exits[0], st.wrong_exits[0], st.offroad[0], st.crashes[0]) == (1, 0, 0, 0)


def test_turning_car_through_wrong_arm_is_wrong_exit():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0, 1.0, -cfg.L + 2.5, np.pi / 2, v=cfg.v_max, arm=0, man=LEFT)  # goes straight instead
    for _ in range(200):
        sim.step(zeros(sim))
    assert sim.stats.wrong_exits[0] == 1 and sim.stats.exits[0] == 0


def test_offroad_geometry_rounded_corner():
    cfg = SimConfig()
    W, Rc = cfg.W, cfg.Rc
    c = W + Rc
    assert not offroad_point(0.0, 0.0, W, Rc)
    assert not offroad_point(W - 0.01, -40.0, W, Rc)
    assert offroad_point(W + 0.01, -40.0, W, Rc)
    # inside the fillet (rounded curb) the road is wider than the arms
    assert not offroad_point(W + 1.0, W + 1.0, W, Rc)
    # on the 45 deg diagonal the curb is at distance Rc from the fillet centre
    e = c - Rc / np.sqrt(2)
    assert not offroad_point(e - 0.05, -(e - 0.05), W, Rc)
    assert offroad_point(-(e + 0.05), e + 0.05, W, Rc)


def test_offroad_detection_at_corner():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    c = cfg.W + cfg.Rc
    place(sim, 0, 0, c - 1.0, c - 1.0, np.pi / 4)  # inside the north-east corner block
    place(sim, 0, 1, 0.0, -20.0, np.pi / 2)  # centre of a road, fine
    place(sim, 0, 2, cfg.W - 0.5, -30.0, np.pi / 2)  # body sticks over the edge
    sim.step(zeros(sim))
    assert sim.stats.offroad[0] == 2
    assert sim.alive[0, 1] and not sim.alive[0, 0] and not sim.alive[0, 2]
    assert sum(k == "offroad" for _, k, _, _ in sim.events) == 2


def test_collision_of_overlapping_cars():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0, 0.0, -20.0, np.pi / 2)
    place(sim, 0, 1, 1.0, -21.0, np.pi / 2)
    place(sim, 0, 2, -1.5, 20.0, -np.pi / 2)  # alone
    place(sim, 0, 3, 1.5, 20.0, -np.pi / 2)  # side by side, 3 m apart: no contact
    sim.step(zeros(sim))
    assert sim.stats.crashes[0] == 2
    assert not sim.alive[0, 0] and not sim.alive[0, 1] and sim.alive[0, 2] and sim.alive[0, 3]


def test_crn_identical_scenarios_identical_states():
    cfg = SimConfig(demand=0.3, signal_noise=0.1)
    sim = Sim(cfg, 4, seed=7, n_scenarios=2)
    rng = np.random.default_rng(0)
    for _ in range(300):
        sim.observe()
        a = rng.uniform(-1, 1, (2, cfg.n_slots, act_dim(cfg)))
        a[..., 0] = np.abs(a[..., 0])
        a[..., 1] *= 0.05
        sim.step(a[[0, 1, 0, 1]])
    for name in ("alive", "x", "y", "th", "v", "arm", "man", "queue"):
        arr = getattr(sim, name)
        assert np.array_equal(arr[0], arr[2]) and np.array_equal(arr[1], arr[3]), name
    for name in ("arrived", "spawned", "exits", "crashes", "offroad"):
        arr = getattr(sim.stats, name)
        assert arr[0] == arr[2] and arr[1] == arr[3], name
    assert sim.stats.spawned.sum() > 20


def test_crn_queue_attributes_independent_of_policy():
    """Cars keep the attributes drawn at their arrival even if a policy holds the queue
    (the lateral offset depends on which positions are clear, so only the manoeuvre is compared)."""
    cfg = SimConfig(demand=0.3)
    sim = Sim(cfg, 2, seed=11, n_scenarios=1)
    seen = [dict(), dict()]  # per env: uid -> (arm, man, lateral)
    for i in range(400):
        a = zeros(sim)
        a[0, :, 0] = 1.0  # env 0 drives on (straight ahead)
        a[1, :, 0] = np.where(sim.v[1] > 0, -1.0, 0.0)  # env 1 stops at once (no reverse) -> queues back up
        sim.step(a)
        for b in range(2):
            for n in np.nonzero(sim.alive[b])[0]:
                if sim.age[b, n] == 0.0:
                    k = sim.arm[b, n]
                    lat = [sim.x, sim.y, sim.x, sim.y][k][b, n]
                    assert abs(lat) <= sim.cfg.W - 1.2 + 1e-9
                    seen[b][sim.uid[b, n]] = (k, sim.man[b, n])
    for k in range(4):
        s0 = [v[1:] for _, v in sorted(seen[0].items()) if v[0] == k]
        s1 = [v[1:] for _, v in sorted(seen[1].items()) if v[0] == k]
        assert len(s1) < len(s0)
        assert s1 == s0[: len(s1)]


def test_conservation():
    cfg = SimConfig(demand=0.3)
    sim = Sim(cfg, 3, seed=5)
    rng = np.random.default_rng(1)
    for _ in range(400):
        a = rng.uniform(-1, 1, (sim.B, cfg.n_slots, act_dim(cfg)))
        sim.step(a)
        st = sim.stats
        assert np.array_equal(st.arrived, st.spawned + sim.queue.sum(1))
        removed = st.exits + st.wrong_exits + st.crashes + st.offroad + st.timeouts
        assert np.array_equal(st.spawned, sim.alive.sum(1) + removed)
    assert (st.crashes + st.offroad).sum() > 0 and st.spawned.sum() > 30


def test_observation_shape():
    cfg = SimConfig()
    sim = Sim(cfg, 2, seed=0)
    for _ in range(30):
        sim.step(zeros(sim))
    obs = sim.observe()
    assert obs.shape == (2, cfg.n_slots, obs_dim(cfg))
    assert np.isfinite(obs).all()
    assert (obs[~sim.alive] == 0).all()


@pytest.mark.parametrize("control", ["reservation", "light"])
@pytest.mark.parametrize("narrow", [False, True])
def test_rule_based_baselines_are_safe(control, narrow):
    cfg = SimConfig(demand=0.1, **(NARROW if narrow else {}))
    sim = Sim(cfg, 2, seed=1000)
    drv = LaneDriver(sim, control)
    for _ in range(600):
        sim.step(drv.act())
    st = sim.stats
    assert st.crashes.sum() == 0 and st.offroad.sum() == 0
    # (light: cars queue at red; 60 s window, so the bar is modest)
    assert st.exits.sum() >= 0.55 * st.arrived.sum()
    assert st.wrong_exits.sum() == 0


def test_no_control_baseline_stays_on_road():
    cfg = SimConfig(demand=0.2)
    sim = Sim(cfg, 2, seed=1000)
    drv = LaneDriver(sim, "none")
    for _ in range(600):
        sim.step(drv.act())
    assert sim.stats.offroad.sum() == 0 and sim.stats.wrong_exits.sum() == 0


def test_render_smoke(tmp_path):
    from smartcrossing.render import SpaceTime, contact_sheet, draw_frame

    cfg = SimConfig(demand=0.2)
    sim = Sim(cfg, 1, seed=0)
    drv = LaneDriver(sim, "reservation")
    st = SpaceTime(sim)
    frames = []
    for i in range(100):
        sim.step(drv.act())
        st.record()
        if i % 50 == 0:
            frames.append(draw_frame(sim, 0, 200))
    contact_sheet(frames).save(tmp_path / "s.png")
    st.image().save(tmp_path / "st.png")
    assert (tmp_path / "st.png").stat().st_size > 0


def test_turn_manoeuvre_constants():
    assert (RIGHT, STRAIGHT, LEFT) == (1, 2, 3)


def test_steering_rate_limit_and_kappa_obs():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0, 0.0, -30.0, np.pi / 2, v=5.0)
    a = zeros(sim)
    a[0, 0, 1] = 1.0  # full lock target
    a[0, 0, 0] = -0.2  # slow down: at 5 m/s the grip limit (0.16) would bind before kappa_max
    sim.step(a)
    assert sim.kappa[0, 0] == pytest.approx(cfg.kappa_rate * cfg.dt)
    for _ in range(5):
        sim.step(a)
    assert sim.alive[0, 0] and sim.v[0, 0] < 4.4 and sim.kappa[0, 0] == pytest.approx(cfg.kappa_max)
    o = sim.observe()[0, 0]
    assert o[OBS_RAYS + N_RAYS] == pytest.approx(1.0)
    assert o[OBS_RAYS + N_RAYS + 1] == pytest.approx(1.0)  # grip allows full lock at this speed


@pytest.mark.parametrize("v", [6.0, 10.0, 14.0])
def test_grip_limit_caps_lateral_acceleration(v):
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0, 0.0, -45.0, np.pi / 2, v=v)
    a = zeros(sim)
    a[0, 0, 1] = 1.0
    for _ in range(8):
        sim.step(a)
        k_grip = min(cfg.kappa_max, cfg.a_lat_max / sim.v[0, 0] ** 2)
        assert abs(sim.kappa[0, 0]) <= k_grip + 1e-12
        assert sim.v[0, 0] ** 2 * abs(sim.kappa[0, 0]) <= cfg.a_lat_max + 1e-9
    assert sim.kappa[0, 0] == pytest.approx(min(cfg.kappa_max, cfg.a_lat_max / sim.v[0, 0] ** 2))
    o = sim.observe()[0, 0]
    assert o[OBS_RAYS + N_RAYS + 1] == pytest.approx(min(1.0, cfg.a_lat_max / sim.v[0, 0] ** 2 / cfg.kappa_max))
    # the old physics (a_lat_max 0): full lock at any speed
    old = Sim(SimConfig(demand=0.0, **NARROW), 1, seed=0)
    place(old, 0, 0, 0.0, -9.0, np.pi / 2, v=v)  # in the box: room to turn
    for _ in range(5):
        old.step(a)
    assert old.kappa[0, 0] == pytest.approx(cfg.kappa_max)


def test_geometry_defaults_and_spawn_width():
    cfg = SimConfig()
    assert (cfg.W, cfg.Rc) == (7.0, 9.0) and cfg.a_lat_max > 0
    for spawn_right, lo in ((False, -(cfg.W - 1.2)), (True, 1.0)):
        sim = Sim(SimConfig(demand=0.5, spawn_right=spawn_right), 4, seed=5)
        lats = []
        for _ in range(600):
            a = zeros(sim)
            a[..., 0] = 1.0
            sim.step(a)
            new = sim.alive & (sim.age == 0.0)
            c, s = np.rint(np.cos(sim.arm * np.pi / 2)), np.rint(np.sin(sim.arm * np.pi / 2))
            lats += (c * sim.x + s * sim.y)[new].tolist()  # offset right of travel
        lats = np.array(lats)
        assert len(lats) > 400
        assert lats.min() >= lo - 1e-9 and lats.max() <= cfg.W - 1.2 + 1e-9
        assert lats.min() < lo + 0.5 and lats.max() > cfg.W - 1.7
        if not spawn_right:
            assert abs(np.mean(lats > 0) - 0.5) < 0.1  # no side preferred


def test_impact_speed_penalty():
    cfg = SimConfig(demand=0.0)
    rews = []
    for v in (1.0, 8.0):
        sim = Sim(cfg, 1, seed=0)
        place(sim, 0, 0, 0.0, -20.0, np.pi / 2, v=0.0)
        place(sim, 0, 1, 0.0, -24.65, np.pi / 2, v=v)  # rear-ends car 0
        place(sim, 0, 2, cfg.W - 1.0, 20.0, np.pi / 2 + 0.6, v=v)  # leaves the road
        sim.step(zeros(sim))
        assert list(sim.done_kind[0, :3]) == [DONE_CRASH, DONE_CRASH, DONE_OFFROAD]
        rews.append(sim.car_reward[0, :3].copy())
    lo, hi = rews
    # both cars of a pair pay the same impact-speed term (no blame), faster impact costs more
    assert hi[0] < lo[0] - 0.5 and hi[1] < lo[1] - 0.5 and hi[2] < lo[2] - 0.5
    assert hi[0] == pytest.approx(cfg.r_crash - cfg.r_crash_speed * 8.0, abs=0.02)


def test_ray_edge_dist_matches_sampling():
    cfg = SimConfig()
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-12, 12, 400), rng.uniform(-12, 12, 400)
    on = ~offroad_point(x, y, cfg.W, cfg.Rc)
    x, y = x[on], y[on]
    ang = rng.uniform(-np.pi, np.pi, len(x))
    d = ray_edge_dist(x, y, np.cos(ang), np.sin(ang), cfg.W, cfg.Rc)
    t = np.linspace(0, 20, 20001)
    hit = offroad_point(x[:, None] + np.cos(ang)[:, None] * t, y[:, None] + np.sin(ang)[:, None] * t, cfg.W, cfg.Rc)
    first = np.where(hit.any(1), t[hit.argmax(1)], 20.0)
    np.testing.assert_allclose(d, first, atol=2e-3)


def test_car_rewards_and_done_info():
    cfg = SimConfig(demand=0.0, r_time=0.1, r_queue=0.5)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0, 0.0, -20.0, np.pi / 2)
    place(sim, 0, 1, 1.0, -21.0, np.pi / 2)  # crashes with car 0
    place(sim, 0, 2, 1.0, cfg.L - 0.1, np.pi / 2, v=10.0)  # exits north (straight from south)
    place(sim, 0, 3, cfg.W - 0.5, 30.0, np.pi / 2)  # offroad
    place(sim, 0, 4, -1.5, -35.0, np.pi / 2, v=5.0)  # keeps driving
    uids = sim.uid[0].copy()
    sim.queue[0, 1] = SPAWN_MAX_PER_STEP + 2  # some stay queued (at most that many enter per step)
    team = sim.step(zeros(sim))
    k, r = sim.done_kind[0], sim.car_reward[0]
    assert list(k[:5]) == [DONE_CRASH, DONE_CRASH, DONE_EXIT, DONE_OFFROAD, DONE_NONE]
    assert list(sim.done_uid[0, :5]) == list(uids[:5])
    assert r[0] < cfg.r_crash + 0.1 and r[1] < cfg.r_crash + 0.1 and r[2] > 0.9 and r[3] < cfg.r_offroad + 0.1
    assert -0.02 < r[4] < 0.02 and r[4] > -cfg.r_time * cfg.dt - 1e-9  # moving forward offsets the time cost
    assert sim.queue_cost[0] > 0 and team[0] == pytest.approx(r.sum() - sim.queue_cost[0])


def test_demand_range_per_stream():
    cfg = SimConfig(demand_range=(0.02, 0.4))
    sim = Sim(cfg, 8, seed=3, n_scenarios=4)
    assert np.allclose(sim.demand[:4], sim.demand[4:]) and sim.demand.std() > 0.01
    for _ in range(600):
        sim.step(zeros(sim))
    lo, hi = np.argmin(sim.demand[:4]), np.argmax(sim.demand[:4])
    assert sim.stats.arrived[hi] > sim.stats.arrived[lo]


def test_reverse_gear():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0, 2.0, -30.0, np.pi / 2, v=0.5)
    a = zeros(sim)
    a[0, 0, 0] = -1.0
    sim.step(a)
    assert sim.v[0, 0] == 0.0  # braking stops the car, it does not jump into reverse
    for _ in range(20):
        sim.step(a)
    assert sim.v[0, 0] == pytest.approx(-cfg.v_reverse)  # then reverses, capped
    y = sim.y[0, 0]
    a[0, 0, 1] = 1.0  # steer left while reversing: heading turns clockwise
    th0 = sim.th[0, 0]
    for _ in range(5):
        sim.step(a)
    assert sim.y[0, 0] < y and sim.th[0, 0] < th0
    # the old physics: no reverse
    old = Sim(SimConfig(demand=0.0, **NARROW), 1, seed=0)
    place(old, 0, 0, 1.0, -30.0, np.pi / 2, v=0.0)
    old.step(a)
    assert old.v[0, 0] == 0.0


def test_reversing_out_of_the_entry_arm_is_a_wrong_exit():
    cfg = SimConfig(demand=0.0)
    sim = Sim(cfg, 1, seed=0)
    place(sim, 0, 0, 2.0, -cfg.L + 0.1, np.pi / 2, v=-2.0, arm=0, man=STRAIGHT)
    sim.step(zeros(sim))
    from smartcrossing.sim import DONE_WRONG

    assert sim.done_kind[0, 0] == DONE_WRONG and sim.stats.wrong_exits[0] == 1


@pytest.mark.parametrize("spawn_right", [False, True])
def test_spawn_capacity_at_high_demand(spawn_right):
    """With free-flowing traffic the entry is not the bottleneck at 1 arrival / s / arm: a
    blocked spot falls back to another clear one and cars may enter side by side."""
    cfg = SimConfig(demand=1.0, spawn_right=spawn_right, wreck_time=0.0)  # (wrecks would block the box)
    sim = Sim(cfg, 4, seed=2)
    a = zeros(sim)
    a[..., 0] = 1.0
    for _ in range(600):
        sim.step(a)
    st = sim.stats
    assert st.spawned.sum() > 0.9 * st.arrived.sum()
    assert sim.alive.sum(1).max() < cfg.n_slots  # slots are not binding either


def test_spawn_falls_back_to_a_clear_spot():
    cfg = SimConfig(demand=0.0, spawn_right=False)  # full width: the other half has room
    sim = Sim(cfg, 1, seed=0)
    lats = spawn_lats(sim.cfg)
    # standing car right in front of the u = 1 spot (right edge of arm 0): that side is blocked
    place(sim, 0, 0, lats[-1], -cfg.L + 2.5 + 5.0, np.pi / 2)
    sim._qbuf[0, 0, 0] = (1.0 - 1e-9, STRAIGHT, 0)
    sim.queue[0, 0] = 1
    sim.step(zeros(sim))
    new = sim.alive[0] & (sim.age[0] == 0.0)
    assert new.sum() == 1 and sim.queue[0, 0] == 0
    x = sim.x[0][new][0]
    assert x < lats[-1] - 5.0  # entered at the rightmost clear spot instead of waiting


def test_arrival_fifo_growth_keeps_the_queue_order():
    """A tiny initial FIFO that has to grow (several times) gives exactly the default run."""
    cfg = SimConfig(n_slots=4, demand=1.5)  # few slots, high demand: long queues
    runs = []
    for cap in (64, 2):
        sim = Sim(cfg, 3, seed=4, group_probs=[0.6, 0.4])
        sim._qcap, sim._qbuf = cap, np.zeros((3, 4, cap, 3))
        for _ in range(150):
            sim.step(np.tile([0.5, 0.0, 0.0], (3, 4, 1)))
        runs.append(sim)
    a, b = runs
    assert b._qcap >= 16 and a.queue.max() > 8
    for name in ("queue", "man", "group", "arm", "uid", "x", "y"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
