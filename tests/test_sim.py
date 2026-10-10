import numpy as np
import pytest

from smartcrossing.baselines import LaneDriver
from smartcrossing.sim import LEFT, RIGHT, STRAIGHT, Sim, SimConfig, act_dim, obs_dim, offroad_point


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
    assert sim.alive[0].sum() == 4
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
    """Cars keep the attributes drawn at their arrival even if a policy holds the queue."""
    cfg = SimConfig(demand=0.3)
    sim = Sim(cfg, 2, seed=11, n_scenarios=1)
    seen = [dict(), dict()]  # per env: uid -> (arm, man, lateral)
    for i in range(400):
        a = zeros(sim)
        a[0, :, 0] = 1.0  # env 0 drives on (straight ahead)
        a[1, :, 0] = -1.0  # env 1 stops at once -> its queues back up
        sim.step(a)
        for b in range(2):
            for n in np.nonzero(sim.alive[b])[0]:
                if sim.age[b, n] == 0.0:
                    k = sim.arm[b, n]
                    lat = [sim.x, sim.y, sim.x, sim.y][k][b, n]
                    seen[b][sim.uid[b, n]] = (k, sim.man[b, n], round(float(lat), 9))
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
def test_rule_based_baselines_are_safe(control):
    cfg = SimConfig(demand=0.1)
    sim = Sim(cfg, 2, seed=1000)
    drv = LaneDriver(sim, control)
    for _ in range(600):
        sim.step(drv.act())
    st = sim.stats
    assert st.crashes.sum() == 0 and st.offroad.sum() == 0
    assert st.exits.sum() >= 0.6 * st.arrived.sum()
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
