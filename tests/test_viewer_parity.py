"""Parity between the Python simulator/policy and the browser port in viewer/sim.js.

Python builds a set of states (a random-traffic trajectory plus hand-made edge
cases: near-collisions, cars at the curb, exits through right/wrong arms,
timeouts, queued cars at blocked spawn points), runs ``observe`` and one
``step`` with fixed actions and arrivals disabled (demand 0, queued cars still
spawn), and a node script does the same with sim.js. Everything must agree to 1e-6.
"""

import copy
import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from smartcrossing.policy import PolicyDriver, default_in_scale, export_json, init_params, layer_sizes, save_npz
from smartcrossing.sim import LEFT, RIGHT, STRAIGHT, Sim, SimConfig, act_dim

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "viewer" / "test" / "parity_runner.js"
NODE = shutil.which("node")
TOL = 1e-6

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


# ---------------------------------------------------------------- state export
def export_state(sim: Sim) -> dict:
    assert sim.B == 1
    st = sim.stats
    queues = []
    for k in range(4):
        q = []
        for j in range(int(sim.queue[0, k])):
            lat, man, grp = sim._qbuf[0, k, (sim._qhead[0, k] + j) % sim._qcap]
            q.append([float(lat), int(man), int(grp)])
        queues.append(q)
    return {
        "alive": sim.alive[0].tolist(),
        "x": sim.x[0].tolist(),
        "y": sim.y[0].tolist(),
        "th": sim.th[0].tolist(),
        "v": sim.v[0].tolist(),
        "kappa": sim.kappa[0].tolist(),
        "age": sim.age[0].tolist(),
        "prev_dist": sim._prev_dist[0].tolist(),
        "sig": sim.sig[0].tolist(),
        "arm": sim.arm[0].tolist(),
        "man": sim.man[0].tolist(),
        "uid": sim.uid[0].tolist(),
        "group": sim.group[0].tolist(),
        "queues": queues,
        "next_uid": int(sim.next_uid),
        "t": float(sim.t),
        "stats": {
            k: int(getattr(st, k)[0])
            for k in ("arrived", "spawned", "exits", "wrong_exits", "crashes", "offroad", "timeouts")
        },
    }


def cfg_dict(cfg: SimConfig) -> dict:
    d = dataclasses.asdict(cfg)
    d["turn_probs"] = list(d["turn_probs"])
    return d


def make_case(name, sim: Sim, action) -> dict:
    """Run observe + step on a copy of `sim` with arrivals disabled; record both sides."""
    cp = copy.deepcopy(sim)
    cp.cfg = dataclasses.replace(sim.cfg, demand=0.0)
    before = export_state(cp)
    obs = cp.observe()[0]
    n_travel = len(cp.stats.travel[0])
    reward = float(cp.step(action[None])[0])
    return {
        "name": name,
        "cfg": cfg_dict(cp.cfg),
        "state": before,
        "action": np.asarray(action).tolist(),
        "expect": {
            "obs": obs,
            "reward": reward,
            "car_reward": cp.car_reward[0].copy(),
            "done_kind": cp.done_kind[0].tolist(),
            "done_uid": cp.done_uid[0].tolist(),
            "queue_cost": float(cp.queue_cost[0]),
            "state": export_state(cp),
            "travel": np.array(cp.stats.travel[0][n_travel:]).reshape(-1, 2),
            "events": [(k, float(x), float(y)) for _, k, x, y in cp.events],
        },
    }


def place(sim, n, x, y, th, v, arm=0, man=STRAIGHT, age=1.0, sig=(0.0, 0.0), group=0):
    cfg = sim.cfg
    sim.alive[0, n] = True
    sim.x[0, n], sim.y[0, n], sim.th[0, n], sim.v[0, n] = x, y, th, v
    sim.arm[0, n], sim.man[0, n], sim.age[0, n], sim.group[0, n] = arm, man, age, group
    sim.sig[0, n, : cfg.n_signal] = sig[: cfg.n_signal]
    sim.uid[0, n] = sim.next_uid
    sim.next_uid += 1
    sim._prev_dist[0, n] = sim.dist_to_gate()[0, n] + 0.3


def push_queue(sim, k, lat, man, grp=0):
    j = (sim._qhead[0, k] + sim.queue[0, k]) % sim._qcap
    sim._qbuf[0, k, j] = (lat, man, grp)
    sim.queue[0, k] += 1


# ---------------------------------------------------------------- cases
def trajectory_cases(seed, cfg, n_steps=160, every=4):
    sim = Sim(cfg, 1, seed=seed)
    sim.group_probs = np.array([0.5, 0.3, 0.2])
    rng = np.random.default_rng(seed + 100)
    N, A = cfg.n_slots, act_dim(cfg)
    steer = rng.normal(0, 0.25, N)
    cases = []
    for t in range(n_steps):
        if t % 25 == 0:
            steer = rng.normal(0, 0.25, N)
        a = np.empty((N, A))
        a[:, 0] = rng.uniform(-0.6, 1.0, N)
        a[:, 1] = steer + rng.normal(0, 0.1, N)
        a[:, 2:] = rng.uniform(-1.3, 1.3, (N, A - 2))  # some beyond [-1, 1] to test clipping
        if t % every == 0 and sim.alive.any():
            cases.append(make_case(f"traj{seed}_t{t}", sim, a))
        sim.step(a[None])
    return cases


def edge_cases(cfg):
    N, A = cfg.n_slots, act_dim(cfg)
    W, L = cfg.W, cfg.L
    hp = np.pi / 2
    sim = Sim(dataclasses.replace(cfg, demand=0.0), 1, seed=0)
    sim.t = 12.3
    # near-collisions: nose-to-tail pairs just inside / just outside contact
    place(sim, 0, 0.5, -30.0, hp, 0.0, sig=(0.7, -0.2))
    place(sim, 1, 0.5, -34.55, hp, 0.5)  # gap 4.55 + motion -> contact
    place(sim, 2, -1.0, -20.0, hp, 3.0)
    place(sim, 3, -1.0, -24.595, hp, 3.0, sig=(-1.0, 1.0))  # disc gap 1.795 < 1.8 -> contact
    place(sim, 14, 2.0, -40.0, hp, 3.0)
    place(sim, 15, 2.0, -44.605, hp, 3.0)  # disc gap 1.805 -> no contact
    # crossing paths in the box
    place(sim, 4, 0.0, 2.0, 0.0, 6.0, arm=3, man=STRAIGHT)
    place(sim, 5, 2.4, 0.3, hp + 0.2, 6.0, arm=0, man=LEFT, group=1)
    # near curb in the north arm: one barely inside, one barely over
    place(sim, 6, W - 0.905, 25.0, hp, 5.0, arm=2, man=STRAIGHT)
    place(sim, 7, -(W - 0.895), 35.0, -hp, 5.0, arm=2, man=STRAIGHT)
    # near a rounded corner
    place(sim, 8, W + 1.2, W + 1.3, np.pi / 4, 4.0, arm=0, man=RIGHT)
    # exits: correct (north, straight from south), wrong (east, straight from south), west correct
    place(sim, 9, 1.0, L - 0.2, hp, 10.0, arm=0, man=STRAIGHT, age=7.0)
    place(sim, 10, L - 0.1, 1.0, 0.0, 9.0, arm=0, man=STRAIGHT, age=8.0)
    place(sim, 11, -L + 0.3, -1.0, np.pi, 12.0, arm=0, man=LEFT, age=9.0)
    # timeout and heading wrap-around
    place(sim, 12, 0.0, 40.0, hp, 2.0, arm=1, man=RIGHT, age=cfg.max_age - 0.05)
    place(sim, 13, 1.5, -10.0, -np.pi + 0.005, 4.0, arm=1, man=LEFT, group=2)
    cases = []
    rng = np.random.default_rng(7)
    for i in range(3):
        a = rng.uniform(-1, 1, (N, A))
        a[1, 0], a[0, 0] = 1.0, -1.0
        a[13, 1] = -1.0
        if i == 0:  # threshold cases: keep them exactly on their lines
            a[[2, 3, 6, 7, 14, 15], :2] = 0.0
        cases.append(make_case(f"edge_{i}", sim, a))
        a2 = rng.uniform(-0.2, 0.2, (N, A))
        sim.step(a2[None])  # evolve the same state a little (no arrivals: demand 0)

    # spawning from queues: arm 0 blocked by a standing car, arm 1 clear, arm 2 blocked
    # by an oncoming car in the corridor, arm 3 clear but a fast car closing from behind
    sp = Sim(dataclasses.replace(cfg, demand=0.0), 1, seed=1)
    sp.t = 5.0
    place(sp, 0, 0.3, -L + 2.5 + 6.0, hp, 0.0)
    place(sp, 1, 0.0, L - 2.5 - 20.0, hp, 12.0, arm=0, man=STRAIGHT)  # heads into arm-2 spawn
    place(sp, 2, -(L - 2.5 - 14.0), 1.0, np.pi, 6.0, arm=1, man=STRAIGHT)  # moving towards arm-3 spawn
    for k, (lat, man, grp) in enumerate([(0.5, RIGHT, 0), (-1.0, LEFT, 1), (0.0, STRAIGHT, 2), (1.5, LEFT, 0)]):
        push_queue(sp, k, lat, man, grp)
        push_queue(sp, k, -lat, STRAIGHT, 1)
    a = np.zeros((N, A))
    cases.append(make_case("spawn", sp, a))
    sp2 = copy.deepcopy(sp)
    sp2.alive[0] = False
    sp2.x[0], sp2.y[0] = 1e4, 1e4
    cases.append(make_case("spawn_all_clear", sp2, a))
    return cases


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    cfg = SimConfig(n_slots=16, r_time=0.05, r_queue=0.1)
    cases = trajectory_cases(0, dataclasses.replace(cfg, demand=0.5))
    cases += trajectory_cases(1, dataclasses.replace(cfg, demand=0.35, signal_enabled=False, turn_probs=(0.5, 0.2, 0.3)))
    cases += edge_cases(cfg)

    # policy: real export format (save_npz -> export_json)
    sizes = layer_sizes(cfg)
    params = [init_params(sizes, s, out_scale=1.0) for s in (3, 4)]
    scale = default_in_scale(cfg)
    pols = []
    for i, p in enumerate(params):
        save_npz(tmp / f"p{i}.npz", p, sizes, cfg, in_scale=scale)
        pols.append(export_json(tmp / f"p{i}.npz"))
    obs = np.concatenate([c["expect"]["obs"] for c in cases[:12]])
    obs = np.concatenate([obs, np.random.default_rng(5).normal(size=(20, obs.shape[1]))])
    # driver with two groups, group 1 muted
    dsim = Sim(dataclasses.replace(cfg, demand=0.6), 1, seed=3)
    dsim.group_probs = np.array([0.5, 0.5])
    for _ in range(120):
        dsim.step(np.tile([0.4, 0.0, 0.5, -0.5], (cfg.n_slots, 1))[None])
    assert len(set(dsim.group[0][dsim.alive[0]])) == 2
    driver = PolicyDriver(dsim, params, sizes, mute=[False, True], in_scale=scale)
    expect_drv = driver.act()[0]

    payload = {
        "cases": [{k: c[k] for k in ("name", "cfg", "state", "action")} for c in cases],
        "policies": [
            {"policy": pols[0], "obs": obs.tolist()},
            {
                "policy": pols[1],
                "obs": obs[:5].tolist(),
                "driver": {"cfg": cfg_dict(dsim.cfg), "state": export_state(dsim), "policies": pols, "mute": [False, True]},
            },
        ],
    }
    (tmp / "cases.json").write_text(json.dumps(payload))
    res = subprocess.run([NODE, str(RUNNER), str(tmp / "cases.json")], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)
    from smartcrossing.policy import forward

    expect_pol = [forward(params[0], obs, sizes, scale), forward(params[1], obs[:5], sizes, scale)]
    return {"cases": cases, "js": js, "expect_pol": expect_pol, "expect_drv": expect_drv, "dsim": dsim}


def test_defaults_match_simconfig(parity):
    py = cfg_dict(SimConfig())
    js = parity["js"]["defaults"]
    assert set(js) == set(py)
    for k, v in py.items():
        if isinstance(v, (list, tuple)):
            np.testing.assert_allclose(js[k], v, atol=1e-12)
        else:
            assert js[k] == v, k


def test_case_coverage(parity):
    """The case set must actually exercise the interesting branches."""
    tot = {k: 0 for k in ("exits", "wrong_exits", "crashes", "offroad", "timeouts", "spawned")}
    for c in parity["cases"]:
        b, a = c["state"]["stats"], c["expect"]["state"]["stats"]
        for k in tot:
            tot[k] += a[k] - b[k]
    assert all(v > 0 for v in tot.values()), tot
    assert len(parity["cases"]) >= 40


def test_observe_parity(parity):
    for c, j in zip(parity["cases"], parity["js"]["cases"]):
        np.testing.assert_allclose(np.array(j["obs"]), c["expect"]["obs"], atol=TOL, err_msg=c["name"])


def test_step_parity(parity):
    for c, j in zip(parity["cases"], parity["js"]["cases"]):
        e, s, name = c["expect"]["state"], j["state"], c["name"]
        assert abs(j["reward"] - c["expect"]["reward"]) < TOL, name
        np.testing.assert_allclose(j["car_reward"], c["expect"]["car_reward"], atol=TOL, err_msg=name)
        assert j["done_kind"] == c["expect"]["done_kind"], name
        assert j["done_uid"] == c["expect"]["done_uid"], name
        assert abs(j["queue_cost"] - c["expect"]["queue_cost"]) < TOL, name
        assert s["alive"] == e["alive"], name  # removals and spawns
        alive = np.array(e["alive"])
        for k in ("x", "y", "th", "v", "age", "prev_dist", "kappa"):
            np.testing.assert_allclose(np.array(s[k]), np.array(e[k]), atol=TOL, err_msg=f"{name}:{k}")
        np.testing.assert_allclose(np.array(s["sig"]), np.array(e["sig"]), atol=TOL, err_msg=f"{name}:sig")
        for k in ("arm", "man", "uid", "group"):
            assert np.array_equal(np.array(s[k])[alive], np.array(e[k])[alive]), f"{name}:{k}"
        assert s["next_uid"] == e["next_uid"], name
        assert abs(s["t"] - e["t"]) < TOL, name
        assert len(s["queues"]) == 4
        for qs, qe in zip(s["queues"], e["queues"]):
            assert len(qs) == len(qe), name
            if qe:
                np.testing.assert_allclose(np.array(qs), np.array(qe), atol=TOL, err_msg=name)
        for k, v in e["stats"].items():
            assert s["stats"][k] == v, f"{name}:{k}"
        tr = np.array([r[1:3] for r in s["stats"]["travel"]]).reshape(-1, 2)
        np.testing.assert_allclose(tr, c["expect"]["travel"], atol=TOL, err_msg=f"{name}:travel")
        ev_e, ev_s = c["expect"]["events"], j["events"]
        assert [k for k, *_ in ev_s] == [k for k, *_ in ev_e], name
        if ev_e:
            np.testing.assert_allclose([p for _, *p in ev_s], [p for _, *p in ev_e], atol=TOL, err_msg=name)


def test_policy_parity(parity):
    js = parity["js"]["policies"]
    for exp, got in zip(parity["expect_pol"], js):
        np.testing.assert_allclose(np.array(got["out"]), exp, atol=TOL)
    alive = parity["dsim"].alive[0]
    acted = np.array(js[1]["acted"])
    np.testing.assert_allclose(acted[alive], parity["expect_drv"][alive], atol=TOL)
    muted = alive & (parity["dsim"].group[0] == 1)
    assert muted.any() and np.all(acted[muted, 2:] == 0)
