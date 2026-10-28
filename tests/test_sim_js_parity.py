"""Parity between the Python simulator/policy and the browser port in site/src/js/sim.js.

Python builds a set of states (a random-traffic trajectory plus hand-made edge
cases: near-collisions, cars at the curb, exits through right/wrong arms,
timeouts, queued cars at blocked spawn points), runs ``observe`` and one
``step`` with fixed actions and arrivals disabled (demand 0, queued cars still
spawn), and a node script does the same with sim.js. Everything must agree to 1e-6.
sim.js gets its constants from js/sim-config.js, generated here by the site build's own
``sim_config_js`` (test_generated_config checks it against sim.py).
"""

import copy
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import policy_set
from smartcrossing.policy import PolicyDriver, default_in_scale, export_json, init_params, layer_sizes, save_npz
from smartcrossing import sim as simmod
from smartcrossing.sim import LEFT, NARROW, RIGHT, STRAIGHT, Sim, SimConfig, act_dim, js_constants
from conftest import exit_spawn_scene

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "sim_js_parity_runner.js"
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
            u, man, grp = sim._qbuf[0, k, (sim._qhead[0, k] + j) % sim._qcap]
            q.append([float(u), int(man), int(grp)])
        queues.append(q)
    return {
        "alive": sim.alive[0].tolist(),
        "wreck": sim.wreck[0].tolist(),
        "wreck_left": sim.wreck_left[0].tolist(),
        "x": sim.x[0].tolist(),
        "y": sim.y[0].tolist(),
        "th": sim.th[0].tolist(),
        "v": sim.v[0].tolist(),
        "kappa": sim.kappa[0].tolist(),
        "acc": sim.acc[0].tolist(),
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
            for k in ("arrived", "spawned", "exits", "wrong_exits", "crashes", "wreck_hits", "offroad", "timeouts")
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


def push_queue(sim, k, u, man, grp=0):
    """Queue a car on arm k; u in [0, 1) picks its lateral position among the clear ones."""
    j = (sim._qhead[0, k] + sim.queue[0, k]) % sim._qcap
    sim._qbuf[0, k, j] = (u, man, grp)
    sim.queue[0, k] += 1


# ---------------------------------------------------------------- cases
def trajectory_cases(seed, cfg, n_steps=160, every=4, acc_range=(-0.6, 1.0)):
    sim = Sim(cfg, 1, seed=seed, group_probs=[0.5, 0.3, 0.2])
    rng = np.random.default_rng(seed + 100)
    N, A = cfg.n_slots, act_dim(cfg)
    steer = rng.normal(0, 0.25, N)
    cases = []
    for t in range(n_steps):
        if t % 25 == 0:
            steer = rng.normal(0, 0.25, N)
        a = np.empty((N, A))
        a[:, 0] = rng.uniform(*acc_range, N)
        a[:, 1] = steer + rng.normal(0, 0.1, N)
        a[:, 2:] = rng.uniform(-1.3, 1.3, (N, A - 2))  # some beyond [-1, 1] to test clipping
        if t % every == 0 and sim.alive.any():
            cases.append(make_case(f"traj{seed}_t{t}", sim, a))
        sim.step(a[None])
    return cases


def edge_cases(cfg, tag=""):
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
        cases.append(make_case(f"{tag}edge_{i}", sim, a))
        a2 = rng.uniform(-0.2, 0.2, (N, A))
        sim.step(a2[None])  # evolve the same state a little (no arrivals: demand 0)

    # spawning from queues: arm 0 blocked by a standing car, arm 1 clear, arm 2 blocked
    # by an oncoming car in the corridor, arm 3 clear but a fast car closing from behind
    sp = Sim(dataclasses.replace(cfg, demand=0.0), 1, seed=1)
    sp.t = 5.0
    place(sp, 0, 0.3, -L + 2.5 + 6.0, hp, 0.0)
    place(sp, 1, 0.0, L - 2.5 - 20.0, hp, 12.0, arm=0, man=STRAIGHT)  # heads into arm-2 spawn
    place(sp, 2, -(L - 2.5 - 14.0), 1.0, np.pi, 6.0, arm=1, man=STRAIGHT)  # moving towards arm-3 spawn
    for k, (u, man, grp) in enumerate([(0.5, RIGHT, 0), (0.1, LEFT, 1), (0.0, STRAIGHT, 2), (0.9, LEFT, 0)]):
        push_queue(sp, k, u, man, grp)
        push_queue(sp, k, 1.0 - u - 1e-9, STRAIGHT, 1)
    a = np.zeros((N, A))
    cases.append(make_case(f"{tag}spawn", sp, a))
    sp2 = copy.deepcopy(sp)
    sp2.alive[0] = False
    sp2.x[0], sp2.y[0] = 1e4, 1e4
    cases.append(make_case(f"{tag}spawn_all_clear", sp2, a))
    # free-position spawning: long queues on every arm (several cars may enter side by side
    # in one step), u at the ends of [0, 1), a standing car blocking part of the arm-0 width,
    # a car moving away in arm 1, and the same with lane-disciplined arrivals (spawn_right)
    for sr in (False, True):
        mp = Sim(dataclasses.replace(cfg, demand=0.0, spawn_right=sr), 1, seed=3)
        mp.t = 7.0
        place(mp, 0, 4.0, -L + 2.5 + 16.0, hp, 0.0)
        place(mp, 1, -(L - 2.5 - 9.0), -1.5, 0.0, 9.0, arm=3, man=STRAIGHT)
        for k in range(4):
            for j, uu in enumerate([0.0, 0.999999, 0.5, 0.25, 0.75][: 3 + k % 3]):
                push_queue(mp, k, uu, [RIGHT, STRAIGHT, LEFT][(j + k) % 3], j % 3)
        cases.append(make_case(f"{tag}spawn_multi{'_right' if sr else ''}", mp, a))
        # only two free slots: the release stops when the slots run out
        full = copy.deepcopy(mp)
        for i, n in enumerate(range(2, N - 2)):  # standing cars mid-arm, out of spawn reach
            k, lat = i % 4, (-4.5, 0.0, 4.5)[(i // 4) % 3]
            c, s_ = np.rint(np.cos(k * hp)), np.rint(np.sin(k * hp))
            d = 20.0 + 0.37 * i  # distinct distances (no neighbour-order ties)
            place(full, n, c * lat + s_ * d, s_ * lat - c * d, hp + k * hp, 0.0, arm=k)
        cases.append(make_case(f"{tag}spawn_slots{'_right' if sr else ''}", full, a))
    # spawn_exit_static: a car leaving along arm 0 outside the spawn corridor (blocks only under the old rule,
    # i.e. in the legacy_ cases), and the same car standing heading in (blocks under both rules)
    for sign, v in ((-1.0, 8.0), (1.0, 0.0)):
        cases.append(make_case(f"{tag}spawn_exiting{'_in' if sign > 0 else ''}", exit_spawn_scene(cfg, sign, v), a))
    # grip limit: fast cars steering hard (curvature clipped to a_lat_max / v^2), a slow one
    # (kappa_max wins), and a head-on crash pair at speed (impact-speed penalty)
    gp = Sim(dataclasses.replace(cfg, demand=0.0), 1, seed=2)
    place(gp, 0, 0.5, -30.0, hp, 14.0)
    gp.kappa[0, 0] = 0.15  # above the grip limit at 14 m/s: clipped in the step
    place(gp, 1, -1.0, -10.0, hp, 2.0)
    place(gp, 2, 1.0, 20.0, -hp, 9.0, arm=2)
    place(gp, 3, 1.3, 15.8, hp, 9.0, arm=0)  # head-on with car 2
    place(gp, 4, -W + 1.0, -20.0, hp + 0.3, 12.0)  # into the curb at speed
    place(gp, 5, -2.0, -35.0, hp, -1.5)  # reversing, steering left
    place(gp, 6, 2.0, -38.0, hp, 0.0)  # standing, pedal back -> starts reversing
    place(gp, 7, 4.0, -38.0, hp, 0.05)  # nearly stopped, braking: stops at 0 (no sign change)
    place(gp, 8, -4.0, -40.0, hp, -0.1)  # reversing slowly, pedal forward: stops at 0
    gp.acc[0, :5] = [2.0, -7.0, 0.5, -3.0, 0.0]
    a = np.zeros((N, A))
    a[:9, 1] = [1.0, 1.0, -1.0, 0.3, 0.8, 0.7, -0.5, 0.0, 0.2]
    a[:9, 0] = [0.5, -0.4, 1.0, -1.0, 0.2, -0.6, -1.0, -1.0, 1.0]
    cases.append(make_case(f"{tag}grip", gp, a))
    if cfg.wreck_time > 0:
        cases += wreck_cases(cfg, tag)
    return cases


def wreck_cases(cfg, tag=""):
    """Wrecks: one expiring this step, one fresh; a car about to hit a wreck, one watching it
    (hazard feature), a queued car whose spawn spot is blocked by a wreck, a braking neighbour
    right at the brake-light threshold, and a crash pair that becomes new wrecks."""
    N, A = cfg.n_slots, act_dim(cfg)
    L, hp = cfg.L, np.pi / 2
    ws = Sim(dataclasses.replace(cfg, demand=0.0), 1, seed=4)
    ws.t = 3.0
    place(ws, 0, 0.0, 0.0, hp, 0.0)  # wreck, expires now (its slot is free for the queue this step)
    place(ws, 1, 3.0, -20.0, 0.3, 0.0)  # fresh wreck
    place(ws, 2, 3.0, -23.4, hp, 8.0)  # drives into wreck 1 within the step
    place(ws, 3, -4.0, -15.0, hp, 2.0)  # sees wreck 1 (hazard) and car 4 (brake light)
    place(ws, 4, -4.0, -8.0, hp, 6.0)  # brakes just beyond the threshold
    place(ws, 5, -3.0, -L + 2.5 + 4.0, hp, 0.0)  # wreck at the entry of arm 0
    place(ws, 6, 5.0, 20.0, -hp, 7.0, arm=2)  # head-on with car 7: new wrecks
    place(ws, 7, 5.2, 15.6, hp, 7.0)
    ws.acc[0, 4] = simmod.BRAKE_ACC - 1e-3
    ws.acc[0, 3] = simmod.BRAKE_ACC + 1e-3
    ws._make_wreck(np.isin(np.arange(N), [0, 1, 5])[None])
    ws.wreck_left[0, 0] = 1
    ws.wreck_left[0, 1] = simmod.wreck_steps(cfg)
    push_queue(ws, 0, 0.0, STRAIGHT)
    push_queue(ws, 0, 0.6, LEFT)
    a = np.zeros((N, A))
    a[4, 0] = -0.2
    cases = [make_case(f"{tag}wrecks", ws, a)]
    for i in range(2):  # evolve: the hitter is a wreck now
        ws.step(a[None])
        cases.append(make_case(f"{tag}wrecks_{i + 1}", ws, a))
    return cases


def site_build():
    """The site build's pages module (it writes the js/sim-config.js the site loads)."""
    sys.path[:0] = [str(ROOT / "site")]
    from sitebuild import pages

    return pages


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    cfg = SimConfig(n_slots=16, r_time=0.05, r_queue=0.1)
    cases = trajectory_cases(0, dataclasses.replace(cfg, demand=0.5))
    cases += trajectory_cases(1, dataclasses.replace(cfg, demand=0.35, signal_enabled=False, turn_probs=(0.5, 0.2, 0.3)))
    cases += trajectory_cases(2, dataclasses.replace(cfg, demand=0.3, neighbour_vis=55.0), n_steps=80)
    cases += trajectory_cases(3, dataclasses.replace(cfg, demand=0.3, speed_control=True), n_steps=80)
    # the old 7 m crossing without grip limit, and full-width arrivals with a strong grip limit
    cases += trajectory_cases(4, dataclasses.replace(cfg, demand=0.4, **NARROW), n_steps=80)
    cases += trajectory_cases(5, dataclasses.replace(cfg, demand=0.4, spawn_right=False, a_lat_max=1.5), n_steps=80)
    # mostly braking: cars stop and use the reverse gear
    cases += trajectory_cases(6, dataclasses.replace(cfg, demand=0.4), n_steps=120, acc_range=(-1.0, 0.4))
    # short wrecks (many appear and vanish within the trajectory)
    cases += trajectory_cases(7, dataclasses.replace(cfg, demand=0.5, wreck_time=0.8), n_steps=120)
    # the old checkpoints' interface and world: two lamps, acceleration feature, full-width spawning, no wrecks
    legacy = dict(n_signal=2, brake_light=False, spawn_right=False, wreck_time=0.0, spawn_exit_static=False)
    cases += trajectory_cases(8, dataclasses.replace(cfg, demand=0.4, **legacy), n_steps=80)
    # more neighbour slots (set-encoder interface)
    cfg8 = dataclasses.replace(cfg, demand=0.6, n_neighbours=8)
    k8 = trajectory_cases(9, cfg8, n_steps=80)
    cases += k8
    cases += edge_cases(cfg)
    cases += edge_cases(dataclasses.replace(cfg, **NARROW), tag="narrow_")
    cases += edge_cases(dataclasses.replace(cfg, **legacy), tag="legacy_")

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
    # (short wrecks: with the long default wreck time the slots of this small world fill up with wrecks)
    dsim = Sim(dataclasses.replace(cfg, demand=0.6, wreck_time=0.8), 1, seed=3, group_probs=[0.5, 0.5])
    for _ in range(120):
        dsim.step(np.tile([0.4, 0.0, 0.5, -0.5][: act_dim(cfg)], (cfg.n_slots, 1))[None])
    assert len(set(dsim.group[0][dsim.alive[0]])) == 2
    driver = PolicyDriver(dsim, params, sizes, mute=[False, True], in_scale=scale)
    expect_drv = driver.act()[0]

    # set-encoder policies (policy_set.py), one per pooling, on K = 8 observations; a driver mixing
    # a set brain (group 0) and an MLP of the same interface (group 1, muted)
    scale8 = default_in_scale(cfg8)
    sizes8 = layer_sizes(cfg8)
    obs8 = np.concatenate([c["expect"]["obs"] for c in k8])
    obs8 = np.concatenate([obs8[np.abs(obs8).sum(1) > 0], np.zeros((1, obs8.shape[1]))])
    set_pols, expect_set = [], []
    for i, pool in enumerate(policy_set.POOLS):
        arch = policy_set.make_arch(cfg8, hidden=(16, 12), phi=(10, 8), pool=pool, heads=2, dk=4, dv=3)
        p = policy_set.init_params(arch, 10 + i, out_scale=1.0)
        save_npz(tmp / f"set{i}.npz", p, sizes8, cfg8, in_scale=scale8, arch=arch)
        set_pols.append(export_json(tmp / f"set{i}.npz"))
        expect_set.append(np.tanh(policy_set.forward(p, arch, obs8 * scale8)))
    p_mlp8 = init_params(sizes8, 7, out_scale=1.0)
    save_npz(tmp / "mlp8.npz", p_mlp8, sizes8, cfg8, in_scale=scale8)
    dsim8 = Sim(cfg8, 1, seed=4, group_probs=[0.5, 0.5])
    for _ in range(120):
        dsim8.step(np.tile([0.4, 0.0, 0.5][: act_dim(cfg8)], (cfg8.n_slots, 1))[None])
    drv8 = PolicyDriver.load(f"{tmp / 'set0.npz'}+{tmp / 'mlp8.npz'}", dsim8, mute=[False, True])
    expect_drv8 = drv8.act()[0]
    pols8 = [set_pols[0], export_json(tmp / "mlp8.npz")]

    # a memory brain (policy_set mem) over a multi-step trajectory with slot reuse: Python's PolicyDriver drives,
    # JS gets each step's state (setState) and keeps its own memory state on its sim
    cfgm = dataclasses.replace(cfg8, demand=0.8, wreck_time=0.3)
    arch_m = policy_set.make_arch(cfgm, hidden=(16, 12), phi=(10, 8), heads=2, dk=4, dv=3, mem=5)
    pm = policy_set.init_params(arch_m, 21, out_scale=1.0)
    Pm = policy_set.unpack(pm, arch_m)
    for name in (policy_set.MEM_OUT, "gru.bx", "gru.bh"):
        Pm[name][...] += np.random.default_rng(22).normal(0, 0.7, Pm[name].shape)
    save_npz(tmp / "mem.npz", pm, sizes8, cfgm, in_scale=scale8, arch=arch_m)
    msim = Sim(cfgm, 1, seed=6)
    mdrv = PolicyDriver.load(tmp / "mem.npz", msim)
    mem_states, mem_expect = [], {"act": [], "h": [], "alive": [], "uid": []}
    for _ in range(100):
        mem_states.append(export_state(msim))
        a = mdrv.act()
        for k, v in (("act", a[0]), ("h", mdrv.h[0][0]), ("alive", msim.alive[0]), ("uid", msim.uid[0])):
            mem_expect[k].append(v.copy())
        msim.step(a)

    payload = {
        "memtraj": {"cfg": cfg_dict(msim.cfg), "policy": export_json(tmp / "mem.npz"), "states": mem_states},
        "cases": [{k: c[k] for k in ("name", "cfg", "state", "action")} for c in cases],
        "policies": [
            {"policy": pols[0], "obs": obs.tolist()},
            {
                "policy": pols[1],
                "obs": obs[:5].tolist(),
                "driver": {"cfg": cfg_dict(dsim.cfg), "state": export_state(dsim), "policies": pols, "mute": [False, True]},
            },
            *[{"policy": sp, "obs": obs8.tolist()} for sp in set_pols[1:]],
            {
                "policy": set_pols[0],
                "obs": obs8.tolist(),
                "driver": {"cfg": cfg_dict(dsim8.cfg), "state": export_state(dsim8), "policies": pols8,
                           "mute": [False, True]},
            },
        ],
    }
    (tmp / "cases.json").write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    (tmp / "sim-config.js").write_text(site_build().sim_config_js(), encoding="utf-8", newline="\n")
    res = subprocess.run([NODE, str(RUNNER), str(tmp / "cases.json"), str(tmp / "sim-config.js")],
                         capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)
    from smartcrossing.policy import forward

    expect_pol = [forward(params[0], obs, sizes, scale), forward(params[1], obs[:5], sizes, scale)]
    expect_pol += [*expect_set[1:], expect_set[0]]
    return {"cases": cases, "js": js, "expect_pol": expect_pol, "expect_drv": expect_drv, "dsim": dsim,
            "expect_drv8": expect_drv8, "dsim8": dsim8, "mem": {k: np.array(v) for k, v in mem_expect.items()}}


def test_defaults_match_simconfig(parity):
    py = cfg_dict(SimConfig())
    js = parity["js"]["defaults"]
    assert set(js) == set(py)
    for k, v in py.items():
        if isinstance(v, (list, tuple)):
            np.testing.assert_allclose(js[k], v, atol=1e-12)
        else:
            assert js[k] == v, k


def test_generated_config(parity):
    """js/sim-config.js carries sim.py's constants, and sim.js uses them."""
    cfg = parity["js"]["config"]
    assert cfg == json.loads(json.dumps(js_constants()))
    np.testing.assert_array_equal(cfg["RAY_ANGLES"], simmod.RAY_ANGLES)
    np.testing.assert_array_equal(cfg["DISC_OFFSETS"], simmod.DISC_OFFSETS)
    np.testing.assert_array_equal(cfg["ARM_U"], simmod.ARM_U)
    ex = parity["js"]["exported"]
    for k in ("RIGHT", "STRAIGHT", "LEFT", "DISC_R", "K_NEIGHBOURS", "NEIGHBOUR_RANGE", "OBS_DIM", "NB_FEATURES",
              "NB_FEATURES_LEGACY", "BRAKE_ACC", "POS_SCALE", "GOAL_SCALE", "RAY_RANGE"):
        assert ex[k] == getattr(simmod, k), k
    assert ex["DONE"] == {name: i for i, name in enumerate(simmod.DONE_NAMES)}
    for k in ("SPAWN_LAT_MARGIN", "SPAWN_BACK", "SPAWN_CURB_GAP", "SPAWN_RIGHT_MIN", "SPAWN_LAT_STEP",
              "SPAWN_MAX_PER_STEP", "SPEED_TAU", "N_RAYS", "EXIT_COS",
              "LANE_WIDTH", "OBS_RAYS"):
        assert cfg[k] == getattr(simmod, k), k


def test_case_coverage(parity):
    """The case set must actually exercise the interesting branches."""
    tot = {k: 0 for k in ("exits", "wrong_exits", "crashes", "wreck_hits", "offroad", "timeouts", "spawned")}
    for c in parity["cases"]:
        b, a = c["state"]["stats"], c["expect"]["state"]["stats"]
        for k in tot:
            tot[k] += a[k] - b[k]
    assert all(v > 0 for v in tot.values()), tot
    assert len(parity["cases"]) >= 40
    assert any(c["cfg"]["W"] == 3.5 for c in parity["cases"]) and any(c["cfg"]["W"] == 7.0 for c in parity["cases"])
    # wrecks appear, block and vanish; both observation interfaces and lamp counts are covered
    wrecks = [(sum(c["state"]["wreck"]), sum(c["expect"]["state"]["wreck"])) for c in parity["cases"]]
    assert any(b < a for b, a in wrecks) and any(b > a for b, a in wrecks)
    assert {(c["cfg"]["n_signal"], c["cfg"]["brake_light"]) for c in parity["cases"]} == {(1, True), (2, False)}
    brake = [c for c in parity["cases"] if c["cfg"]["brake_light"]]
    lit = np.concatenate([np.asarray(c["expect"]["obs"])[:, simmod.OBS_DIM + 7::simmod.NB_FEATURES + 1].ravel() for c in brake])
    haz = np.concatenate([np.asarray(c["expect"]["obs"])[:, simmod.OBS_DIM + 8::simmod.NB_FEATURES + 1].ravel() for c in brake])
    assert (lit == 1).any() and (haz == 1).any()
    # spawn_exit_static: the exiting car blocks the spawn only under the old rule (legacy_ cases)
    spawned = {c["name"]: c["expect"]["state"]["stats"]["spawned"] - c["state"]["stats"]["spawned"] for c in parity["cases"]}
    assert spawned["spawn_exiting"] == 1 and spawned["legacy_spawn_exiting"] == 0
    assert spawned["spawn_exiting_in"] == spawned["legacy_spawn_exiting_in"] == 0


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
        assert s["wreck"] == e["wreck"] and s["wreck_left"] == e["wreck_left"], name
        alive = np.array(e["alive"])
        for k in ("x", "y", "th", "v", "age", "prev_dist", "kappa", "acc"):
            np.testing.assert_allclose(np.array(s[k]), np.array(e[k]), atol=TOL, err_msg=f"{name}:{k}")
        np.testing.assert_allclose(np.array(s["sig"]), np.array(e["sig"]), atol=TOL, err_msg=f"{name}:sig")
        alive = alive | np.array(e["wreck"])
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


def test_set_policy_parity(parity):
    """Set-encoder forward (all poolings) and a mixed set / MLP driver on K = 8 observations."""
    js = parity["js"]["policies"]
    assert len(js) == 2 + len(policy_set.POOLS) == len(parity["expect_pol"])
    alive = parity["dsim8"].alive[0]
    acted = np.array(js[-1]["acted"])
    groups = parity["dsim8"].group[0][alive]
    assert alive.sum() >= 4 and len(set(groups)) == 2
    np.testing.assert_allclose(acted[alive], parity["expect_drv8"][alive], atol=TOL)


def test_memory_policy_trajectory_parity(parity):
    """Memory brain over 100 steps: JS actions and memory states equal PolicyDriver's for every driving car, incl.
    slots that got a new car in the same step the old one left (state back to 0); JS's read-only memIn + forward
    (the selected car's next action on the site) equals what act() then does."""
    js, e = parity["js"]["memtraj"], parity["mem"]
    al = e["alive"]
    assert al.sum() > 200
    reuse = (al[1:] & al[:-1] & (e["uid"][1:] != e["uid"][:-1])).sum()
    assert reuse > 0
    np.testing.assert_allclose(np.array(js["act"])[al], e["act"][al], atol=TOL)
    np.testing.assert_allclose(np.array(js["h"])[al], e["h"][al], atol=TOL)
    assert np.abs(e["h"][al]).max() > 0.05 and js["peek_diff"] < 1e-12
