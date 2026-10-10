"""Emergence analysis of a trained policy: signal interventions, signal meaning, driving side.

  poetry run python -m smartcrossing.analyze <ckpt.npz> [--demand 0.15] [--envs 16] [--seconds 120]

Prints one JSON object:
- bench: cli-bench metrics with the received signals normal / zeroed / shuffled across cars
  (only meaningful for policies trained with the signal on);
- signal: mutual information (bits, plug-in estimate on 3x3 lamp bins) and correlations
  between the car's own lamp output and its manoeuvre, braking, speed and position;
- side: signed lateral offset to the right of the travel direction (m) on the arms,
  approaching (entry arm) vs leaving (target arm), overall and per world arm.
"""

import argparse
import json

import numpy as np

from .metrics import summarize
from .policy import PolicyDriver
from .sim import K_NEIGHBOURS, OBS_DIM, Sim, SimConfig, ARM_U

MODES = ("normal", "zero", "shuffle")


def _intervene(sim: Sim, mode: str, rng):
    """Wrap sim.observe so the lamp channels of the neighbour block are zeroed or
    shuffled across all visible neighbours of the batch (breaks sender <-> lamp link)."""
    if mode == "normal":
        return
    orig = sim.observe
    S = sim.cfg.n_signal
    blk = 6 + S
    cols = np.array([OBS_DIM + k * blk + 6 + q for k in range(K_NEIGHBOURS) for q in range(S)]).reshape(K_NEIGHBOURS, S)
    pres = np.array([OBS_DIM + k * blk + 5 for k in range(K_NEIGHBOURS)])

    def observe():
        obs = orig()
        if mode == "zero":
            obs[..., cols] = 0.0
        else:
            flat = obs.reshape(-1, obs.shape[-1])
            lamps = []
            for k in range(K_NEIGHBOURS):
                rows = np.nonzero(flat[:, pres[k]] > 0.5)[0]
                lamps.append((k, rows, flat[rows][:, cols[k]]))
            pool = np.concatenate([lv for _, _, lv in lamps]) if lamps else np.zeros((0, S))
            pool = pool[rng.permutation(len(pool))]
            i = 0
            for k, rows, lv in lamps:
                flat[np.ix_(rows, cols[k])] = pool[i : i + len(rows)]
                i += len(rows)
            obs = flat.reshape(obs.shape)
        return obs

    sim.observe = observe


def run(ckpt, demand, envs, seconds, seed=1000, mode="normal", record=False):
    sim = Sim(SimConfig(demand=demand), envs, seed=seed)
    drv = PolicyDriver.load(ckpt, sim)
    _intervene(sim, mode, np.random.default_rng(seed + 7))
    rec = []
    for _ in range(int(seconds / sim.cfg.dt)):
        act = drv.act()
        if record:
            m = sim.alive
            rec.append(np.column_stack([sim.arm[m], sim.man[m], sim.x[m], sim.y[m], sim.v[m], act[m][:, 0],
                                        act[m][:, 2:] if act.shape[-1] > 2 else np.zeros((m.sum(), 0))]))
        sim.step(act)
    return summarize(sim, seconds), (np.concatenate(rec) if record else None)


def mutual_info(a, b):
    """Plug-in MI (bits) between two discrete label arrays."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    j = np.zeros((len(ua), len(ub)))
    np.add.at(j, (ia, ib), 1)
    j /= j.sum()
    pa, pb = j.sum(1, keepdims=True), j.sum(0, keepdims=True)
    nz = j > 0
    return float((j[nz] * np.log2(j[nz] / (pa @ pb)[nz])).sum())


def region(cfg, arm, man, x, y):
    """0 = approaching on the entry arm, 1 = inside the box, 2 = leaving on the target arm, 3 = other."""
    c0 = cfg.W + cfg.Rc
    in_box = (np.abs(x) <= c0) & (np.abs(y) <= c0)
    cur = np.argmax(np.stack([-y, x, y, -x], -1), -1)
    return np.where(in_box, 1, np.where(cur == arm, 0, np.where(cur == (arm + man) % 4, 2, 3)))


def signal_stats(R, cfg):
    arm, man, x, y, v, acc = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2], R[:, 3], R[:, 4], R[:, 5]
    sig = R[:, 6:]
    if sig.shape[1] == 0 or not sig.any():  # no signal (muted policy)
        return None
    reg = region(cfg, arm, man, x, y)
    bins = np.digitize(sig, [-1 / 3, 1 / 3])
    sym = bins @ (3 ** np.arange(sig.shape[1]))
    brake = acc < -0.05
    dist = np.hypot(x, y)
    out = {
        "lamp_mean": sig.mean(0).round(3).tolist(),
        "lamp_std": sig.std(0).round(3).tolist(),
        "H_lamp_bits": round(mutual_info(sym, sym), 3),
        "MI_manoeuvre": round(mutual_info(sym, man), 4),
        "MI_brake": round(mutual_info(sym, brake), 4),
        "MI_region": round(mutual_info(sym, reg), 4),
        "MI_manoeuvre_approach": round(mutual_info(sym[reg == 0], man[reg == 0]), 4) if (reg == 0).any() else None,
        "brake_frac": round(float(brake.mean()), 3),
        "corr": {},
        "lamp_by_manoeuvre": {},
    }
    for q in range(sig.shape[1]):
        s = sig[:, q]
        out["corr"][f"lamp{q + 1}"] = {
            "accel": round(float(np.corrcoef(s, acc)[0, 1]), 3),
            "speed": round(float(np.corrcoef(s, v)[0, 1]), 3),
            "dist_centre": round(float(np.corrcoef(s, dist)[0, 1]), 3),
        }
    for m, name in ((1, "right"), (2, "straight"), (3, "left")):
        out["lamp_by_manoeuvre"][name] = sig[(man == m) & (reg == 0)].mean(0).round(3).tolist()
    return out


def side_stats(R, cfg):
    """Lateral offset (m) to the right of the direction of travel on the arms (10..40 m out)."""
    arm, man, x, y = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2], R[:, 3]
    reg = region(cfg, arm, man, x, y)
    cur = np.argmax(np.stack([-y, x, y, -x], -1), -1)
    u = ARM_U[cur]
    along = x * u[:, 0] + y * u[:, 1]
    # travel direction: -u approaching, +u leaving; right normal of d = (d_y, -d_x)
    sgn = np.where(reg == 0, -1.0, 1.0)
    off = sgn * (x * u[:, 1] - y * u[:, 0])
    ok = (along > 10) & (along < 40)
    out = {}
    for name, r in (("approach", 0), ("leave", 2)):
        m = ok & (reg == r)
        per_arm = {int(k): round(float(off[m & (cur == k)].mean()), 2) for k in range(4) if (m & (cur == k)).any()}
        per_man = {nm: round(float(off[m & (man == k)].mean()), 2) for k, nm in ((1, "right"), (2, "straight"), (3, "left"))
                   if (m & (man == k)).any()}
        out[name] = {"mean_right_offset_m": round(float(off[m].mean()), 3), "frac_right": round(float((off[m] > 0).mean()), 3),
                     "per_arm": per_arm, "per_manoeuvre": per_man, "n": int(m.sum())}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--demand", type=float, default=0.15)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--modes", default="normal,zero,shuffle")
    a = p.parse_args(argv)
    cfg = SimConfig()
    res = {"ckpt": a.ckpt, "demand": a.demand, "bench": {}}
    for mode in a.modes.split(","):
        s, R = run(a.ckpt, a.demand, a.envs, a.seconds, mode=mode, record=mode == "normal")
        res["bench"][mode] = {k: round(s[k], 3) for k in ("throughput_per_min", "exit_ok_rate", "crash_cars_per_1000",
                                                          "crash_cars", "offroad_cars", "delay_mean_s")}
        if mode == "normal":
            res["signal"] = signal_stats(R, cfg)
            res["side"] = side_stats(R, cfg)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
