"""Emergence analysis of a trained policy: signal interventions, signal meaning, driving side.

  poetry run python -m smartcrossing.analyze <ckpt.npz> [--demand 0.15] [--envs 16] [--seconds 120]

Prints one JSON object:
- bench: cli-bench metrics with the received signals normal / zeroed / shuffled across cars
  (only meaningful for policies trained with the signal on);
- signal: mutual information (bits, plug-in estimate on 3x3 lamp bins) and correlations
  between the car's own lamp output and its manoeuvre, braking, speed and position;
- side: signed lateral offset to the right of the travel direction (m) on the arms,
  approaching (entry arm) vs leaving (target arm), overall and per world arm;
- lanes: histogram of that lateral offset (approach / leave, 0.5 m bins) and a bimodality
  measure (do cars form lanes on a road wide enough for two cars per direction?);
- wobble: weaving on the straight arm segments (mean |yaw rate|, curvature sign changes
  per second, heading std); computed for any driver (also baselines via --driver).
"""

import argparse
import json

import numpy as np

from .metrics import summarize
from .policy import PolicyDriver
from .sim import ARM_U, K_NEIGHBOURS, OBS_DIM, Sim, SimConfig, nb_features

MODES = ("normal", "zero", "shuffle")


def _intervene(sim: Sim, mode: str, rng):
    """Wrap sim.observe so the lamp channels of the neighbour block are zeroed or
    shuffled across all visible neighbours of the batch (breaks sender <-> lamp link)."""
    if mode == "normal":
        return
    orig = sim.observe
    S = sim.cfg.n_signal
    nbf = nb_features(sim.cfg)
    blk = nbf + S
    cols = np.array([OBS_DIM + k * blk + nbf + q for k in range(K_NEIGHBOURS) for q in range(S)]).reshape(K_NEIGHBOURS, S)
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


def run(ckpt, demand, envs, seconds, seed=1000, mode="normal", record=False, cfg_over=None, traj=None):
    """ckpt: a policy .npz (or 'light' / 'reservation' / 'none' for the baselines).
    record: also return R (rows arm, man, x, y, v, accel action, lamps...). traj: a dict
    that is filled with per-car-step arrays (env, uid, step, arm, man, x, y, v, th, kappa)
    for the wobble analysis. Returns (summary, R, sim)."""
    sim = Sim(SimConfig(demand=demand, **(cfg_over or {})), envs, seed=seed)
    drv = _make_driver(ckpt, sim)
    _intervene(sim, mode, np.random.default_rng(seed + 7))
    rec, tr = [], []
    for k in range(int(seconds / sim.cfg.dt)):
        act = drv.act()
        if record:
            m = sim.alive
            rec.append(np.column_stack([sim.arm[m], sim.man[m], sim.x[m], sim.y[m], sim.v[m], act[m][:, 0],
                                        act[m][:, 2:] if act.shape[-1] > 2 else np.zeros((m.sum(), 0))]))
        sim.step(act)
        if traj is not None:  # state after the step (actual curvature incl. rate and grip limits)
            bi, ni = np.nonzero(sim.alive)
            tr.append(np.column_stack([bi, sim.uid[bi, ni], np.full(len(bi), k), sim.arm[bi, ni], sim.man[bi, ni],
                                       sim.x[bi, ni], sim.y[bi, ni], sim.v[bi, ni], sim.th[bi, ni], sim.kappa[bi, ni]]))
    if traj is not None:
        T = np.concatenate(tr) if tr else np.zeros((0, 10))
        for j, key in enumerate(("env", "uid", "step", "arm", "man", "x", "y", "v", "th", "kappa")):
            traj[key] = T[:, j]
    return summarize(sim, seconds), (np.concatenate(rec) if record else None), sim


KIND_NAMES = ("exit", "wrong_exit", "crash", "offroad", "timeout")


def _make_driver(ckpt, sim):
    if ckpt in ("light", "reservation", "none"):
        from .baselines import LaneDriver

        return LaneDriver(sim, ckpt)
    return PolicyDriver.load(ckpt, sim)


def outcome_counts(ckpt, demand, envs=16, seconds=120.0, seed=1000, cfg_over=None, solo=False):
    """Outcome of every car that finished (exit / wrong_exit / crash / offroad / timeout),
    per manoeuvre. solo=True: one car slot per env (every car drives alone; the queue is
    always full, demand is irrelevant). Returns ({manoeuvre: {kind: count}}, summary)."""
    over = dict(cfg_over or {})
    if solo:
        over.update(n_slots=1)
        demand = max(demand, 0.5)
    sim = Sim(SimConfig(demand=demand, **over), envs, seed=seed)
    drv = _make_driver(ckpt, sim)
    cnt = {m: dict.fromkeys(KIND_NAMES, 0) for m in ("right", "straight", "left")}
    names = {1: "right", 2: "straight", 3: "left"}
    for _ in range(int(seconds / sim.cfg.dt)):
        act = drv.act()
        man = sim.man.copy()  # slots refilled in this step get a new manoeuvre after removal
        sim.step(act)
        b, n = np.nonzero(sim.done_kind > 0)
        for m, k in zip(man[b, n], sim.done_kind[b, n]):
            cnt[names[int(m)]][KIND_NAMES[int(k) - 1]] += 1
    return cnt, summarize(sim, seconds)


def failure_breakdown(ckpt, demands=(0.02, 0.05, 0.15, 0.3), envs=16, seconds=120.0, seed=1000, cfg_over=None):
    """Rates (per finished car) of exit_ok / crash (car-car) / offroad / timeout / wrong exit,
    overall and per manoeuvre, for a solo car and for traffic at each demand."""
    out = {}
    for label, d, solo in [("solo", 0.5, True)] + [(f"d{d:g}", d, False) for d in demands]:
        cnt, summ = outcome_counts(ckpt, d, envs, seconds, seed, cfg_over, solo)
        rows = {}
        tot = dict.fromkeys(KIND_NAMES, 0)
        for m, c in cnt.items():
            n = sum(c.values())
            rows[m] = {"n": n, **{k: round(v / max(n, 1), 4) for k, v in c.items()}}
            for k in KIND_NAMES:
                tot[k] += c[k]
        n = sum(tot.values())
        rows["all"] = {"n": n, **{k: round(v / max(n, 1), 4) for k, v in tot.items()},
                       "throughput_per_min": round(summ["throughput_per_min"], 2), "delay_mean_s": round(summ["delay_mean_s"], 2)}
        out[label] = rows
    return out


def breakdown_table(fb) -> str:
    """Markdown table of failure_breakdown(): rows = condition, columns = exit/crash/offroad (%)
    overall and exit/offroad per manoeuvre."""
    lines = ["| condition | n | exit ok | crash | offroad | timeout+wrong | R exit/off | S exit/off | L exit/off | thr/min |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    for label, r in fb.items():
        a = r["all"]
        mm = [f"{pct(r[m]['exit'])}/{pct(r[m]['offroad'])}" for m in ("right", "straight", "left")]
        lines.append(f"| {label} | {a['n']} | {pct(a['exit'])} | {pct(a['crash'])} | {pct(a['offroad'])} | "
                     f"{pct(a['timeout'] + a['wrong_exit'])} | {' | '.join(mm)} | {a['throughput_per_min']} |")
    return "\n".join(lines)


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


def _arm_offset(cfg, arm, man, x, y):
    """(region, current arm, distance out along the arm, offset to the right of travel)."""
    reg = region(cfg, arm, man, x, y)
    cur = np.argmax(np.stack([-y, x, y, -x], -1), -1)
    u = ARM_U[cur]
    along = x * u[:, 0] + y * u[:, 1]
    # travel direction: -u approaching, +u leaving; right normal of d = (d_y, -d_x)
    sgn = np.where(reg == 0, -1.0, 1.0)
    off = sgn * (x * u[:, 1] - y * u[:, 0])
    return reg, cur, along, off


def lane_stats(R, cfg, bin_m=0.5):
    """Histogram of the right-offset on the arms (between the box and 5 m before the
    spawn line), approaching and leaving. 'bimodality' = Sarle's coefficient
    (skew^2 + 1) / kurtosis: > 5/9 ~ 0.555 (the uniform value) hints at two modes.
    'modes' = local maxima of the 3-bin smoothed histogram above 40% of its peak."""
    arm, man, x, y = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2], R[:, 3]
    reg, cur, along, off = _arm_offset(cfg, arm, man, x, y)
    ok = (along > cfg.W + cfg.Rc + 3.0) & (along < cfg.L - 5.0)
    edges = np.arange(-cfg.W, cfg.W + 1e-9, bin_m)
    out = {"bin_edges_m": [round(float(e), 2) for e in edges]}
    for name, r in (("approach", 0), ("leave", 2)):
        o = off[ok & (reg == r)]
        if len(o) < 10:
            continue
        h, _ = np.histogram(o, edges)
        hs = np.convolve(h, np.ones(3) / 3, mode="same")
        pk = [int(i) for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] > 0.4 * hs.max()]
        z = (o - o.mean()) / (o.std() + 1e-9)
        bc = (np.mean(z**3) ** 2 + 1) / np.mean(z**4)
        per_arm = {}
        for k in range(4):
            ok_k = ok & (reg == r) & (cur == k)
            if ok_k.sum() > 10:
                per_arm[k] = np.histogram(off[ok_k], edges)[0].tolist()
        out[name] = {"hist": h.tolist(), "hist_per_arm": per_arm, "bimodality": round(float(bc), 3),
                     "modes_m": [round(float(0.5 * (edges[i] + edges[i + 1])), 2) for i in pk],
                     "std_m": round(float(o.std()), 3), "n": int(len(o))}
    return out


def wobble_stats(T, cfg, min_kappa=0.002):
    """Weaving on the straight parts of the arms (between the box + 5 m and the spawn /
    exit line - 5 m): mean |yaw rate| v*kappa (rad/s), sign changes per second of the
    actual curvature (ignoring |kappa| < min_kappa), heading std (deg) relative to the arm
    axis, and the mean |lateral speed| (m/s). Uses the per-car-step arrays of run(traj=...)."""
    if len(T.get("uid", [])) == 0:
        return None
    arm, man = T["arm"].astype(int), T["man"].astype(int)
    reg, cur, along, off = _arm_offset(cfg, arm, man, T["x"], T["y"])
    ok = (reg != 1) & (along > cfg.W + cfg.Rc + 5.0) & (along < cfg.L - 5.0)
    th, v, kap = T["th"], T["v"], T["kappa"]
    # heading relative to the travel direction along the arm (approach: -u, leave/other: +u)
    u = ARM_U[cur]
    d_ang = np.arctan2(np.where(reg == 0, -u[:, 1], u[:, 1]), np.where(reg == 0, -u[:, 0], u[:, 0]))
    dh = (th - d_ang + np.pi) % (2 * np.pi) - np.pi
    # sign changes per car (consecutive steps of the same car inside the segment)
    key = T["env"] * 1e7 + T["uid"]
    order = np.lexsort((T["step"], key))

    def stats(mask):
        k_o, s_o, ok_o, kap_o = key[order], T["step"][order], mask[order], kap[order]
        sg = np.where(np.abs(kap_o) < min_kappa, 0, np.sign(kap_o))
        cont = np.r_[False, (k_o[1:] == k_o[:-1]) & (s_o[1:] == s_o[:-1] + 1) & ok_o[1:] & ok_o[:-1]]
        changes, last = 0, 0.0
        for i in range(len(sg)):  # last nonzero sign carried forward inside a contiguous run
            if not cont[i]:
                last = sg[i] if ok_o[i] else 0.0
                continue
            if sg[i] != 0:
                if last != 0 and sg[i] != last:
                    changes += 1
                last = sg[i]
        seconds = mask.sum() * cfg.dt
        if seconds == 0:
            return None
        return {
            "yaw_rate_abs_mean": round(float(np.mean(np.abs(v[mask] * kap[mask]))), 4),
            "kappa_sign_changes_per_s": round(float(changes / seconds), 4),
            "heading_std_deg": round(float(np.degrees(np.sqrt(np.mean(dh[mask] ** 2)))), 3),
            "lat_speed_abs_mean": round(float(np.mean(np.abs(v[mask] * np.sin(dh[mask])))), 4),
            "car_seconds": round(float(seconds), 1),
        }

    # leaving segment: no spawn-offset merging, the cleanest wobble measure
    return {**stats(ok), "approach": stats(ok & (reg == 0)), "leave": stats(ok & (reg == 2))}


def reverse_stats(T):
    """How much the reverse gear is used: share of car-steps with v < 0, share of cars that
    ever reversed, mean / max reversing time per reversing car (s)."""
    if len(T.get("uid", [])) == 0:
        return None
    key = T["env"] * 1e7 + T["uid"]
    rev = T["v"] < -1e-6
    cars = np.unique(key)
    rcars, cnt = np.unique(key[rev], return_counts=True)
    return {"frac_car_steps": round(float(rev.mean()), 5), "frac_cars": round(len(rcars) / max(len(cars), 1), 4),
            "n_cars": int(len(cars)), "n_reversing_cars": int(len(rcars)),
            "mean_rev_s": round(float(cnt.mean() * 0.1), 2) if len(cnt) else 0.0,
            "max_rev_s": round(float(cnt.max() * 0.1), 2) if len(cnt) else 0.0}


def side_stats(R, cfg):
    """Lateral offset (m) to the right of the direction of travel on the arms (10..40 m out)."""
    arm, man, x, y = R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2], R[:, 3]
    reg, cur, along, off = _arm_offset(cfg, arm, man, x, y)
    ok = (along > cfg.W + cfg.Rc + 0.5) & (along < cfg.L - 10.0)
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
    p.add_argument("ckpt", help="policy .npz, or light / reservation / none (baselines: no signal stats)")
    p.add_argument("--demand", type=float, default=0.15)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--modes", default="normal,zero,shuffle")
    p.add_argument("--cfg", default="", help="JSON SimConfig overrides (a policy's own interface/geometry wins)")
    p.add_argument("--hist", action="store_true", help="include the lateral-offset histograms")
    p.add_argument("--breakdown", action="store_true", help="only the failure breakdown (solo + demands), as JSON + table")
    p.add_argument("--demands", default="0.02,0.05,0.15,0.3")
    a = p.parse_args(argv)
    over = json.loads(a.cfg) if a.cfg else None
    if a.breakdown:
        fb = failure_breakdown(a.ckpt, tuple(float(d) for d in a.demands.split(",")), a.envs, a.seconds, cfg_over=over)
        print(json.dumps({"ckpt": a.ckpt, "breakdown": fb}))
        print(breakdown_table(fb))
        return
    baseline = a.ckpt in ("light", "reservation", "none")
    modes = ["normal"] if baseline else a.modes.split(",")
    res = {"ckpt": a.ckpt, "demand": a.demand, "bench": {}}
    for mode in modes:
        traj = {} if mode == "normal" else None
        s, R, sim = run(a.ckpt, a.demand, a.envs, a.seconds, mode=mode, record=mode == "normal", cfg_over=over, traj=traj)
        res["bench"][mode] = {k: round(s[k], 3) for k in ("throughput_per_min", "exit_ok_rate", "crash_cars_per_1000",
                                                          "crash_cars", "offroad_cars", "delay_mean_s")}
        if mode == "normal":
            cfg = sim.cfg
            res["geometry"] = {"W": cfg.W, "Rc": cfg.Rc, "L": cfg.L, "a_lat_max": getattr(cfg, "a_lat_max", None)}
            if not baseline:
                res["signal"] = signal_stats(R, cfg)
            res["side"] = side_stats(R, cfg)
            ls = lane_stats(R, cfg)
            if not a.hist:
                ls = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("hist")} if isinstance(v, dict) else v
                      for k, v in ls.items() if k != "bin_edges_m"}
            res["lanes"] = ls
            res["wobble"] = wobble_stats(traj, cfg)
            res["reverse"] = reverse_stats(traj)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
