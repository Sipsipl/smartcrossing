"""Deadlock study (PLAN (G), measurement only): demand sweeps with per-step recording, an operational
deadlock detector, its constellation / ending / lamp analysis, action-noise and lamp interventions, and the
fundamental-diagram data (throughput vs density and vs demand). The scatter cloud of the fundamental diagram
(``cli cloud``) is fundamental_cloud.py, the reward accounting of a stuck car (``cli incentives``) incentives.py.

    poetry run python -m smartcrossing.cli deadlock --driver policy:results/round7_best.npz --driver light \
        --driver reservation --seeds 4000,5000,6000 --seconds 180 --out runs/deadlock/x.jsonl \
        [--demands 0.3,0.5] [--noise policy|0.1] [--lamp zero|shuffle] [--workers 4]
    poetry run python -m smartcrossing.cli deadlock-summarize runs/deadlock/*.jsonl \
        [--fundamental results/deadlock_fundamental.json --name round7_best=brain] [--json rows.json]

Deadlock (operational definition, ``detect``): the zone Z is the box incl. the curb rounding, Chebyshev
distance max(|x|, |y|) <= W + Rc (16 m in the default world); the conflict area proper is max(|x|, |y|) <=
W + INNER_PAD (9 m). A car is *stuck* when it is in Z and its net displacement over the last T_MIN = 5 s is
below STUCK_D = 1.5 m (standing, creeping or rocking on the spot: at high demand the brain's cars rarely stand
still but jitter at 0.3-1 m/s, so a pure speed threshold misses them). An env is deadlocked while >= N_MIN = 3
cars are stuck, >= N_INNER = 2 of them inside the conflict area; intervals closer than 1 s are merged, the
start is back-dated by T_MIN. Why: one car waiting for a gap or a queue at the box mouth (cars of one arm
waiting at a red light) never has two stuck cars inside the conflict area; two crossing / facing cars
stuck inside it for 5 s with a third one behind is a gridlock, whether or not right-turners still pass at the
corners (so no "zone outflow = 0" condition; ``zone_exits`` reports it per episode). Variants N = 2 / 4 and
T = 10 s are counted as a sensitivity check. End causes (``end_cause``): a stuck car was removed at max_age
(``timeout``) or crashed (``crash``) in the last second, else one moved off (``moved``; ``moved_rev``: reversing);
``censored`` = still running at the end.
"""

import json
import time
from pathlib import Path

import numpy as np

from .drivers import make_driver
from .jobs import append_jsonl, pool_map
from .metrics import summarize
from .protocol import ENVS, SELECTION_SEEDS, SWEEP_DEMANDS, SWEEP_SECONDS
from .sim import DONE_CRASH, DONE_OFFROAD, DONE_TIMEOUT, Sim, SimConfig

SLOW_V = 0.3  # m/s: below this a car counts as standing (jam metric)
STUCK_D = 1.5  # m: net displacement over T_MIN below this = stuck (standing, creeping or rocking on the spot)
N_MIN = 3  # stuck cars in the zone at once ...
N_INNER = 2  # ... of which this many inside the conflict area proper
T_MIN = 5.0  # s window of the displacement test
MERGE_GAP = 1.0  # s: deadlock intervals closer than this are one episode
WARMUP = 30.0  # s from the empty start before density / steady throughput are measured
BLOCKER_RANGE = 10.0  # m: a car within this distance in the +-BLOCKER_CONE ahead blocks a car
BLOCKER_CONE = np.radians(45.0)
INNER_PAD = 2.0  # a car is "inner" (in the conflict area proper) when max(|x|, |y|) <= W + INNER_PAD
LEAD_S, LAST_S = 5.0, 3.0  # lamp windows: before the deadlock began / before it ended
MAN_NAMES = {1: "right", 2: "straight", 3: "left"}


# ------------------------------------------------------------------ recording
def record_run(spec: str, cfg: SimConfig, seed: int, envs: int = ENVS, seconds: float = SWEEP_SECONDS, noise=None,
               lamp: str = "normal"):
    """Run ``spec`` and record the state BEFORE every step (alive, wreck, uid, x, y, th, v, arm, man, age),
    the emitted lamp (action[2]) and what the step removed (done_kind per slot). lamp: normal / zero / shuffle
    (analyze._intervene on the received lamps).
    Returns (rec dict of (T, B, N) arrays + queue (T, B), summarize dict, sim)."""
    from .analyze import _intervene

    sim = Sim(cfg, envs, seed=seed)
    drv = make_driver(spec, sim, noise, seed)
    _intervene(sim, lamp, np.random.default_rng(seed + 7))
    T, B, N = int(round(seconds / cfg.dt)), envs, cfg.n_slots
    f32 = lambda: np.zeros((T, B, N), np.float32)  # noqa: E731
    rec = {k: f32() for k in ("x", "y", "th", "v", "age", "lamp")}
    rec.update(alive=np.zeros((T, B, N), bool), wreck=np.zeros((T, B, N), bool),
               uid=np.zeros((T, B, N), np.int64), arm=np.zeros((T, B, N), np.int8), man=np.zeros((T, B, N), np.int8),
               done=np.zeros((T, B, N), np.int8), queue=np.zeros((T, B), np.int32))
    for i in range(T):
        for k in ("x", "y", "th", "v", "age", "alive", "wreck", "uid", "arm", "man"):
            rec[k][i] = getattr(sim, k)
        rec["queue"][i] = sim.queue.sum(1)
        act = drv.act()
        if act.shape[-1] > 2:
            rec["lamp"][i] = np.where(sim.alive, act[..., 2], 0.0)
        sim.step(act)
        rec["done"][i] = sim.done_kind
    return rec, summarize(sim, seconds), sim


# ------------------------------------------------------------------ detection
def _blocker(x, y, th, cand, i):
    """Index of the nearest candidate within BLOCKER_RANGE in the +-BLOCKER_CONE ahead of car i (-1 = none)."""
    dx, dy = x - x[i], y - y[i]
    d = np.hypot(dx, dy)
    ang = np.abs((np.arctan2(dy, dx) - th[i] + np.pi) % (2 * np.pi) - np.pi)
    ok = cand & (d < BLOCKER_RANGE) & (ang < BLOCKER_CONE)
    ok[i] = False
    return int(np.argmin(np.where(ok, d, np.inf))) if ok.any() else -1


def _relation(arm_i, arm_j) -> str:
    """Entry arm of j seen from i: same / opposite / right (j comes from i's right) / left."""
    return {0: "same", 2: "opposite", 1: "right", 3: "left"}[int(arm_j - arm_i) % 4]


def constellation(rec, cfg, i, b, slots) -> dict:
    """Geometry of the stuck cars ``slots`` of env b at step i: who blocks whom (nearest car in the cone ahead),
    cycles in that graph, manoeuvres / entry arms of the inner cars and the entry-arm relation of each car to
    its blocker."""
    x, y, th = rec["x"][i, b].astype(float), rec["y"][i, b].astype(float), rec["th"][i, b].astype(float)
    alive, wreck = rec["alive"][i, b], rec["wreck"][i, b]
    cheb = np.maximum(np.abs(x), np.abs(y))
    cand = (alive | wreck) & (cheb < cfg.W + cfg.Rc + BLOCKER_RANGE)
    nxt = {int(n): _blocker(x, y, th, cand, int(n)) for n in slots}
    involved = set(int(n) for n in slots)
    cycles, heads = [], []
    for n in involved:
        path, cur = [], n
        while cur in involved and cur not in path:
            path.append(cur)
            cur = nxt[cur]
        if cur in path:  # a cycle (each car's blocker is the next one)
            cyc = path[path.index(cur):]
            if sorted(cyc) not in [sorted(c) for c in cycles]:
                cycles.append(cyc)
        else:  # the chain ends at a car that is not stuck, a wreck or nothing
            heads.append("none" if cur == -1 else "wreck" if wreck[cur] else "mover")
    inner = [int(n) for n in slots if cheb[n] <= cfg.W + INNER_PAD]
    arm, man = rec["arm"][i, b], rec["man"][i, b]
    rel, face = [], []
    for n in involved:
        j = nxt[n]
        if j >= 0 and alive[j]:
            rel.append(_relation(arm[n], arm[j]))
            dh = abs((th[j] - th[n] + np.pi) % (2 * np.pi) - np.pi)
            face.append("head_on" if dh > np.radians(135) else "crossing" if dh > np.radians(45) else "following")
    longest = max((len(c) for c in cycles), default=0)
    kind = ("cycle" if longest >= 3 else "mutual" if longest == 2 else "wreck" if "wreck" in heads
            else "free_head" if "none" in heads else "mover")
    return {"kind": kind, "cycle_len": longest, "n_cycles": len(cycles), "heads": heads,
            "n_inner": len(inner), "inner_man": [MAN_NAMES[int(man[n])] for n in inner],
            "inner_arms": sorted({int(arm[n]) for n in inner}), "blocker_rel": rel, "blocker_face": face,
            "wreck_in_zone": bool((wreck & (cheb <= cfg.W + cfg.Rc)).any()),
            "cars": [{"slot": int(n), "arm": int(arm[n]), "man": MAN_NAMES[int(man[n])], "x": round(float(x[n]), 1),
                      "y": round(float(y[n]), 1), "th_deg": round(float(np.degrees(th[n])), 0),
                      "inner": bool(cheb[n] <= cfg.W + INNER_PAD), "blocker": nxt[int(n)]} for n in sorted(involved)]}


def stuck_mask(rec, cfg, t_min: float = T_MIN):
    """(T, B, N): car in the zone Z whose net displacement over the last t_min seconds (same car) is < STUCK_D
    (standing, creeping or rocking back and forth on the spot); and the inner flag (conflict area proper)."""
    Ts = int(round(t_min / cfg.dt))
    cheb = np.maximum(np.abs(rec["x"]), np.abs(rec["y"]))
    inZ = rec["alive"] & (cheb <= cfg.W + cfg.Rc)
    stuck = np.zeros_like(inZ)
    if Ts < len(inZ):
        same = rec["alive"][Ts:] & rec["alive"][:-Ts] & (rec["uid"][Ts:] == rec["uid"][:-Ts])
        disp = np.hypot(rec["x"][Ts:] - rec["x"][:-Ts], rec["y"][Ts:] - rec["y"][:-Ts])
        stuck[Ts:] = inZ[Ts:] & same & (disp < STUCK_D)
    return stuck, inZ, stuck & (cheb <= cfg.W + INNER_PAD)


def detect(rec, cfg, n_min: int = N_MIN, t_min: float = T_MIN, details: bool = True) -> list[dict]:
    """Deadlock episodes of a recorded run (module docstring). details: constellation + lamp analysis."""
    T, B, N = rec["x"].shape
    Ts = int(round(t_min / cfg.dt))
    gap = int(round(MERGE_GAP / cfg.dt))
    stuck, inZ, inner = stuck_mask(rec, cfg, t_min)
    cond = (stuck.sum(2) >= n_min) & (inner.sum(2) >= N_INNER)  # (T, B)
    dep = np.zeros_like(inZ)  # drive-outs of the zone between step i-1 and i
    dep[1:] = inZ[:-1] & rec["alive"][1:] & (rec["uid"][1:] == rec["uid"][:-1]) & ~inZ[1:]
    episodes = []
    for b in range(B):
        on = np.nonzero(cond[:, b])[0]
        if len(on) == 0:
            continue
        # intervals of cond, merged over gaps <= MERGE_GAP
        cuts = np.nonzero(np.diff(on) > gap + 1)[0]
        for lo, hi in zip(np.r_[on[0], on[cuts + 1]], np.r_[on[cuts], on[-1]]):
            onset, last = int(lo), int(hi)
            end = last + 1
            slots = np.nonzero(stuck[onset, b])[0]
            uids = set()
            for i in range(onset, end):
                uids.update(int(u) for u in rec["uid"][i, b, stuck[i, b]])
            ep = {"env": b, "start": max(onset - Ts, 0), "onset": onset, "end": end,
                  "uids": sorted(uids), "censored": end >= T,
                  "zone_exits": int(dep[onset:end, b].sum()),
                  "n_timeouts": int(sum((rec["done"][onset:end, b][rec["uid"][onset:end, b] == u] == DONE_TIMEOUT).any()
                                        for u in uids)),
                  "n_crashed": int(sum(np.isin(rec["done"][onset:end, b][rec["uid"][onset:end, b] == u],
                                               (DONE_CRASH, DONE_OFFROAD)).any() for u in uids))}
            ep["cause"] = "censored" if end >= T else end_cause(rec, cfg, b, last, stuck)
            if details:
                ep["geo"] = constellation(rec, cfg, onset, b, slots)
            episodes.append(ep)
    for ep in episodes:
        ep["dur_s"] = round((ep["end"] - ep["start"]) * cfg.dt, 1)
        ep["n_cars"] = len(ep["uids"])
        if details:
            ep["lamp"] = lamp_profile(rec, cfg, ep, inZ)
    episodes.sort(key=lambda e: (e["env"], e["start"]))
    for prev, ep in zip(episodes, episodes[1:]):  # chains: a new deadlock within 10 s after the previous one
        if ep["env"] == prev["env"] and ep["onset"] - prev["end"] <= int(10 / cfg.dt):
            ep["reformed"] = True
    return episodes


def end_cause(rec, cfg, b, last, stuck) -> str:
    """Why the deadlock condition stopped after step ``last``: a stuck car was removed (timeout / crash or
    off-road) in the last second, else a stuck car moved off ("moved_rev" if it was reversing)."""
    T = rec["x"].shape[0]
    members = np.nonzero(stuck[last, b])[0]
    w = int(round(1.0 / cfg.dt))
    done = rec["done"][max(last - w, 0):min(last + 2, T), b][:, members]
    if np.isin(done, (DONE_CRASH, DONE_OFFROAD)).any():
        return "crash"
    if (done == DONE_TIMEOUT).any():
        return "timeout"
    nxt = min(last + 1, T - 1)
    moved = members[~stuck[nxt, b, members] & rec["alive"][nxt, b, members]
                    & (rec["uid"][nxt, b, members] == rec["uid"][last, b, members])]
    if len(moved) and (rec["v"][max(nxt - w, 0):nxt + 1, b][:, moved] < -0.05).any():
        return "moved_rev"
    return "moved"


def lamp_profile(rec, cfg, ep, inZ) -> dict:
    """Lamps of the involved cars: mean before the stall (LEAD_S), during it, in the LAST_S before its end;
    the "goer" is the involved car that first drives out of Z after the start (others: waiters)."""
    b, s, e = ep["env"], ep["start"], ep["end"]
    T = rec["x"].shape[0]
    lead, last = int(LEAD_S / cfg.dt), int(LAST_S / cfg.dt)
    cars, out_step = [], {}
    for u in ep["uids"]:
        hit = np.nonzero(rec["uid"][s:min(e + 1, T), b] == u)
        if len(hit[0]) == 0:
            continue
        n = int(hit[1][0])
        mine = rec["uid"][:, b, n] == u
        steps = np.nonzero(mine)[0]
        left = steps[(steps > s) & ~inZ[steps, b, n] & inZ[steps - 1, b, n]]
        out_step[u] = int(left[0]) if len(left) else None
        lp = rec["lamp"][:, b, n]
        win = lambda lo, hi: lp[max(lo, 0):max(hi, 0)][mine[max(lo, 0):max(hi, 0)]]  # noqa: E731
        m = lambda a: round(float(a.mean()), 3) if len(a) else None  # noqa: E731
        cars.append({"uid": int(u), "man": MAN_NAMES[int(rec["man"][steps[0], b, n])],
                     "age_start": round(float(rec["age"][s, b, n]), 1) if mine[s] else None,
                     "lamp_lead": m(win(s - lead, s)), "lamp_during": m(win(s, e)), "lamp_last": m(win(e - last, e))})
    outs = {u: t for u, t in out_step.items() if t is not None}
    goer = min(outs, key=outs.get) if outs else None
    for c in cars:
        c["goer"] = c["uid"] == goer
    ages = [c["age_start"] for c in cars if c["age_start"] is not None]
    g = next((c for c in cars if c["goer"]), None)
    return {"cars": cars, "goer_man": g["man"] if g else None,
            "goer_oldest": (g["age_start"] == max(ages)) if g and g["age_start"] is not None and ages else None}


# ------------------------------------------------------------------ run metrics
def run_metrics(rec, cfg, summ, seconds, episodes, sensitivity=None) -> dict:
    """Bench metrics + density / steady throughput / queues / deadlock statistics of one recorded run."""
    T, B, N = rec["x"].shape
    w = int(WARMUP / cfg.dt)
    alive = rec["alive"][w:]
    cheb = np.maximum(np.abs(rec["x"][w:]), np.abs(rec["y"][w:]))
    finished = int((rec["done"] > 0).sum())
    exits_steady = int((rec["done"][w:] == 1).sum())
    steady_min = (T - w) * cfg.dt / 60 * B
    hours = T * cfg.dt / 3600 * B
    in_dl = sum(min(e["end"], T) - e["start"] for e in episodes)
    occ = (rec["alive"] | rec["wreck"]).sum(2)
    lampZ = rec["lamp"][rec["alive"] & (np.maximum(np.abs(rec["x"]), np.abs(rec["y"])) <= cfg.W + cfg.Rc)]
    chebA = np.maximum(np.abs(rec["x"]), np.abs(rec["y"]))
    to = rec["done"] == DONE_TIMEOUT
    cr = np.isin(rec["done"], (DONE_CRASH, DONE_OFFROAD))
    # "jam": >= N_MIN cars standing in the zone for >= T_MIN (drive-outs allowed, unlike a deadlock)
    slowZ = rec["alive"] & (chebA <= cfg.W + cfg.Rc) & (np.abs(rec["v"]) < SLOW_V)
    run = np.zeros((B, N), np.int64)
    jam = np.zeros((T, B), bool)
    for i in range(T):
        same = rec["alive"][i] & (rec["uid"][i] == rec["uid"][i - 1]) if i else np.zeros((B, N), bool)
        run = np.where(slowZ[i] & same, run + 1, slowZ[i].astype(np.int64))
        jam[i] = (run >= int(round(T_MIN / cfg.dt))).sum(1) >= N_MIN
    out = {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()},
           "timeouts_in_zone_frac": round(float((to & (chebA <= cfg.W + cfg.Rc)).sum() / max(to.sum(), 1)), 3),
           "crashes_in_zone_frac": round(float((cr & (chebA <= cfg.W + cfg.Rc)).sum() / max(cr.sum(), 1)), 3),
           "jam_time_frac": round(float(jam[w:].mean()), 4),
           "timeouts_per_1000": round(1000 * summ["timeouts"] / finished, 2) if finished else None,
           "throughput_steady_per_min": round(exits_steady / steady_min, 3),
           "cars_map_mean": round(float(alive.sum(2).mean()), 3),
           "cars_box_mean": round(float((alive & (cheb <= cfg.W + cfg.Rc)).sum(2).mean()), 3),
           "density_veh_per_km": round(float(alive.sum(2).mean()) / (4 * cfg.L / 1000), 2),
           "queue_end_per_env": round(float(rec["queue"][-1].mean()), 2),
           "queue_mean_per_env": round(float(rec["queue"][w:].mean()), 2),
           "slots_full_frac": round(float((occ[w:] >= N).mean()), 4),
           "lamp_zone_mean": round(float(lampZ.mean()), 3) if len(lampZ) else None,
           "lamp_zone_std": round(float(lampZ.std()), 3) if len(lampZ) else None,
           "deadlocks": len(episodes), "deadlocks_per_hour": round(len(episodes) / hours, 2),
           "deadlock_time_frac": round(in_dl / (T * B), 4)}
    if sensitivity:
        out["sensitivity"] = sensitivity
    return out


def run_job(job) -> dict:
    """One sweep cell: (spec, cfg_d, demand, seed, envs, seconds, noise, lamp) -> one JSON line (metrics +
    the deadlock episodes)."""
    spec, cfg_d, demand, seed, envs, seconds, noise, lamp = job
    t0 = time.time()
    cfg = SimConfig(**{**(cfg_d or {}), "demand": demand})
    rec, summ, sim = record_run(spec, cfg, seed, envs, seconds, noise, lamp)
    eps = detect(rec, sim.cfg)
    sens = {f"N{n}_T{t:g}": len(detect(rec, sim.cfg, n, t, details=False)) for n, t in ((2, 5.0), (3, 10.0), (4, 5.0))}
    line = {"driver": spec, "demand": demand, "seed": seed, "envs": envs, "seconds": seconds,
            "noise": noise or "none", "lamp": lamp, **run_metrics(rec, sim.cfg, summ, seconds, eps, sens),
            "episodes": eps, "wall_s": round(time.time() - t0, 1)}
    return json.loads(json.dumps(line, default=_jsonable))


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def sweep(drivers, demands=SWEEP_DEMANDS, seeds=SELECTION_SEEDS, envs=ENVS, seconds=SWEEP_SECONDS, noise=None, lamp="normal",
          cfg_d=None, out=None, workers=1, log=print) -> list[dict]:
    """Every driver x demand x seed; appends one JSON line per run to ``out``."""
    jobs = [(d, cfg_d, dm, s, envs, seconds, noise, lamp) for d in drivers for dm in demands for s in seeds]
    t0 = time.time()
    lines = pool_map(run_job, jobs, workers)
    if out:
        append_jsonl(out, lines)
    for ln in lines:
        log(f"{label(ln['driver'])} noise={ln['noise']} lamp={ln['lamp']} d={ln['demand']} s={ln['seed']}: "
            f"thr {ln['throughput_per_min']}, crash/1000 {ln['crash_cars_per_1000']}, timeouts {ln['timeouts']}, "
            f"deadlocks {ln['deadlocks']} ({ln['wall_s']} s)")
    log(f"deadlock sweep: {len(lines)} runs in {time.time() - t0:.0f} s" + (f" -> {out}" if out else ""))
    return lines


# ------------------------------------------------------------------ aggregation
def label(spec: str) -> str:
    return Path(spec.removeprefix("policy:")).stem if spec.startswith("policy:") or spec.endswith(".npz") else spec


MEAN_KEYS = ("demand_per_min", "throughput_per_min", "throughput_steady_per_min", "crash_cars_per_1000",
             "timeouts_per_1000", "delay_mean_s", "delay_p95_s", "queue_end_per_env", "cars_map_mean",
             "cars_box_mean", "density_veh_per_km", "slots_full_frac", "deadlocks_per_hour", "deadlock_time_frac",
             "jam_time_frac", "timeouts_in_zone_frac")


def aggregate(lines, cfg: SimConfig | None = None) -> list[dict]:
    """Seed-averaged rows per (driver, noise, lamp, demand) incl. pooled deadlock statistics (cfg: the world of the
    runs, for its step dt; default world)."""
    dt = (cfg or SimConfig()).dt
    groups = {}
    for ln in lines:
        groups.setdefault((ln["driver"], ln["noise"], ln["lamp"], ln["demand"]), []).append(ln)
    rows = []
    for (drv, noise, lamp, dem), ls in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2], kv[0][3])):
        row = {"driver": label(drv), "noise": noise, "lamp": lamp, "demand": dem, "n_seeds": len(ls)}
        for k in MEAN_KEYS:
            vals = [ln[k] for ln in ls if ln.get(k) is not None and ln[k] == ln[k]]
            row[k] = round(float(np.mean(vals)), 3) if vals else None
        row["seed_spread_thr"] = [ln["throughput_per_min"] for ln in ls]
        eps = [e for ln in ls for e in ln["episodes"]]
        row.update(episode_stats(eps, dt))
        sens = {}
        for ln in ls:
            for k, v in ln.get("sensitivity", {}).items():
                sens[k] = sens.get(k, 0) + v
        hours = sum(ln["envs"] * ln["seconds"] for ln in ls) / 3600
        row["sensitivity_per_hour"] = {k: round(v / hours, 1) for k, v in sens.items()}
        rows.append(row)
    return rows


def episode_stats(eps, dt: float) -> dict:
    """Pooled statistics of deadlock episodes (detect) of runs with step dt."""
    if not eps:
        return {"n_episodes": 0}
    dur = np.array([e["dur_s"] for e in eps])
    causes = {}
    for e in eps:
        causes[e["cause"]] = causes.get(e["cause"], 0) + 1
    out = {"n_episodes": len(eps), "dur_median_s": float(np.median(dur)), "dur_p90_s": float(np.percentile(dur, 90)),
           "dur_max_s": float(dur.max()), "causes": causes, "n_cars_mean": round(float(np.mean([e["n_cars"] for e in eps])), 2),
           "reformed_frac": round(sum(bool(e.get("reformed")) for e in eps) / len(eps), 3)}
    spells, cur = [], None
    for e in eps:  # chains of episodes (a new one within 10 s after the previous, same run and env)
        if cur is not None and e.get("reformed"):
            cur[1] = e["end"]
        else:
            if cur is not None:
                spells.append(cur)
            cur = [e["start"], e["end"]]
    spells.append(cur)
    sd = np.array([(hi - lo) * dt for lo, hi in spells])
    out.update(n_spells=len(spells), spell_dur_median_s=round(float(np.median(sd)), 1),
               spell_dur_p90_s=round(float(np.percentile(sd, 90)), 1), spell_dur_max_s=round(float(sd.max()), 1))
    out["no_zone_exit_frac"] = round(float(np.mean([e["zone_exits"] == 0 for e in eps])), 3)
    out["zone_exits_per_min_during"] = round(float(sum(e["zone_exits"] for e in eps) / max(
        sum(e["end"] - e["onset"] for e in eps) / (60 / dt), 1e-9)), 2)
    out["timeouts_of_involved"] = int(sum(e["n_timeouts"] for e in eps))
    out["crashed_of_involved"] = int(sum(e["n_crashed"] for e in eps))
    out["cars_involved"] = int(sum(e["n_cars"] for e in eps))
    geo = [e["geo"] for e in eps if "geo" in e]
    if geo:
        cnt = lambda key: {k: sum(g[key] == k for g in geo) for k in sorted({g[key] for g in geo})}  # noqa: E731
        out["kind"] = cnt("kind")
        out["cycle_len"] = cnt("cycle_len")
        out["n_inner_mean"] = round(float(np.mean([g["n_inner"] for g in geo])), 2)
        out["n_inner_arms"] = {k: sum(len(g["inner_arms"]) == k for g in geo) for k in range(5)}
        flat = lambda key: [v for g in geo for v in g[key]]  # noqa: E731
        for key in ("inner_man", "blocker_rel", "blocker_face"):
            vals = flat(key)
            out[key] = {k: vals.count(k) for k in sorted(set(vals))}
        out["wreck_in_zone_frac"] = round(float(np.mean([g["wreck_in_zone"] for g in geo])), 3)
    lamps = [e["lamp"] for e in eps if "lamp" in e]
    if lamps:
        cars = [c for lp in lamps for c in lp["cars"]]

        def lm(sel, key):
            v = [c[key] for c in cars if sel(c) and c[key] is not None]
            return round(float(np.mean(v)), 3) if v else None

        out["lamp_episode"] = {f"{who}_{w}": lm(sel, f"lamp_{w}") for who, sel in (("goer", lambda c: c["goer"]),
                                                                             ("waiter", lambda c: not c["goer"]))
                       for w in ("lead", "during", "last")}
        gm = [lp["goer_man"] for lp in lamps if lp["goer_man"]]
        out["goer_man"] = {k: gm.count(k) for k in sorted(set(gm))}
        go = [lp["goer_oldest"] for lp in lamps if lp["goer_oldest"] is not None]
        out["goer_oldest_frac"] = round(float(np.mean(go)), 3) if go else None
    return out


FUND_KEYS = ("demand_per_min", "cars_map_mean", "cars_box_mean", "density_veh_per_km", "throughput_per_min",
             "throughput_steady_per_min", "queue_end_per_env", "timeouts_per_1000", "crash_cars_per_1000",
             "delay_mean_s", "deadlocks_per_hour", "deadlock_time_frac",
             "jam_time_frac", "timeouts_in_zone_frac")


def fundamental(lines, names=None) -> dict:
    """Fundamental-diagram data: per driver a list of seed-averaged points (only plain runs: no noise, lamps
    normal). names: optional {spec label: display key}. The meta records the envs / seconds / seeds of the lines."""
    names = names or {}
    plain = [ln for ln in lines if ln["noise"] == "none" and ln["lamp"] == "normal"]
    setup = lambda key: sorted({ln[key] for ln in plain})  # noqa: E731
    out = {"meta": {
        "density": "cars_map_mean = mean number of driving cars on the map (box + 4 arms of L = 50 m, both "
                   "directions) per environment after a 30 s warm-up; density_veh_per_km = cars_map_mean / 0.2 km "
                   "of road (4 x 50 m, both directions); cars_box_mean = cars within max(|x|,|y|) <= 16 m",
        "throughput": "throughput_per_min = exits per minute over the whole run (as cli bench); "
                      "throughput_steady_per_min = after the 30 s warm-up",
        "demand": "demand (arrivals /s/arm, x 240 = cars/min); demand_per_min = measured arrivals",
        "protocol": f"envs {setup('envs')} x seconds {setup('seconds')}, traffic seeds {setup('seed')}, "
                    "default world (cli deadlock)",
        "deadlock": f"deadlock = >= {N_MIN} cars stuck in the box zone (net displacement < {STUCK_D:g} m over "
                    f"{T_MIN:g} s), >= {N_INNER} of them inside the conflict area (smartcrossing/deadlock.py detect)"},
        "drivers": {}}
    for r in aggregate(plain):
        key = names.get(r["driver"], r["driver"])
        d = out["drivers"].setdefault(key, {"label": r["driver"], "points": []})
        d["points"].append({"demand": r["demand"], "density": r["cars_map_mean"],
                            "throughput": r["throughput_steady_per_min"], **{k: r[k] for k in FUND_KEYS}})
    return out


def table(rows) -> str:
    """Markdown table of aggregate() rows."""
    head = ("| driver | noise | lamp | demand | thr/min | crash+off/1000 | timeouts/1000 | delay mean / p95 | "
            "queue end/env | cars map / box | deadlocks/h | time in DL | DL dur med / p90 s |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    f = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    for r in rows:
        lines.append(f"| {r['driver']} | {r['noise']} | {r['lamp']} | {r['demand']} | {f(r['throughput_per_min'])} | "
                     f"{f(r['crash_cars_per_1000'], 0)} | {f(r['timeouts_per_1000'], 0)} | {f(r['delay_mean_s'])} / "
                     f"{f(r['delay_p95_s'])} | {f(r['queue_end_per_env'])} | {f(r['cars_map_mean'])} / "
                     f"{f(r['cars_box_mean'])} | {f(r['deadlocks_per_hour'])} | {f(100 * r['deadlock_time_frac'], 1)}% | "
                     f"{f(r.get('dur_median_s'))} / {f(r.get('dur_p90_s'))} |")
    return "\n".join(lines)


def summarize_files(files, names=None, fundamental_out=None, json_out=None, log=print) -> list[dict]:
    """``cli deadlock-summarize``: aggregate sweep JSONL files and print the table; optionally write the rows
    (json_out) and the fundamental-diagram JSON (names: {spec label: display key}, e.g. {"round7_best": "brain"})."""
    lines = [json.loads(s) for f in files for s in Path(f).read_text(encoding="utf-8").splitlines() if s.strip()]
    rows = aggregate(lines)
    log(table(rows))
    if json_out:
        Path(json_out).write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")
    if fundamental_out:
        Path(fundamental_out).write_text(json.dumps(fundamental(lines, names), indent=1), encoding="utf-8",
                                         newline="\n")
    return rows
