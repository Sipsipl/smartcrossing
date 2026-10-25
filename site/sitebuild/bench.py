"""Measurements in the Python sim: the cached site bench (throughput vs demand of the reference drivers and the
selected brains) and the side-by-side replays of "Vergleich"."""

import base64
import hashlib
import json
import multiprocessing as mp
import time

import numpy as np

from smartcrossing.protocol import ENVS as BENCH_ENVS
from smartcrossing.protocol import REPORT_SEEDS
from smartcrossing.protocol import SECONDS as BENCH_SECONDS
from smartcrossing.sim import BRAKE_ACC, SimConfig

from .context import Context
from .discover import Brain

MP_START = "fork" if "fork" in mp.get_all_start_methods() else "spawn"  # Windows has no fork
# Traffic seeds, the same for all drivers (paired); results are averaged. These are the held-out REPORTING seeds
# (protocol.REPORT_SEEDS = 4000/5000/6000), not the SELECTION seeds 1000/2000/3000 that cli select / evalset use to
# pick checkpoints: numbers on the website must not be flattered by the selection.
BENCH_SEEDS = REPORT_SEEDS


def policy_driver(sim, payload):
    """PolicyDriver for a bench / replay payload (params, sizes, in_scale[, arch]; arch None = MLP)."""
    from smartcrossing.policy import PolicyDriver

    params, sizes, in_scale, *rest = payload
    return PolicyDriver(sim, np.asarray(params), sizes, in_scale=np.asarray(in_scale), archs=rest[0] if rest else None)


def make_driver(sim, kind: str, payload):
    """Driver of a bench / replay job: a rule-based baseline (kind "baseline", payload = its name: the shared
    smartcrossing.drivers.make_driver) or a policy from its weights (policy_driver)."""
    if kind == "baseline":
        from smartcrossing.drivers import make_driver as by_spec

        return by_spec(payload, sim)
    return policy_driver(sim, payload)


def pol_entry(c: dict) -> tuple:
    """(bench key, kind, payload, physics) of a discovered checkpoint."""
    return (f"pol:{c['hash']}", "policy", Brain(c["policy"]).payload(), c["physics"])


# ------------------------------------------------------------------ bench (cached)
def _bench_job(job):
    """One bench run (smartcrossing.evalset.bench_one), its KEEP_KEYS metrics."""
    from smartcrossing.evalset import bench_one, keep

    kind, payload, physics, demand, seed = job
    cfg = SimConfig(**{**physics, "demand": demand})
    return keep(bench_one(lambda sim: make_driver(sim, kind, payload), cfg, seed, BENCH_ENVS, BENCH_SECONDS))


def _bench_ck(key, physics, d, seed, sim_src):
    """Cache key of one bench run (format fixed: the committed cache site/data/bench_cache.json uses it)."""
    return f"{key}|{json.dumps(physics, sort_keys=True)}|{d}|{BENCH_ENVS}x{BENCH_SECONDS}@{seed}|{sim_src}"


def _mean_results(rs: list[dict]) -> dict:
    """Mean over the traffic seeds, field by field (None if a seed has no value)."""
    out = {}
    for k in rs[0]:
        v = [r[k] for r in rs if r[k] is not None]
        out[k] = round(sum(v) / len(v), 3) if v else None
    return out


def run_bench(ctx: Context, entries: list[tuple[str, str, object, dict, list]], force: bool, workers: int) -> dict:
    """entries: (key, kind, payload, physics, demands) -> {key: {demand: result}}. Results are cached in
    ctx.cache, keyed by policy weights, physics and the sim source (sim.py + baselines.py)."""
    sim_src = hashlib.sha1((ctx.root / "smartcrossing" / "sim.py").read_bytes()
                           + (ctx.root / "smartcrossing" / "baselines.py").read_bytes()).hexdigest()[:12]
    cache, n_stored = {}, 0
    if ctx.cache.exists() and not force:
        cache = json.loads(ctx.cache.read_text(encoding="utf-8"))
        n_stored = len(cache)
        cache = {k: v for k, v in cache.items() if k.endswith(f"|{sim_src}")}  # drop results of older sim versions
    if len(cache) < n_stored:
        ctx.cache.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
    jobs, keys = [], []
    for key, kind, payload, physics, demands in entries:
        for d in demands:
            for seed in BENCH_SEEDS:
                ck = _bench_ck(key, physics, d, seed, sim_src)
                if ck in cache:
                    continue
                jobs.append((kind, payload, physics, d, seed))
                keys.append(ck)
    if jobs:
        print(f"  benchmarking {len(jobs)} runs ({BENCH_ENVS} envs x {BENCH_SECONDS:.0f} s each, {len(BENCH_SEEDS)} traffic seeds) ...")
        t0 = time.time()
        ctx.cache.parent.mkdir(parents=True, exist_ok=True)
        with mp.get_context(MP_START).Pool(workers) as pool:
            for i, (ck, res) in enumerate(zip(keys, pool.imap(_bench_job, jobs, chunksize=1))):
                cache[ck] = res
                if (i + 1) % 20 == 0 or i + 1 == len(jobs):  # save progress (an interrupted build resumes)
                    ctx.cache.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
                    print(f"    {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
        print(f"  bench done in {time.time() - t0:.0f} s")
    out = {}
    for key, kind, payload, physics, demands in entries:
        out.setdefault(key, {})
        for d in demands:
            out[key][f"{d:g}"] = _mean_results([cache[_bench_ck(key, physics, d, sd, sim_src)] for sd in BENCH_SEEDS])  # as JS String(d)
    return out


# ------------------------------------------------------------------ replays ("Vergleich" side by side)
REPLAY_DEMANDS = [0.05, 0.15, 0.3]
REPLAY_SEED, REPLAY_WARMUP, REPLAY_SECONDS = 7, 15.0, 90.0
Q_POS, Q_TH, Q_LAMP = 0.05, 0.01, 100.0  # quantisation: 5 cm, 0.01 rad, lamp in 1/100


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _replay_job(job):
    """Drive one sim (1 env) and record every car as a compact track:
    [t0, man, end_step, end_kind, qx0, qy0, qth0, width, b64, brake, wreck_from] where b64 holds per
    sample (dx, dy, dth[, lamp per channel]) as int8 (width 1) or int16 (width 2), deltas of the
    quantised values (exact reconstruction), brake the sample ranges [start, end) with the
    brake lights on (acceleration < sim.BRAKE_ACC), flattened, and wreck_from the first sample at which
    the car is a wreck (-1: never; a wreck stays cfg.wreck_time, its samples continue the track).
    Plus crash marks and the queue per arm. "lamps" = lamp channels per sample (0 for the baselines)."""
    from smartcrossing.sim import DONE_NAMES, Sim

    kind, payload, physics, demand = job
    cfg = SimConfig(**{**physics, "demand": demand})
    sim = Sim(cfg, 1, seed=REPLAY_SEED)
    drv = make_driver(sim, kind, payload)
    lamps = 0 if kind == "baseline" else cfg.n_signal
    tracks, marks, queue = {}, [], []
    n_warm, n = int(round(REPLAY_WARMUP / cfg.dt)), int(round(REPLAY_SECONDS / cfg.dt))
    for k in range(-n_warm, n):
        sim.step(drv.act())
        if k < 0:
            continue
        for slot in np.nonzero(sim.done_kind[0])[0]:
            tr = tracks.get(int(sim.done_uid[0, slot]))
            if tr is not None:
                tr["end"] = (k, int(sim.done_kind[0, slot]))
        for _, ev, x, y in sim.events:
            marks.append([k, 1 if ev == "crash" else 2, int(round(x / Q_POS)), int(round(y / Q_POS))])
        for slot in np.nonzero(sim.alive[0] | sim.wreck[0])[0]:
            uid = int(sim.uid[0, slot])
            tr = tracks.get(uid)
            if tr is None:
                if sim.wreck[0, slot]:  # a wreck from before the recording started
                    continue
                tr = tracks[uid] = {"t0": k, "man": int(sim.man[0, slot]), "s": [], "b": [], "end": (None, 0), "w": -1}
            if sim.wreck[0, slot] and tr["w"] < 0:
                tr["w"] = len(tr["s"])
            tr["b"].append(bool(sim.acc[0, slot] < BRAKE_ACC))
            tr["s"].append((sim.x[0, slot], sim.y[0, slot], sim.th[0, slot], *(sim.sig[0, slot, :lamps] if lamps else ())))
        queue.append(np.minimum(sim.queue[0], 255))
    out = []
    for uid, tr in tracks.items():
        s = np.asarray(tr["s"], dtype=float)
        q = np.empty_like(s, dtype=np.int64)
        q[:, 0] = np.round(s[:, 0] / Q_POS)
        q[:, 1] = np.round(s[:, 1] / Q_POS)
        q[:, 2] = np.round(np.unwrap(s[:, 2]) / Q_TH)
        d = q.copy()
        d[1:, :3] = q[1:, :3] - q[:-1, :3]
        d[0, :3] = 0
        if lamps:
            d[:, 3:] = np.round(np.clip(s[:, 3:], -1, 1) * Q_LAMP)
        width = 1 if np.abs(d).max() <= 127 else 2
        end_k, end_kind = tr["end"]
        b = np.diff(np.concatenate([[0], np.asarray(tr["b"], dtype=np.int8), [0]]))
        runs = np.stack([np.nonzero(b == 1)[0], np.nonzero(b == -1)[0]], 1).ravel().tolist()
        out.append([tr["t0"], tr["man"], end_k if end_k is not None else -1, end_kind,
                    int(q[0, 0]), int(q[0, 1]), int(q[0, 2]), width, _b64(d.astype(np.int8 if width == 1 else "<i2")), runs,
                    tr["w"]])
    return {
        "demand": demand, "steps": n, "dt": cfg.dt, "lamps": lamps, "wreck_time": cfg.wreck_time,
        "kinds": list(DONE_NAMES), "tracks": out, "marks": marks,
        "queue": _b64(np.asarray(queue, dtype=np.uint8)),
    }


def record_replays(drivers: list[tuple[str, str, object, dict]], workers: int) -> dict:
    """drivers: (name, kind, payload, physics) -> {"q": quantisation, "runs": {name: {demand: replay}}}."""
    jobs = [(kind, payload, physics, d) for _, kind, payload, physics in drivers for d in REPLAY_DEMANDS]
    t0 = time.time()
    with mp.get_context(MP_START).Pool(workers) as pool:
        res = pool.map(_replay_job, jobs, chunksize=1)
    print(f"  replays: {len(jobs)} recorded in {time.time() - t0:.0f} s")
    runs = {}
    for (name, *_), r in zip([dr for dr in drivers for _ in REPLAY_DEMANDS], res):
        runs.setdefault(name, {})[str(r["demand"])] = r
    return {"q": {"pos": Q_POS, "th": Q_TH, "lamp": Q_LAMP}, "seed": REPLAY_SEED, "seconds": REPLAY_SECONDS,
            "demands": REPLAY_DEMANDS, "runs": runs}
