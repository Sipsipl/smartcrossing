"""Collect and bench phases: which brains the site shows (best per method, main / history / previous), their
Zeitreise timelines, warm-start chains, learning curves and side notes, and the bench of all of them plus the
rule-based drivers."""

import numpy as np

from smartcrossing.policy import default_in_scale, init_params, layer_sizes, unpack
from smartcrossing.policy_compat import interface_cfg, physics_of
from smartcrossing.protocol import SITE_DEMANDS
from smartcrossing.sim import SimConfig

from .bench import pol_entry, run_bench
from .context import Context
from .curves import read_curve, timeline
from .discover import Brain, resolve, run_world, world_of
from .fundamental import gate_text
from .texts import run_info as _run_info

TIMELINE_DEMANDS = [0.05, 0.15]
# throughput / delay along the learning curve (/lernen): checkpoints of the main brain's chain re-measured today
CAPACITY_DEMANDS = {"thr": 0.6, "delay": 0.3}  # overload (144 cars/min): throughput; medium (72/min): delay per car
CAPACITY_POINTS = 30
GEOMETRY_KEYS = ("L", "W", "Rc", "a_lat_max", "v_reverse")  # the "world": same for all drivers compared
REWARD_KEYS = ("r_exit", "r_wrong_exit", "r_crash", "r_crash_speed", "r_offroad", "r_offroad_speed", "r_timeout",
               "r_progress", "r_time", "r_queue", "crash_cost_speed", "L", "v_max", "spawn_speed", "max_age")


def run_info(ctx: Context, run: str | None) -> dict:
    """texts.run_info with the stop-gate numbers of ctx.note_ckpt filled in."""
    return _run_info(run, lambda s: gate_text(ctx, s, ctx.note_ckpt))


def reward_of(train_cfg: dict) -> dict:
    """The reward rules (and the few world numbers an example trip needs) a policy was trained with."""
    d = SimConfig()
    return {k: train_cfg.get(k, getattr(d, k)) for k in REWARD_KEYS}


def choose(ctx: Context, main: list[str], quick: dict) -> dict:
    """Best checkpoint per method by quick bench score (exit-weighted throughput at 0.15)."""
    best = {}
    for cid, c in ctx.found.items():
        r = quick.get(cid)
        if r is None:
            continue
        score = r["throughput_per_min"] * r["exit_ok_rate"]
        if c["method"] not in best or score > best[c["method"]][0]:
            best[c["method"]] = (score, cid)
    picked = {m: cid for m, (_, cid) in best.items()}
    for spec in main:  # explicit --main overrides
        cid = resolve(ctx, spec)
        picked[ctx.found[cid]["method"]] = cid
    return picked


def capacity_points(ctx: Context, chain: list, main_id: str, n_max: int = CAPACITY_POINTS) -> list[dict]:
    """Checkpoints along the warm-start chain (run, offset) of the main brain in its world (older crossings
    are left out: their brains cannot be measured fairly on today's), up to the main brain itself; a run's
    checkpoints stop where the next run of the chain starts. At most n_max, evenly spaced; the main brain stays."""
    found = ctx.found
    main = found[main_id]
    world = world_of(main["physics"])
    off = dict(chain)
    by_run = {}
    for cid, c in found.items():
        if c["run"] in off and c["gen"] is not None and world_of(c["physics"]) == world and "/abl_" not in cid:
            by_run.setdefault(c["run"], []).append((off[c["run"]] + c["gen"], cid))
    end = off[main["run"]] + main["gen"]
    pts = []
    runs = [r for r, _ in chain if r in by_run]
    for i, run in enumerate(runs):
        nxt = min(x for x, _ in by_run[runs[i + 1]]) if i + 1 < len(runs) else end + 1
        pts += [(x, cid) for x, cid in by_run[run] if x < nxt and x <= end]
    by_x = {}
    for x, cid in sorted(pts):
        if x not in by_x or cid == main_id:
            by_x[x] = cid
    pts = sorted(by_x.items())
    if len(pts) > n_max:  # evenly spaced along the x axis (runs with many checkpoints would crowd it otherwise)
        xs = np.array([x for x, _ in pts], dtype=float)
        idx = {int(np.abs(xs - t).argmin()) for t in np.linspace(xs[0], xs[-1], n_max)}
        pts = [pts[i] for i in sorted(idx)]
    return [{"x": x, "id": cid} for x, cid in pts]


def random_policy(physics: dict | None = None) -> dict:
    """Random brain with the observation interface of `physics` (default: the current one)."""
    cfg = interface_cfg(physics) if physics else SimConfig()
    sizes = layer_sizes(cfg)
    params = init_params(sizes, 7)
    return {
        "sizes": sizes, "in_scale": default_in_scale(cfg).tolist(), "activation": "tanh", "output": "tanh",
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in unpack(params, sizes)], "cfg": {},
    }


def collect(ctx: Context, a) -> dict:
    """Collect phase: pick the brains (best per method, main / history / previous), the Zeitreise
    timelines, warm-start chains, learning curves and side notes; returns the selection incl. the
    policies the site shows and the bench entries they need."""
    found = ctx.found

    def cid_of(spec):
        return resolve(ctx, spec)

    # methods without an explicit --main: quick bench of all their checkpoints at demand 0.15
    fixed = {found[cid_of(s)]["method"] for s in a.main}
    quick_entries = [(*pol_entry(c), [0.15]) for c in found.values() if c["method"] not in fixed]
    qb = run_bench(ctx, quick_entries, a.bench, a.workers)
    quick = {cid: qb[pol_entry(c)[0]]["0.15"] for cid, c in found.items() if c["method"] not in fixed}
    picked = choose(ctx, a.main, quick)
    main_id = cid_of(a.main[0]) if a.main else max(picked.values(), key=lambda k: quick[k]["throughput_per_min"] * quick[k]["exit_ok_rate"])
    hist_id = cid_of(a.history) if a.history else None
    prev_id = cid_of(a.previous) if a.previous else None
    print(f"  best per method: {picked}, main: {main_id}, history: {hist_id}")

    policies, bench_entries = {}, []

    def use(cid, demands):
        c = found[cid]
        if cid not in policies:
            info = run_info(ctx, c["run"])
            policies[cid] = {
                "id": cid, "run": c["run"], "run_label": info["label"], "run_label_en": info["label_en"], "file": c["file"],
                "method": c["method"], "gen": c["gen"], "physics": c["physics"], "adapted": c["adapted"],
                "policy": Brain(c["policy"]).rounded(4), "bench": {},
            }
        bench_entries.append((*pol_entry(c), demands))

    timelines, curves, chains, capacity = {}, {}, {}, []
    for method, cid in picked.items():
        use(cid, SITE_DEMANDS)
        parts = [cid]
        if hist_id and found[hist_id]["method"] == method and hist_id != cid:
            use(hist_id, SITE_DEMANDS)
            parts.insert(0, hist_id)
        pts_all, chain_all, shift = [], [], 0
        for i, pid in enumerate(parts):
            # history first (fewer Zeitreise points), then the main chain, shifted to follow it
            pts, chain = timeline(ctx, pid, max(2, a.timeline // 2) if i < len(parts) - 1 else a.timeline, keep={prev_id})
            crv = {r: read_curve(ctx, r, off) for r, off in chain}
            if i > 0 and any(crv.values()):
                lo = min(row["x"] for rows in crv.values() for row in rows)
                if lo < shift:  # iteration counts restart: move the whole chain behind the history
                    chain = [(r, int(off + shift - lo)) for r, off in chain]
                    crv = {r: read_curve(ctx, r, off) for r, off in chain}
                    pts = [{**p, "gen": int(p["gen"] + shift - lo)} for p in pts]
            pts_all += pts
            chain_all += chain
            curves.setdefault(method, {}).update(crv)
            shift = 1 + max([row["x"] for rows in crv.values() for row in rows] + [p["gen"] for p in pts_all] + [0])
        chains[method] = [{"run": r, "offset": off, "world": run_world(ctx, r), **run_info(ctx, r)} for r, off in chain_all]
        if cid == main_id:
            capacity = capacity_points(ctx, chain_all, main_id)
            for p in capacity:
                bench_entries.append((*pol_entry(found[p["id"]]), sorted(CAPACITY_DEMANDS.values())))
        for p in pts_all:
            use(p["id"], TIMELINE_DEMANDS)
        timelines[method] = pts_all
    # random brain as "generation 0" of the ES timeline
    if "es" in timelines:
        phys = found[picked["es"]]["physics"]
        rp = random_policy(phys)
        rid = "zufall"
        policies[rid] = {"id": rid, "run": None, "file": None, "method": "es", "gen": 0, "physics": phys,
                         "adapted": None, "policy": Brain(rp).rounded(4), "bench": {}}
        bench_entries.append(("pol:random7", "policy", Brain(rp).payload(), phys, TIMELINE_DEMANDS))
        if not timelines["es"] or timelines["es"][0]["gen"] > 0:
            timelines["es"].insert(0, {"gen": 0, "id": rid})
    # other (dead-end) runs: their best checkpoint as a side note
    others = []
    in_chain = {r["run"] for ch in chains.values() for r in ch}
    for run in sorted({c["run"] for c in found.values()} - in_chain):
        cands = [cid for cid, c in found.items() if c["run"] == run]
        cid = max(cands, key=lambda k: (found[k]["gen"] or 0))
        use(cid, TIMELINE_DEMANDS)
        others.append({"run": run, "id": cid, "method": found[cid]["method"], **run_info(ctx, run)})

    if prev_id:
        use(prev_id, SITE_DEMANDS)
    return {"picked": picked, "main_id": main_id, "hist_id": hist_id, "prev_id": prev_id,
            "policies": policies, "bench_entries": bench_entries, "timelines": timelines, "curves": curves,
            "chains": chains, "others": others, "capacity": capacity}


def bench_all(ctx: Context, sel: dict, cfg: SimConfig, a) -> dict:
    """Bench phase: the rule-based drivers in the main brain's world (and in each other brain's world on
    "Vergleich", e.g. the old narrow crossing, so every brain is compared on its own crossing) plus all
    selected policies; fills each policy's "bench". Returns the baseline results and worlds."""
    found = ctx.found

    def world_phys(cid):
        return {**physics_of(cfg.__dict__), **{k: v for k, v in found[cid]["physics"].items() if k in GEOMETRY_KEYS}}

    base_worlds = {}  # world name -> physics; the main brain's world first
    for cid in (sel["main_id"], sel["hist_id"], sel["picked"].get("es")):
        if cid:
            base_worlds.setdefault(world_of(found[cid]["physics"]), world_phys(cid))
    if not base_worlds:
        base_worlds[world_of(physics_of(cfg.__dict__))] = physics_of(cfg.__dict__)
    main_world = next(iter(base_worlds))
    base_key = lambda drv, w: f"base:{drv}" if w == main_world else f"base:{drv}:{w}"  # noqa: E731
    entries = list(sel["bench_entries"])
    for w, phys in base_worlds.items():
        for drv in ("light", "reservation"):
            entries.append((base_key(drv, w), "baseline", drv, phys, SITE_DEMANDS))
    res = run_bench(ctx, entries, a.bench, a.workers)
    for cid, p in sel["policies"].items():
        key = f"pol:{found[cid]['hash']}" if cid in found else "pol:random7"
        p["bench"] = res[key]
    by_world = {w: {drv: res[base_key(drv, w)] for drv in ("light", "reservation")} for w in base_worlds}
    for p in sel["capacity"]:  # learning-curve points: throughput at overload, delay at medium traffic
        r = res[f"pol:{found[p['id']]['hash']}"]
        p["thr"] = r[str(CAPACITY_DEMANDS["thr"])]["throughput_per_min"]
        p["delay"] = r[str(CAPACITY_DEMANDS["delay"])]["delay_mean_s"]
    return {"baselines_by_world": by_world, "main_world": main_world, "base_phys": base_worlds[main_world]}
