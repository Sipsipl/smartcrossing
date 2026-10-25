"""Standard evaluation protocol (one command for every new brain and every comparison):

    poetry run python -m smartcrossing.cli evalset --driver policy:results/round7_best.npz \
        --driver reservation --out results/x_eval.jsonl [--workers 4]

Per driver: the bench at protocol.EVAL_DEMANDS x SELECTION_SEEDS (traffic seeds, the same for all drivers:
paired) with protocol.ENVS envs x SECONDS each in the default world (``--cfg`` overrides it), plus the stop gate (gates.py). Appends one
JSON line per bench (``cli bench``'s line + ``seed`` + ``"kind": "bench"``) and one per gate
(``"kind": "gate"``), the format of results/round2_eval.jsonl / round4_eval.jsonl.

``bench_one`` is the one bench run shared by ``cli bench``, ``ppo.eval_job`` and ``site/build.py``;
``KEEP_KEYS`` are the metrics the training evals and the site keep of it.
"""

import time

from .jobs import append_jsonl, pool_map
from .protocol import ENVS, EVAL_DEMANDS, SECONDS, SELECTION_SEEDS
# metrics of a bench that the training evals (ppo.eval_job, + queued_end) and the site bench keep
KEEP_KEYS = ("throughput_per_min", "demand_per_min", "exit_ok_rate", "crash_cars_per_1000", "crash_cars",
             "offroad_cars", "timeouts", "delay_mean_s")


def bench_one(spec, cfg, seed: int, envs: int = ENVS, seconds: float = SECONDS) -> dict:
    """metrics.summarize of one bench run: driver ``spec`` (a cli driver spec such as ``reservation`` /
    ``policy:<file>``, or a callable sim -> driver) in a fresh Sim(cfg, envs, seed) for ``seconds``."""
    from .drivers import make_driver
    from .metrics import summarize
    from .sim import Sim

    sim = Sim(cfg, envs, seed=seed)
    drv = spec(sim) if callable(spec) else make_driver(spec, sim)
    for _ in range(int(seconds / cfg.dt)):
        sim.step(drv.act())
    return summarize(sim, seconds)


def keep(r: dict, keys=KEEP_KEYS) -> dict:
    """The ``keys`` of a summarize dict, rounded to 3 decimals (NaN -> None)."""
    return {k: (round(float(r[k]), 3) if r[k] == r[k] else None) for k in keys}


def bench_line(spec: str, cfg, seed: int, envs: int = ENVS, seconds: float = SECONDS) -> dict:
    """``cli bench``'s JSON line: driver, demand, all metrics (floats rounded to 3), wall_s."""
    t0 = time.time()
    res = {"driver": spec, "demand": cfg.demand, **bench_one(spec, cfg, seed, envs, seconds)}
    res["wall_s"] = round(time.time() - t0, 1)
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}


def gate_line(spec: str, cfg_d: dict | None = None) -> dict:
    """Stop-gate summary as an eval line (driver, kind, then the summary)."""
    from .gates import stop_gate

    s, _ = stop_gate(spec, cfg_d)
    return {"driver": s["driver"], "kind": "gate", **{k: v for k, v in s.items() if k != "driver"}}


def _job(job) -> dict:
    kind, spec, cfg_d, demand, seed, envs, seconds = job
    if kind == "gate":
        return gate_line(spec, cfg_d)
    from .sim import SimConfig

    return {**bench_line(spec, SimConfig(**{**cfg_d, "demand": demand}), seed, envs, seconds), "seed": seed,
            "kind": "bench"}


def evalset(drivers, out=None, cfg_d: dict | None = None, demands=EVAL_DEMANDS, seeds=SELECTION_SEEDS, envs: int = ENVS,
            seconds: float = SECONDS, gate: bool = True, workers: int = 1, log=print) -> list[dict]:
    """Run the protocol for every driver spec; append the lines to ``out`` (if given) in a fixed order
    (per driver: demands x seeds, then the gate) and return them. workers > 1: a spawn process pool."""
    cfg_d = dict(cfg_d or {})
    jobs = []
    for spec in drivers:
        jobs += [("bench", spec, cfg_d, d, s, envs, seconds) for d in demands for s in seeds]
        if gate:
            jobs.append(("gate", spec, cfg_d or None, None, None, envs, seconds))
    t0 = time.time()
    lines = pool_map(_job, jobs, workers)
    if out:
        append_jsonl(out, lines)
    for line in lines:
        if line["kind"] == "gate":
            log(f"{line['driver']}: gate stop_rate_feasible {line['stop_rate_feasible']}, crash {line['crash']}, "
                f"block_stop_rate_feasible {line.get('block_stop_rate_feasible')}")
        else:
            log(f"{line['driver']} d={line['demand']} seed={line['seed']}: thr {line['throughput_per_min']}, "
                f"crash/1000 {line['crash_cars_per_1000']}")
    log(f"evalset: {len(lines)} lines in {time.time() - t0:.0f} s" + (f" -> {out}" if out else ""))
    return lines
