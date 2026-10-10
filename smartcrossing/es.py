"""OpenAI-style evolution strategies for the shared driving policy.

- antithetic noise pairs, centred rank fitness shaping, Adam, L2 weight decay
- common random numbers: within a worker all members drive on the same S traffic
  streams (one Sim with B = (pop + 1) * S envs, member = b // S, stream = b % S);
  the extra member is the unperturbed centre, used only for logging
- 4 worker processes; each owns its own scenarios (and possibly its own demand,
  drawn from the curriculum range) and evaluates the whole population on them
- curriculum: the demand ceiling rises when the centre policy is safe enough

  poetry run python -m smartcrossing.es --name x --gens 300 [--no-signal] ...
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .policy import default_in_scale, forward, init_params, layer_sizes, load_in_scale, load_npz, n_params, save_npz
from .sim import Sim, SimConfig, rot

STAT_KEYS = ("arrived", "spawned", "exits", "wrong_exits", "crashes", "offroad", "timeouts")


ARM_U = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])  # canonical arm directions


def geo_dist(sim: Sim) -> np.ndarray:
    """Route-aware distance to the target gate (no lanes): in the target arm the
    remaining arm length; in the box the distance to the target mouth centre;
    in any other arm via that arm's mouth centre. Unlike the euclidean gate
    distance it does not reward cutting through the curb corners."""
    cfg = sim.cfg
    c0 = cfg.W + cfg.Rc
    co, si = rot(-sim.arm)
    cx = co * sim.x - si * sim.y
    cy = si * sim.x + co * sim.y
    ut = ARM_U[sim.man]  # canonical target arm index == manoeuvre
    in_box = (np.abs(cx) <= c0) & (np.abs(cy) <= c0)
    a = np.argmax(np.stack([-cy, cx, cy, -cx], -1), -1)
    ua = ARM_U[a]
    d_target_arm = cfg.L - (cx * ut[..., 0] + cy * ut[..., 1])
    d_box = np.hypot(cx - c0 * ut[..., 0], cy - c0 * ut[..., 1]) + cfg.L - c0
    d_other = (
        np.hypot(cx - c0 * ua[..., 0], cy - c0 * ua[..., 1])
        + c0 * np.hypot(ua[..., 0] - ut[..., 0], ua[..., 1] - ut[..., 1])
        + cfg.L
        - c0
    )
    return np.where(in_box, d_box, np.where(a == sim.man, d_target_arm, d_other))


def rollout(cfg_d: dict, members: np.ndarray, sizes, n_scen: int, seconds: float, seed: int, demand: float, in_scale=None, r_geo=0.0, r_queue=0.0):
    """Evaluate members (P, D) on n_scen shared traffic streams.
    Returns fitness (P,) summed over streams and stats dict of (P,) arrays."""
    cfg = SimConfig(**{**cfg_d, "demand": demand})
    P = len(members)
    sim = Sim(cfg, P * n_scen, seed=seed, n_scenarios=n_scen)
    env_params = np.repeat(members, n_scen, axis=0)  # env b -> member b // n_scen
    total = np.zeros(sim.B)
    mute = not cfg.signal_enabled
    prev_d, prev_uid = np.zeros(sim.alive.shape), sim.uid.copy()
    for _ in range(int(round(seconds / cfg.dt))):
        a = forward(env_params, sim.observe(), sizes, in_scale)
        if mute:
            a[..., 2:] = 0.0
        total += sim.step(a)
        total -= r_queue * cfg.dt * sim.queue.sum(1)  # cars kept from entering
        if r_geo:
            d = geo_dist(sim)
            same = sim.alive & (sim.uid == prev_uid)
            total += r_geo * np.where(same, prev_d - d, 0.0).sum(1)
            prev_d, prev_uid = d, sim.uid.copy()
    fit = total.reshape(P, n_scen).sum(1)
    stats = {k: getattr(sim.stats, k).reshape(P, n_scen).sum(1) for k in STAT_KEYS}
    stats["alive_end"] = sim.alive.sum(1).reshape(P, n_scen).sum(1)
    return fit, stats


def _worker(args):
    return rollout(*args)


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"


def summarize_stats(st: dict, idx) -> dict:
    s = {k: float(np.sum(v[idx])) for k, v in st.items()}
    fin = s["exits"] + s["wrong_exits"] + s["crashes"] + s["offroad"] + s["timeouts"]
    return {
        "exit_ok": s["exits"] / max(fin, 1),
        "crash_rate": s["crashes"] / max(fin, 1),
        "offroad_rate": s["offroad"] / max(fin, 1),
        "wrong_rate": s["wrong_exits"] / max(fin, 1),
        "finished": fin,
        "spawned": s["spawned"],
        "arrived": s["arrived"],
    }


class Adam:
    def __init__(self, dim, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


def centered_ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--gens", type=int, default=300)
    p.add_argument("--pop", type=int, default=48, help="number of perturbed members (even)")
    p.add_argument("--scen", type=int, default=2, help="traffic streams per worker")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--wd", type=float, default=0.005)
    p.add_argument("--hidden", default="32,32")
    p.add_argument("--demand", type=float, default=0.15, help="final (target) demand per arm")
    p.add_argument("--demand-start", type=float, default=0.03)
    p.add_argument("--demand-step", type=float, default=1.25, help="multiplicative raise of the ceiling")
    p.add_argument("--safe-bad", type=float, default=0.05, help="raise demand when crash+offroad rate below")
    p.add_argument("--safe-exit", type=float, default=0.85, help="... and exit_ok above")
    p.add_argument("--signal", dest="signal", action="store_true", default=True)
    p.add_argument("--no-signal", dest="signal", action="store_false")
    p.add_argument("--cfg", default="", help="JSON SimConfig overrides (reward weights etc.)")
    p.add_argument("--init", default="", help="start from this .npz")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--r-geo", type=float, default=0.0, help="route-aware progress reward per metre (es-side)")
    p.add_argument("--r-queue", type=float, default=0.0, help="penalty per queued car per second (es-side)")
    p.add_argument("--no-in-scale", dest="in_scale", action="store_false", default=True)
    a = p.parse_args(argv)

    out = Path("runs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(n_slots=16, signal_enabled=a.signal)
    if a.cfg:
        cfg = SimConfig(**{**cfg.__dict__, **json.loads(a.cfg)})
    cfg_d = dict(cfg.__dict__)
    sizes = layer_sizes(cfg, tuple(int(h) for h in a.hidden.split(",")))
    D = n_params(sizes)
    rng = np.random.default_rng(a.seed)
    theta = init_params(sizes, rng)
    in_scale = default_in_scale(cfg) if a.in_scale else None
    if a.init:
        theta, s0, _ = load_npz(a.init)
        assert s0 == sizes
        old = load_in_scale(a.init)
        old = np.ones(sizes[0]) if old is None else old
        new = in_scale if in_scale is not None else np.ones(sizes[0])
        # keep the network function identical under the new input scaling
        W1 = theta[: sizes[0] * sizes[1]].reshape(sizes[0], sizes[1])
        theta[: sizes[0] * sizes[1]] = (W1 * (old / new)[:, None]).ravel()
    opt = Adam(D, a.lr)
    (out / "args.json").write_text(json.dumps({**vars(a), "sizes": sizes, "cfg": cfg_d}, indent=1))
    print(f"params {D}, sizes {sizes}", flush=True)

    ceiling = a.demand_start
    best_eval = -np.inf
    half = a.pop // 2
    pool = mp.get_context("fork").Pool(a.workers, initializer=_init_worker)
    log = open(out / "log.jsonl", "a")
    try:
        for gen in range(a.gens):
            t0 = time.time()
            eps = rng.standard_normal((half, D))
            eps = np.concatenate([eps, -eps])
            members = np.concatenate([theta + a.sigma * eps, theta[None]])
            # worker 0 always at the ceiling, others spread below it (avoid forgetting)
            demands = [ceiling] + list(ceiling * rng.uniform(0.4, 1.0, a.workers - 1))
            seeds = rng.integers(1 << 30, size=a.workers)
            jobs = [(cfg_d, members, sizes, a.scen, a.seconds, int(seeds[w]), float(demands[w]), in_scale, a.r_geo, a.r_queue) for w in range(a.workers)]
            res = pool.map(_worker, jobs)
            fit = sum(r[0] for r in res)
            st = {k: sum(r[1][k] for r in res) for k in res[0][1]}
            f = fit[: a.pop]
            shaped = centered_ranks(f)
            grad = (shaped @ eps) / (a.pop * a.sigma)
            theta = theta + opt.step(grad - a.wd * theta)  # ascent with weight decay
            centre = summarize_stats(st, [a.pop])
            popst = summarize_stats(st, slice(0, a.pop))
            raised = False
            if (
                centre["crash_rate"] + centre["offroad_rate"] < a.safe_bad
                and centre["exit_ok"] > a.safe_exit
                and ceiling < a.demand
            ):
                ceiling = min(a.demand, ceiling * a.demand_step)
                raised = True
            rec = {
                "gen": gen,
                "ceiling": round(ceiling, 4),
                "fit_mean": float(f.mean()),
                "fit_max": float(f.max()),
                "fit_centre": float(fit[a.pop]),
                "centre": {k: round(v, 4) for k, v in centre.items()},
                "pop": {k: round(v, 4) for k, v in popst.items()},
                "theta_norm": float(np.linalg.norm(theta)),
                "raised": raised,
                "wall_s": round(time.time() - t0, 2),
            }
            if a.eval_every and (gen % a.eval_every == 0 or gen == a.gens - 1):
                ev = evaluate(pool, cfg_d, theta, sizes, a.seconds, a.workers, a.demand, in_scale, a.r_geo, a.r_queue)
                rec["eval"] = ev
                if ev["score"] > best_eval:
                    best_eval = ev["score"]
                    save_npz(out / "best.npz", theta, sizes, cfg_d, in_scale, gen=gen, eval=ev)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            save_npz(out / "current.npz", theta, sizes, cfg_d, in_scale, gen=gen, ceiling=ceiling)
            print(json.dumps(rec), flush=True)
    finally:
        pool.close()
        log.close()


EVAL_DEMANDS = (0.05, 0.15)


def evaluate(pool, cfg_d, theta, sizes, seconds, workers, target, in_scale=None, r_geo=0.0, r_queue=0.0):
    """Fixed-seed evaluation of the centre on low and target demand.
    score = sum of team reward (comparable across generations)."""
    demands = sorted(set(EVAL_DEMANDS) | {target})
    jobs = [(cfg_d, theta[None], sizes, 4, seconds, 777 + i, d, in_scale, r_geo, r_queue) for i, d in enumerate(demands)]
    res = pool.map(_worker, jobs)
    ev = {"score": round(float(sum(r[0][0] for r in res)), 3)}
    for d, (f, st) in zip(demands, res):
        s = summarize_stats(st, [0])
        ev[f"d{d:g}"] = {k: round(s[k], 3) for k in ("exit_ok", "crash_rate", "offroad_rate", "wrong_rate", "finished")}
    return ev


if __name__ == "__main__":
    main()
