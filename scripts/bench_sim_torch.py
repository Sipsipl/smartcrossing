"""Rollout throughput: numpy Sim vs torch TorchSim (cpu / cuda, float32 / float64).

One timed "step" = what a PPO rollout worker does per step: observe, actor forward
(the trained 59-64-64-4 MLP, results/wide_s1ft2_ckpt2650.npz) + Gaussian sampling + tanh,
critic forward (63-64-64-1) on obs + privileged features, Sim.step. numpy uses ppo.py's
code path (only occupied slots go through the MLPs); torch runs everything densely on
the device (no host syncs besides the FIFO-growth check); "-graph" backends replay the whole
step as one CUDA graph. Roads are filled by a warm-up
before timing; cuda is synchronised around the timed block.

  poetry run python scripts/bench_sim_torch.py                      # all backends, B = 64 .. 16384
  poetry run python scripts/bench_sim_torch.py --backends numpy --sizes 64,256 --out runs/bench.json

torch is not a declared dependency (installed manually in the local venv).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # as in the PPO workers (numpy single-threaded)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartcrossing import ppo  # noqa: E402
from smartcrossing.policy import load_in_scale, load_npz  # noqa: E402
from smartcrossing.sim import Sim, SimConfig, act_dim, obs_dim  # noqa: E402

CKPT = ROOT / "results" / "wide_s1ft2_ckpt2650.npz"
LOGSTD = -1.0


def make_cfg(demand):
    return SimConfig(n_slots=24, r_time=0.02, demand_range=(0.3 * demand, demand))  # as ppo.py late in the curriculum


def nets(cfg):
    actor_p, a_sizes, _ = load_npz(CKPT)
    in_scale = load_in_scale(CKPT)
    assert a_sizes[0] == obs_dim(cfg) and a_sizes[-1] == act_dim(cfg)
    c_sizes = [obs_dim(cfg) + ppo.N_PRIV, 64, 64, 1]
    critic = ppo.MLP(c_sizes, np.random.default_rng(0))
    return actor_p, a_sizes, critic.p, c_sizes, in_scale


# ---------------------------------------------------------------- numpy (ppo.Worker.rollout's per-step work)
def numpy_runner(cfg, B, seed):
    actor_p, a_sizes, critic_p, c_sizes, in_scale = nets(cfg)
    sim = Sim(cfg, B, seed=seed)
    rng = np.random.default_rng(seed)
    std = np.exp(LOGSTD)
    A = act_dim(cfg)

    def step():
        obs = sim.observe()
        bi, ni = np.nonzero(sim.alive)
        pr = ppo.privileged(sim)
        act = np.zeros((B, cfg.n_slots, A))
        if len(bi):
            o = obs[bi, ni] * in_scale
            mu = ppo.mlp_forward(actor_p, a_sizes, o)
            u = mu + std * rng.standard_normal(mu.shape)
            ppo.mlp_forward(critic_p, c_sizes, np.concatenate([o, pr[bi, ni]], 1))
            act[bi, ni] = np.tanh(u)
        sim.step(act)

    return step, (lambda: float(sim.alive.sum()) / B), (lambda: None)


# ---------------------------------------------------------------- torch
def torch_runner(cfg, B, seed, device, dtype, graph=False):
    import torch

    from smartcrossing.sim_torch import TorchSim

    actor_p, a_sizes, critic_p, c_sizes, in_scale = nets(cfg)
    sim = TorchSim(cfg, B, seed=seed, device=device, dtype=dtype, rng="torch")
    kw = dict(device=device, dtype=dtype)

    def layers(p, sizes):
        out, i = [], 0
        for a, b in zip(sizes[:-1], sizes[1:]):
            out.append((torch.tensor(p[i : i + a * b].reshape(a, b), **kw), torch.tensor(p[i + a * b : i + a * b + b], **kw)))
            i += a * b + b
        return out

    A_l, C_l = layers(actor_p, a_sizes), layers(critic_p, c_sizes)
    scale = torch.tensor(in_scale, **kw)
    std = float(np.exp(LOGSTD))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1)
    N = cfg.n_slots

    def mlp(Ls, h):
        for k, (W, b) in enumerate(Ls):
            h = torch.addmm(b, h, W)
            if k < len(Ls) - 1:
                h = torch.tanh(h)
        return h

    def step():
        with torch.no_grad():
            obs = sim.observe()
            o = (obs * scale).reshape(B * N, -1)
            mu = mlp(A_l, o)
            u = mu + std * torch.randn(mu.shape, generator=gen, **kw)
            dem = sim.demand if sim.demand is not None else torch.full((B,), cfg.demand, **kw)
            pr = torch.stack([
                sim.age / cfg.max_age,
                (sim.queue.sum(1).to(dtype) / 10.0)[:, None].expand(B, N),
                (sim.alive.sum(1).to(dtype) / N)[:, None].expand(B, N),
                (dem * 5.0)[:, None].expand(B, N),
            ], -1).reshape(B * N, -1)
            mlp(C_l, torch.cat([o, pr], 1))
            act = torch.where(sim.alive[..., None], torch.tanh(u).reshape(B, N, -1), 0.0)
            sim.step(act)

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    if graph:  # one CUDA-graph replay per step (observe + MLPs + step), no Python/launch overhead
        for _ in range(5):
            step()
        replay = sim.cuda_graph(step, generators=(gen,))
        return replay, (lambda: float(sim.alive.sum()) / B), sync
    return step, (lambda: float(sim.alive.sum()) / B), sync


def bench(backend, B, a):
    cfg = make_cfg(a.demand)
    if backend == "numpy":
        step, cars, sync = numpy_runner(cfg, B, a.seed)
    else:
        import torch

        _, dev, prec, *mode = backend.split("-")
        if dev == "cuda" and not torch.cuda.is_available():
            return None
        step, cars, sync = torch_runner(cfg, B, a.seed, dev, torch.float32 if prec == "f32" else torch.float64,
                                        graph=mode == ["graph"])
    for _ in range(a.warmup):
        step()
    sync()
    n, t0 = 0, time.perf_counter()
    while True:
        step()
        n += 1
        if n >= a.min_steps:
            sync()
            if time.perf_counter() - t0 >= a.min_time or n >= a.max_steps:
                break
    dt = time.perf_counter() - t0
    return {"backend": backend, "B": B, "steps": n, "s_per_step": dt / n, "env_steps_per_s": B * n / dt,
            "cars_per_env": cars(), "car_steps_per_s": B * n / dt * cars()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backends", default="numpy,torch-cpu-f64,torch-cpu-f32,torch-cuda-f64,torch-cuda-f32,"
                   "torch-cuda-f64-graph,torch-cuda-f32-graph")
    p.add_argument("--sizes", default="64,256,1024,4096,16384")
    p.add_argument("--numpy-max", type=int, default=4096, help="largest B for numpy (slow)")
    p.add_argument("--cpu-max", type=int, default=4096, help="largest B for torch-cpu")
    p.add_argument("--threads", type=int, default=1, help="torch cpu threads")
    p.add_argument("--demand", type=float, default=0.3, help="curriculum ceiling (demand ~ U(0.3 d, d))")
    p.add_argument("--warmup", type=int, default=150, help="steps before timing (fills the roads)")
    p.add_argument("--min-steps", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--min-time", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="write the results as JSON")
    a = p.parse_args(argv)
    try:
        import torch

        torch.set_num_threads(a.threads)
    except ImportError:
        pass
    rows = []
    for backend in a.backends.split(","):
        for B in (int(s) for s in a.sizes.split(",")):
            if (backend == "numpy" and B > a.numpy_max) or ("-cpu-" in backend and B > a.cpu_max):
                continue
            try:
                r = bench(backend, B, a)
            except RuntimeError as e:  # e.g. CUDA out of memory
                print(f"{backend:16s} B={B:6d}  failed: {str(e).splitlines()[0]}", flush=True)
                if "cuda" in backend:
                    import torch

                    torch.cuda.empty_cache()
                continue
            if r is None:
                continue
            rows.append(r)
            print(f"{backend:16s} B={B:6d}  {r['s_per_step'] * 1e3:9.2f} ms/step  {r['env_steps_per_s']:10.0f} env-steps/s"
                  f"  {r['car_steps_per_s']:11.0f} car-steps/s  ({r['cars_per_env']:.1f} cars/env)", flush=True)
    base = next((r for r in rows if r["backend"] == "numpy" and r["B"] == 64), None)
    if base:
        print("\nspeed-up vs numpy B=64 (env-steps/s, one process):")
        for r in rows:
            print(f"  {r['backend']:16s} B={r['B']:6d}  x{r['env_steps_per_s'] / base['env_steps_per_s']:.1f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
