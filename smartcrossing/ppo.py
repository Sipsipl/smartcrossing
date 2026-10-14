"""PPO with parameter sharing: every car of B parallel environments is a sample of one
shared driving policy, trained on its own (per-car) reward.

Rollouts and the default update are pure numpy (hand-written tanh-MLP backward pass).
`--update-device cpu|cuda` runs the update in torch instead (ppo_torch.py, optional `gpu`
poetry group; same parameters, ~19x faster on the GPU for dense-traffic batches).
`--rollout-device cuda` also runs the rollouts on the GPU (rollout_torch.py: one
TorchSim with `--rollout-envs` envs, CUDA graph, buffers + GAE on the device, batch handed to
the torch update without a host round trip; traffic and action noise from torch generators).
Evaluation (eval_job) always uses the numpy Sim (paired traffic) in the run's world.
Run tracking: args.json (+ git commit / dirty flag; a resumed run writes args.<n>.json) and the
checkpoints' extra carry the commit; scripts/runs_index.py lists the runs.

- actor: tanh MLP obs -> mu (linear output), Gaussian in pre-squash space with a
  state-independent log-std; the car executes tanh(u). The deterministic policy
  tanh(mu) is exactly the existing policy.py / viewer format (tanh output layer,
  same in_scale), so checkpoints load with PolicyDriver, cli bench/render, export_json.
- `--arch set`: actor (and with `--critic-arch` same/set the critic) is the neighbour set
  encoder of policy_set.py (shared per-neighbour MLP, attention / mean / max pooling, MLP trunk);
  needs the torch update (`--update-device cpu|cuda`, autograd); numpy rollout workers and
  checkpoints (arch field, policy.save_npz) handle both architectures.
- critic: separate tanh MLP on the scaled obs plus a few privileged features (car age,
  queue length, cars on the road, demand of the env); not exported.
- rollouts: persistent worker processes, each with its own vectorised Sim; GAE per car
  trajectory on the dense (T, B, N) slot grid: a slot's chain is cut where the car was
  removed (Sim.done_kind) - a new car in the same slot (new uid) starts a new
  trajectory; at the rollout end / env reset alive cars are bootstrapped with V.
- reward per car: its own reward (Sim.car_reward).
- curriculum: per-env demand drawn from [demand_lo * ceiling, ceiling]; the ceiling rises
  when the training crash+offroad rate is low.

  poetry run python -m smartcrossing.ppo --name ppo1 --iters 400 [--no-signal]
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # small matmuls; avoid thread contention with workers

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import policy_set
from .policy import default_in_scale, layer_sizes, n_params, save_npz
from .ppo_common import LOG2PI, N_DONE, N_PRIV, PRIV_DEMAND_SCALE, PRIV_QUEUE_SCALE, git_info
from .sim import DONE_CRASH, DONE_EXIT, DONE_OFFROAD, DONE_TIMEOUT, DONE_WRONG, Sim, SimConfig, act_dim, obs_dim

BATCH_KEYS = ("obs", "priv", "u", "logp", "adv", "ret", "val")  # per car-step arrays of a rollout


# ---------------------------------------------------------------- tanh MLP with backprop
class MLP:
    """tanh hidden layers, linear output. Params kept as a flat float64 vector in the
    policy.py layout (per layer W (in, out) row-major, then b)."""

    def __init__(self, sizes, rng=None, out_scale=1.0):
        self.sizes = list(sizes)
        rng = np.random.default_rng(rng)
        parts = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            s = 1.0 / np.sqrt(a) * (out_scale if i == len(sizes) - 2 else 1.0)
            parts += [rng.normal(0, s, a * b), np.zeros(b)]
        self.p = np.concatenate(parts)

    def layers(self, p=None):
        p = self.p if p is None else p
        out, i = [], 0
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            out.append((p[i : i + a * b].reshape(a, b), p[i + a * b : i + a * b + b]))
            i += a * b + b
        return out

    def forward(self, x, keep=False):
        hs = [x]
        Ls = self.layers()
        h = x
        for k, (W, b) in enumerate(Ls):
            z = h @ W + b
            h = z if k == len(Ls) - 1 else np.tanh(z)
            hs.append(h)
        return (h, hs) if keep else h

    def backward(self, hs, gout):
        """Gradient of sum(gout * output) wrt the flat params."""
        Ls = self.layers()
        grads = []
        g = gout
        for k in range(len(Ls) - 1, -1, -1):
            W, _ = Ls[k]
            h_in = hs[k]
            grads.append((h_in.T @ g, g.sum(0)))
            if k > 0:
                g = (g @ W.T) * (1.0 - hs[k] ** 2)
        flat = []
        for gW, gb in reversed(grads):
            flat += [gW.ravel(), gb]
        return np.concatenate(flat)


class Adam:
    def __init__(self, dim, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.t = np.zeros(dim), np.zeros(dim), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, g):
        """Returns the update to *subtract* (descent)."""
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


class SetNet:
    """Set-encoder network (policy_set.py) with the MLP's interface for the training loop: flat
    params ``p``, ``sizes`` (checkpoint sizes [in, *hidden, out]) and ``arch``. No numpy backward:
    ppo_torch differentiates it with autograd."""

    def __init__(self, arch, sizes, rng=None, out_scale=1.0):
        self.arch, self.sizes = arch, list(sizes)
        self.p = policy_set.init_params(arch, rng, out_scale)

    def forward(self, x):
        return policy_set.forward(self.p, self.arch, x)


def net_forward(p, sizes, x, arch=None):
    """Stateless forward (workers) of an MLP (sizes) or a set encoder (arch)."""
    return policy_set.forward(p, arch, x) if arch else mlp_forward(p, sizes, x)


def mlp_forward(p, sizes, x):
    """Stateless forward (workers): tanh hidden, linear output."""
    i = 0
    n = len(sizes) - 1
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        W = p[i : i + a * b].reshape(a, b)
        bias = p[i + a * b : i + a * b + b]
        i += a * b + b
        x = x @ W + bias
        if k < n - 1:
            x = np.tanh(x)
    return x


# ---------------------------------------------------------------- rollout worker
def privileged(sim: Sim) -> np.ndarray:
    """(B, N, N_PRIV) extra critic inputs (not visible to the policy; ppo_common.PRIV_FEATURES)."""
    cfg = sim.cfg
    B, N = sim.alive.shape
    dem = np.full(B, cfg.demand) if sim.demand is None else sim.demand
    f = np.empty((B, N, N_PRIV))
    f[..., 0] = sim.age / cfg.max_age
    f[..., 1] = (sim.queue.sum(1) / PRIV_QUEUE_SCALE)[:, None]
    f[..., 2] = (sim.alive.sum(1) / N)[:, None]
    f[..., 3] = (dem * PRIV_DEMAND_SCALE)[:, None]
    return f


class Worker:
    def __init__(self, cfg_d, n_envs, seed, horizon, episode_steps, gamma, lam, in_scale, act_mask, actor_arch=None,
                 critic_arch=None):
        self.cfg_d = cfg_d
        self.a_arch, self.c_arch = actor_arch, critic_arch
        self.B = n_envs
        self.seed = seed
        self.T = horizon
        self.ep_steps = episode_steps
        self.gamma, self.lam = gamma, lam
        self.in_scale = in_scale
        self.act_mask = act_mask
        self.rng = np.random.default_rng(seed + 12345)
        self.sim = None
        self.n_resets = 0
        self.steps_in_ep = 0

    def _reset(self, d_lo, d_hi):
        cfg = SimConfig(**{**self.cfg_d, "demand_range": (d_lo, d_hi)})
        self.sim = Sim(cfg, self.B, seed=self.seed * 100003 + self.n_resets)
        self.n_resets += 1
        self.steps_in_ep = 0

    def rollout(self, actor_p, actor_sizes, logstd, critic_p, critic_sizes, d_lo, d_hi):
        if self.sim is None or self.steps_in_ep >= self.ep_steps:
            self._reset(d_lo, d_hi)
        sim = self.sim
        T, B, N = self.T, self.B, sim.cfg.n_slots
        D, A = obs_dim(sim.cfg), act_dim(sim.cfg)
        std = np.exp(logstd)
        obs_buf = np.zeros((T, B, N, D), np.float32)
        priv_buf = np.zeros((T, B, N, N_PRIV), np.float32)
        u_buf = np.zeros((T, B, N, A), np.float32)
        logp_buf = np.zeros((T, B, N), np.float32)
        val = np.zeros((T + 1, B, N))
        rew = np.zeros((T, B, N))
        done = np.zeros((T, B, N), bool)
        valid = np.zeros((T, B, N), bool)
        kinds = np.zeros(N_DONE, np.int64)
        own_rew_sum = 0.0
        for t in range(T):
            obs = sim.observe()
            al = sim.alive.copy()
            valid[t] = al
            bi, ni = np.nonzero(al)
            pr = privileged(sim)
            if len(bi):
                o = obs[bi, ni] * self.in_scale
                mu = net_forward(actor_p, actor_sizes, o, self.a_arch)
                u = mu + std * self.rng.standard_normal(mu.shape)
                lp = (-0.5 * ((u - mu) / std) ** 2 - logstd - 0.5 * LOG2PI) @ self.act_mask
                v = net_forward(critic_p, critic_sizes, np.concatenate([o, pr[bi, ni]], 1), self.c_arch)[:, 0]
                obs_buf[t, bi, ni] = obs[bi, ni]
                priv_buf[t, bi, ni] = pr[bi, ni]
                u_buf[t, bi, ni] = u
                logp_buf[t, bi, ni] = lp
                val[t, bi, ni] = v
                act = np.zeros((B, N, A))
                act[bi, ni] = np.tanh(u)
            else:
                act = np.zeros((B, N, A))
            sim.step(act)
            r = np.where(al, sim.car_reward, 0.0)
            own_rew_sum += r.sum()
            rew[t] = r
            done[t] = al & (sim.done_kind > 0)
            kinds += np.bincount(sim.done_kind[al].ravel(), minlength=N_DONE)
            self.steps_in_ep += 1
        # bootstrap values for cars still on the road
        obs = sim.observe()
        bi, ni = np.nonzero(sim.alive)
        if len(bi):
            o = obs[bi, ni] * self.in_scale
            val[T, bi, ni] = net_forward(critic_p, critic_sizes, np.concatenate([o, privileged(sim)[bi, ni]], 1),
                                         self.c_arch)[:, 0]
        # GAE per slot chain (chain cut at removals)
        adv = np.zeros((T, B, N))
        last = np.zeros((B, N))
        g, gl = self.gamma, self.gamma * self.lam
        for t in range(T - 1, -1, -1):
            nd = ~done[t]
            delta = rew[t] + g * val[t + 1] * nd - val[t]
            last = np.where(valid[t], delta + gl * nd * last, 0.0)
            adv[t] = last
        ret = adv + val[:T]
        m = valid
        return {
            "obs": obs_buf[m],
            "priv": priv_buf[m],
            "u": u_buf[m],
            "logp": logp_buf[m],
            "adv": adv[m].astype(np.float32),
            "ret": ret[m].astype(np.float32),
            "val": val[:T][m].astype(np.float32),
            "kinds": kinds,
            "car_steps": int(m.sum()),
            "own_rew": own_rew_sum,
            "queued": float(sim.queue.sum()) / B,
            "demand_mean": float(np.mean(sim.demand)),
            "obstacles": float(sim.obstacle.sum()) / max(int(sim.obst_env.sum()), 1),  # standing, per obstacle env
            "obstacle_hits": float(sim.stats.obstacle_hits.sum()),  # in the current episode
        }


def _worker_main(conn, kw):
    os.environ["OMP_NUM_THREADS"] = "1"
    w = Worker(**kw)
    while True:
        cmd, args = conn.recv()
        if cmd == "close":
            break
        conn.send(w.rollout(*args))


# ---------------------------------------------------------------- PPO update
def ppo_update(actor, logstd, critic, opt_a, opt_c, data, a, act_mask, in_scale, rng):
    n = len(data["adv"])
    obs = data["obs"].astype(np.float64) * in_scale
    cin = np.concatenate([obs, data["priv"]], 1)
    u = data["u"].astype(np.float64)
    old_lp = data["logp"].astype(np.float64)
    adv_all = data["adv"].astype(np.float64)
    ret = data["ret"].astype(np.float64)
    nA = len(logstd)
    stats = {"kl": [], "clipfrac": [], "vloss": [], "pg": []}
    mb = int(np.ceil(n / a.minibatches))
    stop = False
    for ep in range(a.epochs):
        perm = rng.permutation(n)
        for s in range(0, n, mb):
            idx = perm[s : s + mb]
            m = len(idx)
            adv = adv_all[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            mu, hs = actor.forward(obs[idx], keep=True)
            std = np.exp(logstd)
            z = (u[idx] - mu) / std
            lp = (-0.5 * z**2 - logstd - 0.5 * LOG2PI) @ act_mask
            ratio = np.exp(lp - old_lp[idx])
            clip_hi, clip_lo = 1 + a.clip, 1 - a.clip
            active = ~(((adv > 0) & (ratio > clip_hi)) | ((adv < 0) & (ratio < clip_lo)))
            # loss = -mean(min(ratio A, clip(ratio) A)) - ent * sum(logstd)
            g_lp = -(adv * ratio * active) / m  # dLoss / dlogp
            g_mu = g_lp[:, None] * (z / std) * act_mask
            g_logstd = (g_lp[:, None] * (z**2 - 1.0) * act_mask).sum(0) - a.ent * act_mask
            ga = actor.backward(hs, g_mu)
            gnorm = np.linalg.norm(ga)
            if gnorm > a.max_grad:
                ga *= a.max_grad / gnorm
            actor.p -= opt_a.step(ga)
            logstd -= a.lr_std * np.clip(g_logstd, -1, 1)
            np.clip(logstd, a.logstd_min, 0.5, out=logstd)
            # critic
            v, hc = critic.forward(cin[idx], keep=True)
            err = v[:, 0] - ret[idx]
            gc = critic.backward(hc, (err / m)[:, None])
            gn = np.linalg.norm(gc)
            if gn > a.max_grad:
                gc *= a.max_grad / gn
            critic.p -= opt_c.step(gc)
            kl = float(np.mean((ratio - 1) - np.log(ratio)))
            stats["kl"].append(kl)
            stats["clipfrac"].append(float(np.mean(np.abs(ratio - 1) > a.clip)))
            stats["vloss"].append(float(np.mean(err**2)))
            stats["pg"].append(float(np.mean(adv * ratio * active)))
            if a.target_kl and kl > 1.5 * a.target_kl:
                stop = True
                break
        if stop:
            break
    out = {k: round(float(np.mean(v)), 5) for k, v in stats.items()}
    out["epochs_done"] = ep + 1
    return out




# ---------------------------------------------------------------- eval (subprocess)
EVAL_DEMANDS = (0.05, 0.15)
EVAL_ENVS, EVAL_SECONDS, EVAL_SEED = 8, 120.0, 1000  # cli bench settings
RENDER_DEMAND, RENDER_SECONDS = 0.15, 60.0


def eval_job(ckpt: str, out_path: str, it: int, demands, envs: int, seconds: float, render_dir: str | None,
             cfg_d: dict | None = None):
    """Bench metrics of the deterministic policy (cli bench settings) in the run's world (cfg_d: the
    training SimConfig; fixed demand per bench, rewards do not matter) plus the stop gate (gates.py,
    ~1 s); appends one JSON line."""
    from .cli import run
    from .gates import stop_gate
    from .metrics import summarize

    world = {**(cfg_d or {}), "demand_range": None}
    rec = {"iter": it, "ckpt": ckpt}
    for d in demands:
        sim, _ = run(f"policy:{ckpt}", SimConfig(**{**world, "demand": d}), envs, seconds, EVAL_SEED)
        s = summarize(sim, seconds)
        rec[f"d{d:g}"] = {k: round(s[k], 3) for k in ("throughput_per_min", "exit_ok_rate", "crash_cars_per_1000",
                                                      "crash_cars", "offroad_cars", "timeouts", "delay_mean_s", "queued_end")}
    g, _ = stop_gate(f"policy:{ckpt}", world)
    rec["gate"] = {k: g[k] for k in ("stop_rate_feasible", "safe_rate_feasible", "stop_rate", "crash", "offroad")}
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec) + "\n")
    if render_dir:
        render_cfg = json.dumps({**world, "demand": RENDER_DEMAND})
        subprocess.run([sys.executable, "-m", "smartcrossing.cli", "render", "--driver", f"policy:{ckpt}", "--demand",
                        str(RENDER_DEMAND), "--cfg", render_cfg, "--seconds", str(RENDER_SECONDS), "--out", render_dir],
                       capture_output=True)


# ---------------------------------------------------------------- arguments and state
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--max-minutes", type=float, default=0.0,
                   help="stop after the iteration that crosses this training wall time (0 = off; the final "
                   "iteration is checkpointed and evaluated like the last of --iters)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--envs", type=int, default=24, help="envs per worker")
    p.add_argument("--horizon", type=int, default=128, help="steps per rollout")
    p.add_argument("--episode", type=float, default=90.0, help="seconds between env resets")
    p.add_argument("--hidden", default="64,64", help="hidden layers of the MLP (--arch set: of the trunk)")
    p.add_argument("--arch", choices=("mlp", "set"), default="mlp",
                   help="actor: MLP on the flat obs (default) or the neighbour set encoder (policy_set.py; needs "
                        "--update-device cpu|cuda)")
    p.add_argument("--critic-arch", choices=("same", "mlp", "set"), default="same", help="critic architecture")
    p.add_argument("--set-phi", default="32,32", help="set encoder: per-neighbour MLP layers")
    p.add_argument("--set-pool", choices=policy_set.POOLS, default="attn", help="set encoder: pooling")
    p.add_argument("--set-heads", type=int, default=4, help="set encoder (attn): heads")
    p.add_argument("--set-dk", type=int, default=8, help="set encoder (attn): key / value dims per head")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=1e-3)
    p.add_argument("--lr-std", type=float, default=3e-3)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--ent", type=float, default=0.0, help="entropy bonus per log-std dim")
    p.add_argument("--target-kl", type=float, default=0.03)
    p.add_argument("--max-grad", type=float, default=1.0)
    p.add_argument("--logstd-init", type=float, default=-0.5)
    p.add_argument("--logstd-min", type=float, default=-2.5)
    p.add_argument("--acc-init", type=float, default=0.3, help="initial mean accel action (random cars otherwise brake to a stop)")
    p.add_argument("--demand", type=float, default=0.3, help="final curriculum ceiling")
    p.add_argument("--demand-start", type=float, default=0.05)
    p.add_argument("--demand-lo", type=float, default=0.3, help="env demand ~ U(lo * ceiling, ceiling)")
    p.add_argument("--demand-step", type=float, default=1.15)
    p.add_argument("--safe-bad", type=float, default=0.03, help="raise ceiling when crash+offroad per finished car below")
    p.add_argument("--safe-exit", type=float, default=0.9)
    p.add_argument("--signal", dest="signal", action="store_true", default=True)
    p.add_argument("--no-signal", dest="signal", action="store_false")
    p.add_argument("--cfg", default="", help="JSON SimConfig overrides")
    p.add_argument("--init", default="", help="resume from a full checkpoint (state.npz)")
    p.add_argument("--init-policy", default="", help="warm-start the actor from a policy .npz of an older interface "
                   "(policy_compat.transplant_actor); critic fresh")
    p.add_argument("--init-bc", default="", help="start the actor by behaviour cloning this MLP "
                   "checkpoint (bc.py; --arch set or mlp); critic fresh")
    p.add_argument("--bc-steps", type=int, default=3000, help="Adam steps of the behaviour cloning")
    p.add_argument("--bc-seconds", type=float, default=120.0, help="behaviour cloning: teacher drive (32 envs)")
    p.add_argument("--critic-warmup", type=int, default=0, help="first N iterations of this run train only the critic")
    p.add_argument("--reset-logstd", action="store_true", help="with --init: restart log-std at --logstd-init")
    p.add_argument("--set-ceiling", type=float, default=0.0, help="with --init: override the stored curriculum ceiling")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--render-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-inputs", default="", help="comma-separated obs indices whose in_scale is set to 0 "
                   "(input ablation, e.g. 5,6,7 = manoeuvre one-hot; stored in the checkpoint)")
    p.add_argument("--update-device", choices=("numpy", "cpu", "cuda"), default="numpy",
                   help="PPO update: numpy (default) or torch on cpu/cuda (ppo_torch.py; rollouts stay numpy)")
    p.add_argument("--update-dtype", choices=("float32", "float64"), default="float32", help="torch update dtype")
    p.add_argument("--rollout-device", choices=("numpy", "cpu", "cuda"), default="numpy",
                   help="rollouts: numpy worker processes (default) or one TorchSim on cpu/cuda (rollout_torch.py; "
                        "implies the torch update on the same device; --workers is then unused)")
    p.add_argument("--rollout-envs", type=int, default=0, help="torch rollout: envs (0 = workers x envs)")
    p.add_argument("--rollout-dtype", choices=("float32", "float64"), default="float32", help="torch rollout dtype")
    p.add_argument("--runs-dir", default="runs", help="parent folder of the run folder")
    return p.parse_args(argv)


@dataclass
class TrainState:
    """Everything the training loop updates or needs about the networks (see build_state)."""

    cfg: SimConfig
    in_scale: np.ndarray
    act_mask: np.ndarray
    a_sizes: list
    c_sizes: list
    rng: np.random.Generator
    actor: MLP
    critic: MLP
    logstd: np.ndarray
    opt_a: Adam
    opt_c: Adam
    ceiling: float
    it0: int = 0
    total_steps: int = 0
    a_arch: dict | None = None  # policy_set arch of a set-encoder actor / critic (None = MLP)
    c_arch: dict | None = None

    @property
    def cfg_d(self) -> dict:
        return dict(self.cfg.__dict__)


def build_state(a) -> TrainState:
    """World config, fresh networks and optimisers, then --init-policy / --init applied."""
    cfg = SimConfig(n_slots=24, signal_enabled=a.signal, r_time=0.02)
    if a.cfg:
        cfg = SimConfig(**{**cfg.__dict__, **json.loads(a.cfg)})
    in_scale = default_in_scale(cfg)
    if a.zero_inputs:
        in_scale[[int(i) for i in a.zero_inputs.split(",")]] = 0.0
    A = act_dim(cfg)
    act_mask = np.ones(A)
    if not cfg.signal_enabled:
        act_mask[2:] = 0.0  # signal outputs have no effect: keep them out of the likelihood
    hidden = tuple(int(h) for h in a.hidden.split(","))
    a_sizes = layer_sizes(cfg, hidden)
    c_sizes = [obs_dim(cfg) + N_PRIV, *hidden, 1]
    rng = np.random.default_rng(a.seed)
    arch = getattr(a, "arch", "mlp")
    c_arch = arch if getattr(a, "critic_arch", "same") == "same" else a.critic_arch
    a_arch = c_arch_d = None
    if arch == "set" or c_arch == "set":
        kw = dict(hidden=hidden, phi=tuple(int(h) for h in a.set_phi.split(",")), pool=a.set_pool, heads=a.set_heads,
                  dk=a.set_dk, dv=a.set_dk)
        a_arch = policy_set.make_arch(cfg, **kw) if arch == "set" else None
        c_arch_d = policy_set.make_arch(cfg, n_extra=N_PRIV, n_out=1, **kw) if c_arch == "set" else None
    actor = SetNet(a_arch, a_sizes, rng, out_scale=0.01) if a_arch else MLP(a_sizes, rng, out_scale=0.01)
    critic = SetNet(c_arch_d, c_sizes, rng, out_scale=1.0) if c_arch_d else MLP(c_sizes, rng, out_scale=1.0)
    actor.p[-A] = np.arctanh(a.acc_init)  # output bias: initial mean acceleration action (last block in both layouts)
    st = TrainState(cfg, in_scale, act_mask, a_sizes, c_sizes, rng, actor, critic, np.full(A, a.logstd_init),
                    Adam(len(actor.p), a.lr), Adam(len(critic.p), a.lr_critic), a.demand_start, a_arch=a_arch,
                    c_arch=c_arch_d)
    if a.init_policy and a_arch:
        raise ValueError("--init-policy transplants an MLP actor; it cannot start a set-encoder actor (--init-bc)")
    if getattr(a, "init_bc", ""):
        from .bc import bc_actor

        dev = a.rollout_device if a.rollout_device != "numpy" else a.update_device
        t_bc = time.time()
        actor.p = bc_actor(a.init_bc, cfg, a_arch or a_sizes, in_scale, actor.p, steps=a.bc_steps, seconds=a.bc_seconds,
                           device="cpu" if dev == "numpy" else dev, seed=a.seed)
        print(f"bc: {time.time() - t_bc:.1f} s", flush=True)
    if a.init_policy:
        from .policy_compat import transplant_actor

        actor.p = transplant_actor(a.init_policy, cfg, in_scale, a_sizes)
    if a.init:
        z = np.load(a.init)
        actor.p, critic.p, st.logstd = z["actor"].copy(), z["critic"].copy(), z["logstd"].copy()
        st.ceiling = float(z["ceiling"]) if "ceiling" in z.files else st.ceiling
        st.it0 = int(z["iter"]) + 1 if "iter" in z.files else 0
        st.total_steps = int(z["total_steps"]) if "total_steps" in z.files else 0
        if a.reset_logstd:
            st.logstd = np.full(A, a.logstd_init)
        if a.set_ceiling:
            st.ceiling = a.set_ceiling
    return st


def write_args(out: Path, a, st: TrainState, git: dict) -> Path:
    """args.json of the run (all arguments incl. init / init_policy, network sizes, world config,
    git commit). A resumed run (--init) keeps an existing args.json and writes args.<n>.json."""
    path = out / "args.json"
    if a.init and path.exists():
        n = 1
        while (out / f"args.{n}.json").exists():
            n += 1
        path = out / f"args.{n}.json"
    rec = {**vars(a), "actor_sizes": st.a_sizes, "critic_sizes": st.c_sizes, "actor_arch": st.a_arch,
           "critic_arch": st.c_arch, "cfg": st.cfg_d, "git": git}
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------- rollout sources
class WorkerRollouts:
    """Numpy rollouts: persistent worker processes (spawn), each with its own vectorised Sim."""

    def __init__(self, ctx, a, st: TrainState, ep_steps: int):
        self.a_sizes, self.c_sizes = st.a_sizes, st.c_sizes
        self.conns = []
        for w in range(a.workers):
            kw = dict(cfg_d=st.cfg_d, n_envs=a.envs, seed=a.seed * 1000 + w, horizon=a.horizon, episode_steps=ep_steps,
                      gamma=a.gamma, lam=a.lam, in_scale=st.in_scale, act_mask=st.act_mask, actor_arch=st.a_arch,
                      critic_arch=st.c_arch)
            c1, c2 = ctx.Pipe()
            ctx.Process(target=_worker_main, args=(c2, kw), daemon=True).start()
            self.conns.append(c1)

    def rollout(self, actor_p, logstd, critic_p, d_lo, d_hi):
        """-> (per-worker results, batch {BATCH_KEYS: numpy arrays})."""
        for c in self.conns:
            c.send(("rollout", (actor_p, self.a_sizes, logstd, critic_p, self.c_sizes, d_lo, d_hi)))
        res = [c.recv() for c in self.conns]
        return res, {k: np.concatenate([r[k] for r in res]) for k in BATCH_KEYS}

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass


class TorchRollouts:
    """Rollouts of one big TorchSim on cpu / cuda (rollout_torch.py); the batch stays on the device."""

    def __init__(self, a, st: TrainState, ep_steps: int):
        from .rollout_torch import TorchRollout

        self.ro = TorchRollout(st.cfg_d, a.rollout_envs or a.workers * a.envs, a.seed * 1000, a.horizon, ep_steps,
                               a.gamma, a.lam, st.in_scale, st.act_mask, st.a_sizes, st.c_sizes,
                               device=a.rollout_device, dtype=a.rollout_dtype, actor_arch=st.a_arch,
                               critic_arch=st.c_arch)

    def rollout(self, actor_p, logstd, critic_p, d_lo, d_hi):
        """-> ([result], batch {BATCH_KEYS: device tensors})."""
        self.ro.set_params(actor_p, logstd, critic_p)
        r = self.ro.rollout(d_lo, d_hi)
        return [r], {k: r[k] for k in BATCH_KEYS}

    def close(self):
        pass


def explained_var(data) -> float:
    """1 - Var(ret - val) / Var(ret) of the batch (numpy arrays or device tensors; population variance)."""
    ret, val = data["ret"], data["val"]
    if isinstance(ret, np.ndarray):
        ev, var_ret = np.var(ret - val), np.var(ret)
    else:
        ev = float((ret - val).double().var(correction=0))
        var_ret = float(ret.double().var(correction=0))
    return round(float(1 - ev / (var_ret + 1e-8)), 3)


# ---------------------------------------------------------------- training loop
def train_loop(a, st: TrainState, out: Path, source, update, ctx, git: dict):
    """PPO iterations it0 .. iters-1: rollout, update, curriculum, log line, checkpoints, evals."""
    actor, critic, logstd, opt_a = st.actor, st.critic, st.logstd, st.opt_a
    cfg_d = st.cfg_d
    eval_proc = None
    t_start = time.time()
    with open(out / "log.jsonl", "a", encoding="utf-8", newline="\n") as log:
        try:
            for it in range(st.it0, a.iters):
                t0 = time.time()
                res, data = source.rollout(actor.p, logstd, critic.p, a.demand_lo * st.ceiling, st.ceiling)
                t1 = time.time()
                kinds = sum(r["kinds"] for r in res)
                car_steps = sum(r["car_steps"] for r in res)
                st.total_steps += car_steps
                warm = it - st.it0 < a.critic_warmup
                if warm:  # critic warm-up: keep the actor (params, log-std, Adam state) unchanged
                    keep = (actor.p.copy(), logstd.copy(), opt_a.m.copy(), opt_a.v.copy(), opt_a.t)
                upd = update(actor, logstd, critic, opt_a, st.opt_c, data, a, st.act_mask, st.in_scale, st.rng)
                if warm:
                    actor.p[...], logstd[...], opt_a.m, opt_a.v, opt_a.t = keep
                    upd["critic_warmup"] = True
                fin = max(kinds[1:].sum(), 1)
                tr = {
                    "exit_ok": kinds[DONE_EXIT] / fin,
                    "crash": kinds[DONE_CRASH] / fin,
                    "offroad": kinds[DONE_OFFROAD] / fin,
                    "timeout": kinds[DONE_TIMEOUT] / fin,
                    "wrong": kinds[DONE_WRONG] / fin,
                    "finished": int(kinds[1:].sum()),
                }
                raised = False
                if tr["crash"] + tr["offroad"] < a.safe_bad and tr["exit_ok"] > a.safe_exit and st.ceiling < a.demand:
                    st.ceiling = min(a.demand, st.ceiling * a.demand_step)
                    raised = True
                rec = {
                    "iter": it,
                    "ceiling": round(st.ceiling, 4),
                    "raised": raised,
                    "car_steps": car_steps,
                    "total_steps": st.total_steps,
                    "rew_per_car_step": round(sum(r["own_rew"] for r in res) / max(car_steps, 1), 5),
                    "train": {k: round(v, 4) if isinstance(v, float) else v for k, v in tr.items()},
                    "queued_per_env": round(float(np.mean([r["queued"] for r in res])), 2),
                    **({"obstacles_per_env": round(float(np.mean([r["obstacles"] for r in res])), 2),
                        "obstacle_hits_ep": int(sum(r["obstacle_hits"] for r in res))} if st.cfg.obstacle_envs > 0 else {}),
                    "expl_var": explained_var(data),
                    "logstd": [round(float(x), 3) for x in logstd],
                    **upd,
                    "t_roll": round(t1 - t0, 2),
                    "t_upd": round(time.time() - t1, 2),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                if getattr(a, "max_minutes", 0) and time.time() - t_start >= 60 * a.max_minutes:
                    a.iters = it + 1  # this iteration is the last one: final checkpoint + eval below
                extra = dict(iter=it, ceiling=st.ceiling, logstd=logstd.tolist(), algo="ppo", **git)
                save_npz(out / "current.npz", actor.p, st.a_sizes, cfg_d, st.in_scale, arch=st.a_arch, **extra)
                np.savez(out / "state.npz", actor=actor.p, critic=critic.p, logstd=logstd, ceiling=st.ceiling, iter=it,
                         total_steps=st.total_steps)
                if a.eval_every and (it % a.eval_every == 0 or it == a.iters - 1):
                    ck = out / f"ckpt_{it:04d}.npz"
                    save_npz(ck, actor.p, st.a_sizes, cfg_d, st.in_scale, arch=st.a_arch, **extra)
                    if it == a.iters - 1 and eval_proc is not None:
                        eval_proc.join()  # always evaluate the final policy
                    if eval_proc is None or not eval_proc.is_alive():
                        rdir = str(out / f"render_{it:04d}") if a.render_every and it % a.render_every == 0 else None
                        job = (str(ck), str(out / "eval.jsonl"), it, EVAL_DEMANDS, EVAL_ENVS, EVAL_SECONDS, rdir, cfg_d)
                        eval_proc = ctx.Process(target=eval_job, args=job)
                        eval_proc.start()
                if it == a.iters - 1:
                    break  # --max-minutes reached (range() above was built with the original --iters)
        finally:
            source.close()
            if eval_proc is not None:
                eval_proc.join(timeout=600)


# ---------------------------------------------------------------- main
def main(argv=None):
    a = parse_args(argv)
    out = Path(a.runs_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    st = build_state(a)
    git = git_info()
    write_args(out, a, st, git)
    n_a = policy_set.n_params(st.a_arch) if st.a_arch else n_params(st.a_sizes)
    n_c = policy_set.n_params(st.c_arch) if st.c_arch else n_params(st.c_sizes)
    print(f"actor {'set ' + json.dumps(st.a_arch) if st.a_arch else st.a_sizes} ({n_a} params), "
          f"critic {'set' if st.c_arch else st.c_sizes} ({n_c} params)", flush=True)

    update = ppo_update
    if a.rollout_device != "numpy" and a.update_device == "numpy":
        a.update_device = a.rollout_device  # the batch stays on the rollout device
    if (st.a_arch or st.c_arch) and a.update_device == "numpy":
        raise SystemExit("--arch set needs the torch update: --update-device cpu|cuda (or --rollout-device cpu|cuda)")
    if a.update_device != "numpy":
        from .ppo_torch import TorchPPO

        update = TorchPPO(a.update_device, a.update_dtype).update

    ctx = mp.get_context("spawn")
    os.environ["OMP_NUM_THREADS"] = "1"  # for the spawned workers
    ep_steps = int(round(a.episode / st.cfg.dt))
    source = WorkerRollouts(ctx, a, st, ep_steps) if a.rollout_device == "numpy" else TorchRollouts(a, st, ep_steps)
    train_loop(a, st, out, source, update, ctx, git)


if __name__ == "__main__":
    main()
