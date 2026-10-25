"""PPO with parameter sharing: every car of B parallel environments is a sample of one
shared driving policy, trained on its own (per-car) reward.

Rollouts and the default update are pure numpy (hand-written tanh-MLP backward pass; ppo_numpy.py).
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
- `--mem N` (with --arch set): the actor gets a GRU memory of N units per car (policy_set.py ``mem``; the critic
  stays stateless); needs the torch rollout + update (sequence minibatches, truncated BPTT over each rollout).
  `--init` / `--init-actor` with a stateless set brain upgrade it (memory added with zero effect). state.npz
  records the actor arch.
- critic: separate tanh MLP on the scaled obs plus a few privileged features (car age,
  queue length, cars on the road, demand of the env); not exported.
- rollouts: persistent worker processes, each with its own vectorised Sim; GAE per car
  trajectory on the dense (T, B, N) slot grid: a slot's chain is cut where the car was
  removed (Sim.done_kind) - a new car in the same slot (new uid) starts a new
  trajectory; at the rollout end / env reset alive cars are bootstrapped with V.
- reward per car: its own reward (Sim.car_reward).
- curriculum: per-env demand drawn from [demand_lo * ceiling, ceiling]; the ceiling rises
  when the training crash+offroad rate is low.

- background mode (owner games / watches videos meanwhile): `--gpu-duty F` sleeps after each iteration
  so the busy fraction is ~F (sleep = t_iter * (1/F - 1)); `runs/<name>/duty.txt` (a float, read every
  iteration) overrides it while the run goes on. `--low-priority`: BELOW_NORMAL process priority
  on Windows (also the spawned workers / eval; no-op elsewhere). `--eval-workers 0`: no background evals
  while training (only the final checkpoint is evaluated). Helpers: background.py.

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
from .background import LOW_PRIORITY_ENV, duty_sleep, maybe_low_priority, read_duty, set_low_priority
from .gates import GATE_KEYS
from .jobs import append_jsonl
from .policy import default_in_scale, layer_sizes, param_count, save_npz
from .ppo_common import N_PRIV, git_info
from .ppo_numpy import BATCH_KEYS, MLP, Adam, SetNet, Worker, _worker_main, ppo_update, privileged
from .protocol import ENVS, SECONDS, SELECTION_SEEDS
from .sim import DONE_CRASH, DONE_EXIT, DONE_OFFROAD, DONE_TIMEOUT, DONE_WRONG, SimConfig, act_dim, obs_dim

# re-exported for the tests / scripts that import them from here
__all__ = ["MLP", "Adam", "SetNet", "Worker", "privileged", "ppo_update", "set_low_priority", "read_duty",
           "duty_sleep", "parse_args", "build_state", "write_args", "eval_job", "train_loop", "main"]


# ---------------------------------------------------------------- eval (subprocess)
TRAIN_EVAL_DEMANDS = (0.05, 0.15)  # the training evals' demands (quick; not protocol.EVAL_DEMANDS)
EVAL_ENVS, EVAL_SECONDS, EVAL_SEED = ENVS, SECONDS, SELECTION_SEEDS[0]  # cli bench settings
RENDER_DEMAND, RENDER_SECONDS = 0.15, 60.0


def eval_job(ckpt: str, out_path: str, it: int, demands, envs: int, seconds: float, render_dir: str | None,
             cfg_d: dict | None = None):
    """Bench metrics of the deterministic policy (cli bench settings) in the run's world (cfg_d: the
    training SimConfig; fixed demand per bench, rewards do not matter) plus the stop gate (gates.py,
    ~1 s); appends one JSON line."""
    from . import evalset
    from .gates import stop_gate

    maybe_low_priority()

    world = {**(cfg_d or {}), "demand_range": None}
    rec = {"iter": it, "ckpt": ckpt}
    for d in demands:
        s = evalset.bench_one(f"policy:{ckpt}", SimConfig(**{**world, "demand": d}), EVAL_SEED, envs, seconds)
        rec[f"d{d:g}"] = evalset.keep(s, (*evalset.KEEP_KEYS, "queued_end"))
    g, _ = stop_gate(f"policy:{ckpt}", world)
    rec["gate"] = {k: g[k] for k in GATE_KEYS}
    append_jsonl(out_path, [rec])
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
    p.add_argument("--mem", type=int, default=0,
                   help="actor memory: GRU state of this size per car (policy_set mem; --arch set and the torch "
                        "rollout --rollout-device cpu|cuda); 0 = stateless. A stateless --init / --init-actor brain "
                        "is upgraded (memory added with zero effect: same actions at the start)")
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
    p.add_argument("--init-actor", default="", help="actor from this policy .npz (same arch, or its stateless "
                   "version with --mem: upgraded); applied after --init (critic, log-std, ceiling, iteration stay "
                   "those of --init)")
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
    p.add_argument("--gpu-duty", type=float, default=1.0,
                   help="background mode: busy fraction 0 < F <= 1 (pause t_iter * (1/F - 1) after each iteration); "
                        "runs/<name>/duty.txt overrides it while running")
    p.add_argument("--low-priority", action="store_true",
                   help="BELOW_NORMAL process priority on Windows (also workers / eval processes)")
    p.add_argument("--eval-workers", type=int, default=1, choices=(0, 1),
                   help="background eval processes while training (0: only the final checkpoint is evaluated)")
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
    arch = a.arch
    c_arch = arch if a.critic_arch == "same" else a.critic_arch
    a_arch = c_arch_d = None
    if arch == "set" or c_arch == "set":
        kw = dict(hidden=hidden, phi=tuple(int(h) for h in a.set_phi.split(",")), pool=a.set_pool, heads=a.set_heads,
                  dk=a.set_dk, dv=a.set_dk)
        a_arch = policy_set.make_arch(cfg, mem=a.mem, **kw) if arch == "set" else None
        c_arch_d = policy_set.make_arch(cfg, n_extra=N_PRIV, n_out=1, **kw) if c_arch == "set" else None
    if a.mem and not a_arch:
        raise ValueError("--mem needs the set-encoder actor (--arch set)")
    actor = SetNet(a_arch, a_sizes, rng, out_scale=0.01) if a_arch else MLP(a_sizes, rng, out_scale=0.01)
    critic = SetNet(c_arch_d, c_sizes, rng, out_scale=1.0) if c_arch_d else MLP(c_sizes, rng, out_scale=1.0)
    # output bias: initial mean acceleration action (the last block of the stateless layouts)
    actor.p[policy_set.out_bias_index(a_arch) if a_arch else len(actor.p) - A] = np.arctanh(a.acc_init)
    st = TrainState(cfg, in_scale, act_mask, a_sizes, c_sizes, rng, actor, critic, np.full(A, a.logstd_init),
                    Adam(len(actor.p), a.lr), Adam(len(critic.p), a.lr_critic), a.demand_start, a_arch=a_arch,
                    c_arch=c_arch_d)
    if a.init_policy and a_arch:
        raise ValueError("--init-policy transplants an MLP actor; it cannot start a set-encoder actor (--init-bc)")
    if a.init_bc:
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
        stored = json.loads(str(z["actor_arch"])) if "actor_arch" in z.files else "unknown"
        actor.p = actor_params(z["actor"], stored, a_arch, a_sizes, f"--init {a.init}", rng)
        critic.p, st.logstd = z["critic"].copy(), z["logstd"].copy()
        st.ceiling = float(z["ceiling"]) if "ceiling" in z.files else st.ceiling
        st.it0 = int(z["iter"]) + 1 if "iter" in z.files else 0
        st.total_steps = int(z["total_steps"]) if "total_steps" in z.files else 0
        if a.reset_logstd:
            st.logstd = np.full(A, a.logstd_init)
        if a.set_ceiling:
            st.ceiling = a.set_ceiling
    if a.init_actor:
        from .policy import load_arch, load_npz

        actor.p = actor_params(load_npz(a.init_actor)[0], load_arch(a.init_actor), a_arch, a_sizes,
                               f"--init-actor {a.init_actor}", rng)
    return st


def actor_params(p, arch_from, arch, sizes, what: str, rng) -> np.ndarray:
    """Actor params p (stored with ``arch_from``: an arch dict, None = MLP, "unknown" = older state.npz without
    the field) for the run's actor (``arch`` / ``sizes``): taken as they are, or a stateless set brain upgraded to
    the run's memory arch (policy_set.upgrade_params: drives the same). Anything else is an error."""
    p = np.asarray(p, dtype=float).copy()
    n = param_count(sizes, arch)
    if arch_from == "unknown":  # infer from the size: the run's arch, or (memory run) its stateless version
        if len(p) == n:
            return p
        arch_from = policy_set.base_arch(arch) if arch and policy_set.mem_size(arch) else arch
    if arch_from == arch and len(p) == n:
        return p
    if arch and policy_set.mem_size(arch) and arch_from == policy_set.base_arch(arch):
        print(f"{what}: stateless actor upgraded to the memory arch (memory -> trunk weights 0)", flush=True)
        return policy_set.upgrade_params(p, arch_from, arch, rng)
    raise ValueError(f"{what}: actor arch {arch_from} ({len(p)} params) does not match the run's {arch} ({n} params)")


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
        return [r], {k: r[k] for k in (*BATCH_KEYS, "seq") if k in r}

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
def train_rates(kinds) -> dict:
    """Shares of the finished cars per done kind in a rollout (kinds: bincount of Sim.done_kind)."""
    fin = max(kinds[1:].sum(), 1)
    return {
        "exit_ok": kinds[DONE_EXIT] / fin,
        "crash": kinds[DONE_CRASH] / fin,
        "offroad": kinds[DONE_OFFROAD] / fin,
        "timeout": kinds[DONE_TIMEOUT] / fin,
        "wrong": kinds[DONE_WRONG] / fin,
        "finished": int(kinds[1:].sum()),
    }


def curriculum_step(a, st: TrainState, tr: dict) -> bool:
    """Raise the demand ceiling when the training crash+offroad rate is low; returns whether it was raised."""
    if tr["crash"] + tr["offroad"] < a.safe_bad and tr["exit_ok"] > a.safe_exit and st.ceiling < a.demand:
        st.ceiling = min(a.demand, st.ceiling * a.demand_step)
        return True
    return False


def iteration_record(it: int, st: TrainState, res: list, data: dict, upd: dict, tr: dict, raised: bool,
                     car_steps: int, t0: float, t1: float) -> dict:
    """The log.jsonl line of one iteration (rollout started at t0, update at t1)."""
    return {
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
        "logstd": [round(float(x), 3) for x in st.logstd],
        **upd,
        "t_roll": round(t1 - t0, 2),
        "t_upd": round(time.time() - t1, 2),
    }


def checkpoint_and_eval(a, st: TrainState, out: Path, it: int, final: bool, git: dict, ctx, eval_proc):
    """current.npz + state.npz every iteration; ckpt_XXXX.npz every --eval-every iterations and at the end (the
    final checkpoint is always written), each with a background eval process (eval_job) unless one is still
    running (the final one waits for it). Returns the eval process now running (or the previous one / None)."""
    actor, critic, logstd, cfg_d = st.actor, st.critic, st.logstd, st.cfg_d
    extra = dict(iter=it, ceiling=st.ceiling, logstd=logstd.tolist(), algo="ppo", **git)
    save_npz(out / "current.npz", actor.p, st.a_sizes, cfg_d, st.in_scale, arch=st.a_arch, **extra)
    np.savez(out / "state.npz", actor=actor.p, critic=critic.p, logstd=logstd, ceiling=st.ceiling, iter=it,
             total_steps=st.total_steps, actor_arch=np.array(json.dumps(st.a_arch)))
    if final or (a.eval_every and it % a.eval_every == 0):
        ck = out / f"ckpt_{it:04d}.npz"
        save_npz(ck, actor.p, st.a_sizes, cfg_d, st.in_scale, arch=st.a_arch, **extra)
        if final and eval_proc is not None:
            eval_proc.join()  # always evaluate the final policy
        if a.eval_every and (a.eval_workers or final) and (eval_proc is None or not eval_proc.is_alive()):
            rdir = str(out / f"render_{it:04d}") if a.render_every and it % a.render_every == 0 else None
            job = (str(ck), str(out / "eval.jsonl"), it, TRAIN_EVAL_DEMANDS, EVAL_ENVS, EVAL_SECONDS, rdir, cfg_d)
            eval_proc = ctx.Process(target=eval_job, args=job)
            eval_proc.start()
    return eval_proc


def train_loop(a, st: TrainState, out: Path, source, update, ctx, git: dict):
    """PPO iterations it0 .. iters-1: rollout, update, curriculum, log line, checkpoints, evals."""
    actor, critic, logstd, opt_a = st.actor, st.critic, st.logstd, st.opt_a
    eval_proc = None
    t_start = time.time()
    duty_file = out / "duty.txt"
    if not 0 < a.gpu_duty <= 1:
        raise SystemExit("--gpu-duty must be in (0, 1]")
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
                tr = train_rates(kinds)
                raised = curriculum_step(a, st, tr)
                rec = iteration_record(it, st, res, data, upd, tr, raised, car_steps, t0, t1)
                duty = read_duty(duty_file, a.gpu_duty)
                pause = duty_sleep(time.time() - t0, duty)
                if duty < 1 or a.gpu_duty < 1 or duty_file.exists():
                    rec["duty"], rec["t_sleep"] = round(duty, 3), round(pause, 2)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                if a.max_minutes and time.time() - t_start >= 60 * a.max_minutes:
                    a.iters = it + 1  # this iteration is the last one: final checkpoint + eval below
                final = it == a.iters - 1
                eval_proc = checkpoint_and_eval(a, st, out, it, final, git, ctx, eval_proc)
                if final:
                    break  # --max-minutes reached (range() above was built with the original --iters)
                if pause > 0:
                    time.sleep(pause)
        finally:
            source.close()
            if eval_proc is not None:
                eval_proc.join(timeout=600)

# ---------------------------------------------------------------- main
def main(argv=None):
    a = parse_args(argv)
    if a.low_priority:
        os.environ[LOW_PRIORITY_ENV] = "1"  # inherited by the spawned workers / eval processes
        set_low_priority()
    out = Path(a.runs_dir) / a.name
    out.mkdir(parents=True, exist_ok=True)
    st = build_state(a)
    git = git_info()
    write_args(out, a, st, git)
    n_a, n_c = param_count(st.a_sizes, st.a_arch), param_count(st.c_sizes, st.c_arch)
    print(f"actor {'set ' + json.dumps(st.a_arch) if st.a_arch else st.a_sizes} ({n_a} params), "
          f"critic {'set' if st.c_arch else st.c_sizes} ({n_c} params)", flush=True)

    update = ppo_update
    if a.rollout_device != "numpy" and a.update_device == "numpy":
        a.update_device = a.rollout_device  # the batch stays on the rollout device
    if (st.a_arch or st.c_arch) and a.update_device == "numpy":
        raise SystemExit("--arch set needs the torch update: --update-device cpu|cuda (or --rollout-device cpu|cuda)")
    if policy_set.mem_size(st.a_arch) and a.rollout_device == "numpy":
        raise SystemExit("--mem needs the torch rollout: --rollout-device cpu|cuda")
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
