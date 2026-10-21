"""PPO rollouts on the GPU (``ppo --rollout-device cuda``): ``TorchRollout`` is the torch counterpart
of :class:`smartcrossing.ppo.Worker` for one big batch of envs (``TorchSim``).

Per step: observe, privileged critic features, actor forward + Gaussian noise + tanh squash,
log-prob, critic forward, ``TorchSim.step``, per-car rewards / done flags, all written into
dense (T, B, N) device buffers. On cuda (``rng="torch"``) the whole step is one CUDA graph.
Bootstrap values, GAE (per slot chain, cut at removals) and the compaction to the valid
car-steps run on the device too; the returned batch holds device tensors that
``ppo_torch.TorchPPO.update`` consumes without a round trip to the host.

Semantics are those of ``Worker.rollout`` (same buffers, dtypes, GAE, episode resets every
``episode_steps``, one host sync per rollout). For the parity test the traffic can be drawn
with sim.py's numpy streams (``rng="numpy"``) and the action noise with Worker's numpy
generator (``noise="numpy"``): then the batch equals Worker's up to float rounding
(tests/test_ppo_gpu_rollout.py). Differences in the fast mode: different random streams
(same distributions); episode resets copy a fresh sim (incl. its generator state) into the
existing tensors; the CUDA-graph capture runs ``warmup`` real steps once.
"""

import numpy as np
import torch

from .ppo_common import LOG2PI, N_DONE, N_PRIV, PRIV_DEMAND_SCALE, PRIV_QUEUE_SCALE
from .policy import param_count
from .ppo_torch import linear_forward
from .sim import SimConfig, act_dim, obs_dim
from .sim_torch import TorchSim


def privileged_t(sim: TorchSim) -> torch.Tensor:
    """(B, N, N_PRIV) critic-only features, as ppo.privileged."""
    cfg = sim.cfg
    B, N = sim.alive.shape
    dem = torch.full((B,), cfg.demand, dtype=sim.dtype, device=sim.device) if sim.demand is None else sim.demand
    return torch.stack([
        sim.age / cfg.max_age,
        (sim.queue.sum(1).to(sim.dtype) / PRIV_QUEUE_SCALE)[:, None].expand(B, N),
        (sim.alive.sum(1).to(sim.dtype) / N)[:, None].expand(B, N),
        (dem * PRIV_DEMAND_SCALE)[:, None].expand(B, N),
    ], -1)


class TorchRollout:
    def __init__(self, cfg_d, n_envs, seed, horizon, episode_steps, gamma, lam, in_scale, act_mask,
                 actor_sizes, critic_sizes, device="cuda", dtype=torch.float32, rng="torch", noise="torch",
                 graph=None, warmup=3, actor_arch=None, critic_arch=None):
        self.cfg_d = dict(cfg_d)
        self.B, self.seed, self.T, self.ep_steps = n_envs, seed, horizon, episode_steps
        self.gamma, self.lam = gamma, lam
        self.dev = torch.device(device)
        self.dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.rng_mode, self.noise_mode = rng, noise
        self.graph = (self.dev.type == "cuda" and rng == "torch" and noise == "torch") if graph is None else graph
        assert not self.graph or (self.dev.type == "cuda" and rng == "torch" and noise == "torch")
        self.warmup = warmup
        self.a_sizes, self.c_sizes = list(actor_sizes), list(critic_sizes)
        self.a_arch, self.c_arch = actor_arch, critic_arch  # None = MLP, else a policy_set arch
        self.np_rng = np.random.default_rng(seed + 12345)  # Worker's action-noise generator
        self.sim = None
        self.n_resets = 0
        self.steps_in_ep = 0
        self._replay = None
        cfg = SimConfig(**self.cfg_d)
        N, D, A = cfg.n_slots, obs_dim(cfg), act_dim(cfg)
        self.N, self.D, self.A = N, D, A
        f, kw = self.dt, dict(device=self.dev)
        self.in_scale = torch.as_tensor(np.asarray(in_scale), **kw).to(f)
        self.mask = torch.as_tensor(np.asarray(act_mask), **kw).to(f)
        self.pa = torch.zeros(param_count(self.a_sizes, actor_arch), dtype=f, **kw)
        self.pc = torch.zeros(param_count(self.c_sizes, critic_arch), dtype=f, **kw)
        self.ls = torch.zeros(A, dtype=f, **kw)
        T, B = horizon, n_envs
        f32 = torch.float32
        # dense buffers (as Worker: obs / priv / u / logp stored in float32)
        self.obs_buf = torch.zeros(T, B, N, D, dtype=f32, **kw)
        self.priv_buf = torch.zeros(T, B, N, N_PRIV, dtype=f32, **kw)
        self.u_buf = torch.zeros(T, B, N, A, dtype=f32, **kw)
        self.logp_buf = torch.zeros(T, B, N, dtype=f32, **kw)
        self.val = torch.zeros(T + 1, B, N, dtype=f, **kw)
        self.rew = torch.zeros(T, B, N, dtype=f, **kw)
        self.done = torch.zeros(T, B, N, dtype=torch.bool, **kw)
        self.valid = torch.zeros(T, B, N, dtype=torch.bool, **kw)
        self.kinds = torch.zeros(N_DONE, dtype=torch.int64, **kw)
        self.own_rew = torch.zeros((), dtype=f, **kw)
        self.tidx = torch.zeros(1, dtype=torch.int64, **kw)
        self.noise_gen = None
        if noise == "torch" and rng != "torch":
            self.noise_gen = torch.Generator(device=self.dev)
            self.noise_gen.manual_seed(seed + 12345)

    # ---------------------------------------------------------------- params
    def set_params(self, actor_p, logstd, critic_p):
        """Copy the current (numpy) parameters into the static device tensors."""
        for dst, src in ((self.pa, actor_p), (self.ls, logstd), (self.pc, critic_p)):
            dst.copy_(torch.as_tensor(np.asarray(src)).to(self.dt), non_blocking=False)

    # ---------------------------------------------------------------- sim / reset
    def _reset(self, d_lo, d_hi):
        cfg = SimConfig(**{**self.cfg_d, "demand_range": (d_lo, d_hi)})
        seed = self.seed * 100003 + self.n_resets
        if self.sim is not None and self.rng_mode == "torch":
            self.sim.reset_inplace(seed, cfg)  # keeps the tensors (and a captured graph) alive
        else:
            self.sim = TorchSim(cfg, self.B, seed=seed, device=self.dev, dtype=self.dt, rng=self.rng_mode)
            self._replay = None
        self.n_resets += 1
        self.steps_in_ep = 0

    def _actor(self, o):
        return linear_forward(self.pa, self.a_sizes, o, self.a_arch)

    def _critic(self, x):
        return linear_forward(self.pc, self.c_sizes, x, self.c_arch)

    def _noise(self, al):
        B, N, A = self.B, self.N, self.A
        if self.noise_mode == "numpy":  # Worker's draws: (n_alive, A) in row-major slot order
            z = self.np_rng.standard_normal((int(al.sum()), A))
            out = torch.zeros(B, N, A, dtype=self.dt, device=self.dev)
            out[al] = torch.as_tensor(z, device=self.dev).to(self.dt)
            return out
        gen = self.noise_gen or self.sim.gen
        return torch.randn(B, N, A, generator=gen, device=self.dev, dtype=self.dt)

    def _step(self):
        """One rollout step (graph-capturable when noise and traffic are drawn on the device)."""
        sim = self.sim
        obs = sim.observe()
        al = sim.alive.clone()
        pr = privileged_t(sim)
        o = obs * self.in_scale
        mu = self._actor(o)
        std = torch.exp(self.ls)
        u = mu + std * self._noise(al)
        lp = (-0.5 * ((u - mu) / std) ** 2 - self.ls - 0.5 * LOG2PI) @ self.mask
        v = self._critic(torch.cat([o, pr], -1))
        v = torch.where(al, v[..., 0], 0.0)
        act = torch.where(al[..., None], torch.tanh(u), 0.0)
        t = self.tidx
        f32 = torch.float32
        self.obs_buf.index_copy_(0, t, obs.to(f32)[None])
        self.priv_buf.index_copy_(0, t, torch.where(al[..., None], pr, 0.0).to(f32)[None])
        self.u_buf.index_copy_(0, t, u.to(f32)[None])
        self.logp_buf.index_copy_(0, t, lp.to(f32)[None])
        self.val.index_copy_(0, t, v[None])
        self.valid.index_copy_(0, t, al[None])
        sim.step(act)
        r = torch.where(al, sim.car_reward, 0.0)
        self.own_rew += r.sum()
        self.rew.index_copy_(0, t, r[None])
        self.done.index_copy_(0, t, (al & (sim.done_kind > 0))[None])
        oh = torch.nn.functional.one_hot(sim.done_kind.to(torch.int64), N_DONE) * al[..., None]
        self.kinds += oh.sum((0, 1))
        self.tidx += 1

    def advance(self, n):
        """n eager steps outside a rollout (buffers are overwritten by the next rollout)."""
        for _ in range(n):
            self._step()
        self.steps_in_ep += n

    # ---------------------------------------------------------------- rollout
    def rollout(self, d_lo, d_hi):
        if self.sim is None or self.steps_in_ep >= self.ep_steps:
            self._reset(d_lo, d_hi)
        sim = self.sim
        if self.graph and self._replay is None:
            self.tidx.zero_()
            self._replay = sim.cuda_graph(self._step, warmup=self.warmup)
            self.steps_in_ep += self.warmup  # the warm-up steps were real steps
        T = self.T
        self.tidx.zero_()
        self.kinds.zero_()
        self.own_rew.zero_()
        for _ in range(T):
            if self._replay is not None:
                self._replay()
            else:
                self._step()
        self.steps_in_ep += T
        # bootstrap values for cars still on the road
        o = sim.observe() * self.in_scale
        v = self._critic(torch.cat([o, privileged_t(sim)], -1))
        self.val[T] = torch.where(sim.alive, v[..., 0], 0.0)
        # GAE per slot chain (chain cut at removals)
        val, rew, valid = self.val, self.rew, self.valid
        adv = torch.zeros_like(rew)
        last = torch.zeros_like(rew[0])
        g, gl = self.gamma, self.gamma * self.lam
        for t in range(T - 1, -1, -1):
            nd = (~self.done[t]).to(self.dt)
            delta = rew[t] + g * val[t + 1] * nd - val[t]
            last = torch.where(valid[t], delta + gl * nd * last, 0.0)
            adv[t] = last
        ret = adv + val[:T]
        idx = valid.reshape(-1).nonzero().squeeze(1)  # the one host sync per rollout
        f32 = torch.float32

        def take(buf):
            return buf.reshape(T * self.B * self.N, *buf.shape[3:])[idx]

        n_obst_env = max(int(sim.obst_env.sum()), 1)
        stats = torch.stack([self.own_rew.to(torch.float64), sim.queue.sum().to(torch.float64) / self.B,
                             sim.demand.to(torch.float64).mean(), sim.fifo_overflow.to(torch.float64),
                             sim.obstacle.sum().to(torch.float64) / n_obst_env,
                             sim.stats.obstacle_hits.sum().to(torch.float64)]).tolist()
        if stats[3]:
            print("warning: TorchSim arrival FIFO overflow (graph mode, capacity 256 per arm)", flush=True)
        return {
            "obs": take(self.obs_buf),
            "priv": take(self.priv_buf),
            "u": take(self.u_buf),
            "logp": take(self.logp_buf),
            "adv": take(adv).to(f32),
            "ret": take(ret).to(f32),
            "val": take(val[:T]).to(f32),
            "kinds": self.kinds.cpu().numpy(),
            "car_steps": int(idx.numel()),
            "own_rew": stats[0],
            "queued": stats[1],
            "demand_mean": stats[2],
            "obstacles": stats[4],  # standing obstacles per obstacle env at the rollout end
            "obstacle_hits": stats[5],  # cars that hit an obstacle in the current episode
        }
