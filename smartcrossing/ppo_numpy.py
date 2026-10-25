"""Numpy parts of the PPO trainer (ppo.py): the tanh MLP with its hand-written backward pass, Adam, the set-encoder
wrapper (forward only; the torch update differentiates it), the privileged critic features, the rollout Worker
(one vectorised Sim per spawned process, GAE per slot chain) with its process entry point, and the numpy PPO update.
"""

import os

import numpy as np

from . import policy_set
from .background import maybe_low_priority
from .policy import linear_forward
from .ppo_common import LOG2PI, N_DONE, N_PRIV, PRIV_DEMAND_SCALE, PRIV_QUEUE_SCALE
from .sim import Sim, SimConfig, act_dim, obs_dim

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
                mu = linear_forward(actor_p, actor_sizes, o, self.a_arch)
                u = mu + std * self.rng.standard_normal(mu.shape)
                lp = (-0.5 * ((u - mu) / std) ** 2 - logstd - 0.5 * LOG2PI) @ self.act_mask
                v = linear_forward(critic_p, critic_sizes, np.concatenate([o, pr[bi, ni]], 1), self.c_arch)[:, 0]
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
            val[T, bi, ni] = linear_forward(critic_p, critic_sizes, np.concatenate([o, privileged(sim)[bi, ni]], 1),
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
    maybe_low_priority()
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
