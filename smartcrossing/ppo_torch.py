"""PyTorch port of :func:`smartcrossing.ppo.ppo_update` (Roadmap C, stage 1: the update on the GPU).

Drop-in replacement: ``TorchPPO(device, dtype).update`` takes exactly the arguments of
``ppo_update`` (numpy ``MLP`` actor/critic, numpy ``Adam`` optimisers, log-std array, rollout
batch, args, act_mask, in_scale, rng) and returns the same stats dict. The numpy objects stay
the source of truth: parameters and Adam moments are copied to the device at the start of
each update and written back (in place) at the end, so rollout workers, checkpoints
(``save_npz`` / ``state.npz``), ``PolicyDriver``, ``policy_compat`` and the website are
unaffected, and the numpy and torch paths can be switched between iterations.

The math mirrors ppo.py operation by operation (hand-written backward pass instead of
autograd, same Adam, same clipping, same minibatch permutation drawn from the numpy ``rng``),
so in float64 it matches the numpy update to rounding (tests/test_ppo_torch.py). float32 on
cuda is the fast path. One host sync per minibatch (the target-KL early stop needs it).
"""

import os

import numpy as np
import torch

LOG2PI = float(np.log(2 * np.pi))


def _layers(p: torch.Tensor, sizes):
    """Views (W (in, out), b (out,)) into the flat parameter vector (policy.py layout)."""
    out, i = [], 0
    for a, b in zip(sizes[:-1], sizes[1:]):
        out.append((p[i : i + a * b].view(a, b), p[i + a * b : i + a * b + b]))
        i += a * b + b
    return out


def mlp_forward(p, sizes, x):
    """tanh hidden layers, linear output; returns (output, activations incl. input)."""
    Ls = _layers(p, sizes)
    hs = [x]
    h = x
    for k, (W, b) in enumerate(Ls):
        z = h @ W + b
        h = z if k == len(Ls) - 1 else torch.tanh(z)
        hs.append(h)
    return h, hs


def mlp_backward(p, sizes, hs, gout):
    """Gradient of sum(gout * output) wrt the flat params (same as ppo.MLP.backward)."""
    Ls = _layers(p, sizes)
    grads = []
    g = gout
    for k in range(len(Ls) - 1, -1, -1):
        W, _ = Ls[k]
        grads.append((hs[k].T @ g, g.sum(0)))
        if k > 0:
            g = (g @ W.T) * (1.0 - hs[k] ** 2)
    flat = []
    for gW, gb in reversed(grads):
        flat += [gW.reshape(-1), gb]
    return torch.cat(flat)


class _Adam:
    """Device copy of a numpy ppo.Adam (same update formula)."""

    def __init__(self, opt, dev, dt):
        self.opt = opt
        self.m = torch.as_tensor(opt.m, device=dev).to(dt)
        self.v = torch.as_tensor(opt.v, device=dev).to(dt)
        self.t = opt.t

    def step(self, g):
        o = self.opt
        self.t += 1
        self.m = o.b1 * self.m + (1 - o.b1) * g
        self.v = o.b2 * self.v + (1 - o.b2) * g * g
        mh = self.m / (1 - o.b1**self.t)
        vh = self.v / (1 - o.b2**self.t)
        return o.lr * mh / (torch.sqrt(vh) + o.eps)

    def write_back(self):
        self.opt.m[...] = self.m.cpu().numpy()
        self.opt.v[...] = self.v.cpu().numpy()
        self.opt.t = self.t


def _clip_norm(g, max_grad):
    n = torch.linalg.vector_norm(g)
    return g * torch.where(n > max_grad, max_grad / n, torch.ones_like(n))


class TorchPPO:
    def __init__(self, device="cuda", dtype="float32"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--update-device cuda: torch.cuda is not available")
        self.dev = torch.device(device)
        self.dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if self.dev.type == "cpu":  # workers idle during the (synchronous) update
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

    def _t(self, x):
        return torch.as_tensor(np.asarray(x), device=self.dev).to(self.dt)

    def update(self, actor, logstd, critic, opt_a, opt_c, data, a, act_mask, in_scale, rng):
        T = self._t
        n = len(data["adv"])
        obs = T(data["obs"]) * T(in_scale)
        cin = torch.cat([obs, T(data["priv"])], 1)
        u, old_lp_all = T(data["u"]), T(data["logp"])
        adv_all, ret_all = T(data["adv"]), T(data["ret"])
        mask = T(act_mask)
        pa, pc, ls = T(actor.p), T(critic.p), T(logstd)
        oa, oc = _Adam(opt_a, self.dev, self.dt), _Adam(opt_c, self.dev, self.dt)
        sa, sc = actor.sizes, critic.sizes
        clip_hi, clip_lo = 1 + a.clip, 1 - a.clip
        stats = {"kl": [], "clipfrac": [], "vloss": [], "pg": []}
        mb = int(np.ceil(n / a.minibatches))
        stop = False
        for ep in range(a.epochs):
            perm = torch.as_tensor(rng.permutation(n), device=self.dev)
            for s in range(0, n, mb):
                idx = perm[s : s + mb]
                m = len(idx)
                adv = adv_all[idx]
                adv = (adv - adv.mean()) / (adv.std(correction=0) + 1e-8)
                mu, hs = mlp_forward(pa, sa, obs[idx])
                std = torch.exp(ls)
                z = (u[idx] - mu) / std
                lp = (-0.5 * z**2 - ls - 0.5 * LOG2PI) @ mask
                ratio = torch.exp(lp - old_lp_all[idx])
                active = ~(((adv > 0) & (ratio > clip_hi)) | ((adv < 0) & (ratio < clip_lo)))
                g_lp = -(adv * ratio * active) / m
                g_mu = g_lp[:, None] * (z / std) * mask
                g_ls = (g_lp[:, None] * (z**2 - 1.0) * mask).sum(0) - a.ent * mask
                ga = _clip_norm(mlp_backward(pa, sa, hs, g_mu), a.max_grad)
                pa = pa - oa.step(ga)
                ls = torch.clamp(ls - a.lr_std * torch.clamp(g_ls, -1, 1), a.logstd_min, 0.5)
                # critic
                v, hc = mlp_forward(pc, sc, cin[idx])
                err = v[:, 0] - ret_all[idx]
                gc = _clip_norm(mlp_backward(pc, sc, hc, (err / m)[:, None]), a.max_grad)
                pc = pc - oc.step(gc)
                st = torch.stack([
                    torch.mean((ratio - 1) - torch.log(ratio)),
                    torch.mean((torch.abs(ratio - 1) > a.clip).to(self.dt)),
                    torch.mean(err**2),
                    torch.mean(adv * ratio * active),
                ]).tolist()  # the one host sync per minibatch
                for k, x in zip(("kl", "clipfrac", "vloss", "pg"), st):
                    stats[k].append(float(x))
                if a.target_kl and st[0] > 1.5 * a.target_kl:
                    stop = True
                    break
            if stop:
                break
        actor.p[...] = pa.cpu().numpy()
        critic.p[...] = pc.cpu().numpy()
        logstd[...] = ls.cpu().numpy()
        oa.write_back()
        oc.write_back()
        out = {k: round(float(np.mean(v)), 5) for k, v in stats.items()}
        out["epochs_done"] = ep + 1
        return out
