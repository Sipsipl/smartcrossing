"""Shared MLP driving policy (pure numpy).

All cars of one group share one flat parameter vector. The network maps the
per-car observation of :mod:`smartcrossing.sim` to (accel, curvature, signal...)
with tanh hidden layers and a tanh output.

Architecture: the default is this MLP on the flat observation (``arch`` None / absent from the
checkpoint). ``arch = {"type": "set", ...}`` is the neighbour set encoder of policy_set.py; its
checkpoints store the arch dict next to the flat params (``save_npz(..., arch=...)``) and keep
``sizes = [obs_dim, *trunk hidden, act_dim]`` so every size check stays valid. A set encoder may
carry a GRU memory (``arch["mem"]``); PolicyDriver then keeps the per-car state.
"""

import json
from pathlib import Path

import numpy as np

from . import policy_set
from .sim import OBS_DIM, Sim, SimConfig, act_dim, nb_features, obs_dim


def layer_sizes(cfg: SimConfig, hidden=(32, 32)) -> list[int]:
    return [obs_dim(cfg), *hidden, act_dim(cfg)]


def default_in_scale(cfg: SimConfig) -> np.ndarray:
    """Fixed per-feature input scaling. Own position (10 m units) and goal vectors (25 m
    units) are already scaled by fixed constants in the sim; neighbour positions are
    divided by 30 m there, rescale them to ~10 m units."""
    s = np.ones(obs_dim(cfg))
    blk = nb_features(cfg) + cfg.n_signal
    for k in range(cfg.n_neighbours):
        # neighbour relative position (~10 m units; a longer perception range keeps far cars in range)
        s[OBS_DIM + k * blk : OBS_DIM + k * blk + 2] = 3.0 * min(1.0, 30.0 / cfg.neighbour_vis)
    return s


def n_params(sizes) -> int:
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def param_count(sizes, arch=None) -> int:
    """Length of the flat parameter vector of an MLP (``sizes``) or a set encoder (``arch``)."""
    return policy_set.n_params(arch) if is_set(arch) else n_params(sizes)


def init_params(sizes, rng=None, out_scale=0.1) -> np.ndarray:
    """Scaled-normal init; the output layer is small so the initial policy keeps
    roughly constant speed and heading."""
    rng = np.random.default_rng(rng)
    parts = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        scale = 1.0 / np.sqrt(a)
        if i == len(sizes) - 2:
            scale *= out_scale
        parts += [rng.normal(0, scale, a * b), np.zeros(b)]
    return np.concatenate(parts)


def unpack(params: np.ndarray, sizes):
    """Split flat params (..., D) into [(W (..., in, out), b (..., out)), ...]."""
    layers, i = [], 0
    lead = params.shape[:-1]
    for a, b in zip(sizes[:-1], sizes[1:]):
        W = params[..., i : i + a * b].reshape(*lead, a, b)
        i += a * b
        bias = params[..., i : i + b]
        i += b
        layers.append((W, bias))
    assert i == params.shape[-1], f"param size {params.shape[-1]} != {i}"
    return layers


def is_set(arch) -> bool:
    return bool(arch) and arch.get("type", "mlp") == "set"


def mlp_linear(params: np.ndarray, sizes, x: np.ndarray) -> np.ndarray:
    """MLP with tanh hidden layers and a linear output (PPO's pre-squash mean; the policy is tanh of it)."""
    layers = unpack(params, sizes)
    for k, (W, b) in enumerate(layers):
        x = x @ W + b
        if k < len(layers) - 1:
            x = np.tanh(x)
    return x


def linear_forward(params: np.ndarray, sizes, x: np.ndarray, arch=None) -> np.ndarray:
    """Linear output of an MLP (``sizes``; arch None / mlp) or a set encoder (policy_set ``arch``).
    The torch twin is ppo_torch.linear_forward."""
    return policy_set.forward(params, arch, x) if is_set(arch) else mlp_linear(params, sizes, x)


def forward(params: np.ndarray, obs: np.ndarray, sizes, in_scale=None, arch=None) -> np.ndarray:
    """Deterministic policy: tanh of the linear output for params (D,) and obs (..., obs_dim).
    in_scale (obs_dim,) multiplies the observation first (None = identity). arch: None / mlp,
    or a policy_set arch."""
    if params.ndim != 1:
        raise ValueError("forward takes one parameter vector")
    h = obs if in_scale is None else obs * in_scale
    return np.tanh(linear_forward(params, sizes, h, arch))


class PolicyDriver:
    """Drives every car of a Sim with one of several parameter vectors chosen by
    ``sim.group``. ``mute[g]`` forces the signal output of group g to zero. Deterministic (tanh of the mean)
    unless ``set_noise`` made it stochastic. Memory brains (policy_set ``mem``) keep a state per car that
    ``act`` advances (call it once per sim step); ``reset_state`` forgets it."""

    def __init__(self, sim: Sim, params, sizes, mute=None, train_cfg: dict | None = None, in_scale=None, archs=None):
        self.sim = sim
        self.in_scale = in_scale
        self.sizes = list(sizes)
        plist = [params] if isinstance(params, np.ndarray) and params.ndim == 1 else list(params)
        self.params = [np.asarray(p, dtype=float) for p in plist]
        self.mute = list(mute) if mute is not None else [False] * len(self.params)
        self.train_cfg = train_cfg or {}
        # architecture per parameter vector (None = MLP); a single dict applies to all
        self.archs = list(archs) if isinstance(archs, (list, tuple)) else [archs] * len(self.params)
        for p, ar in zip(self.params, self.archs):
            if is_set(ar) and (policy_set.in_dim(ar) != obs_dim(sim.cfg) or policy_set.n_params(ar) != p.shape[-1]):
                raise ValueError(f"set-encoder policy {ar} does not match the sim's observation / its params")
        if self.sizes[0] != obs_dim(sim.cfg) or self.sizes[-1] != act_dim(sim.cfg):
            raise ValueError(f"policy sizes {self.sizes} do not match sim obs/act dims")
        self.noise_std, self.rng = None, None
        self.reset_state()

    def set_noise(self, std, rng):
        """Stochastic actions from now on: tanh(mu + std * eps) for the driving cars, eps ~ N(0, 1) from ``rng``
        (one draw per driving car and action dim, group by group); std None = deterministic again."""
        self.noise_std = None if std is None else np.asarray(std, dtype=float)
        self.rng = rng

    def reset_state(self):
        """Forget the memory of every car (memory brains, policy_set ``mem``; no-op for stateless ones). The state
        is kept per slot: h[j] (B, N, Hm) for parameter vector j (j = len(params): groups beyond the list) and the
        uid the state belongs to; a slot whose uid changed (new car) or that holds no driving car starts at 0."""
        self.h = [None] * (len(self.params) + 1)
        self.uid_prev = None

    def _policy(self, p, obs, arch, h=None):
        """Actions of the rows obs (and, with a memory state h, the new state)."""
        if h is None and self.noise_std is None:
            return forward(p, obs, self.sizes, self.in_scale, arch), None
        x = obs if self.in_scale is None else obs * self.in_scale
        if h is None:
            mu, h_new = linear_forward(p, self.sizes, x, arch), None
        else:
            mu, h_new = policy_set.forward_mem(p, arch, x, h)
        if self.noise_std is None:
            return np.tanh(mu), h_new
        return np.tanh(mu + self.noise_std * self.rng.standard_normal(mu.shape)), h_new

    def _run(self, j, p, arch, m, obs, reset, alive):
        """Actions of the slots m with parameter vector j; advances its memory state (if any)."""
        Hm = policy_set.mem_size(arch) if is_set(arch) else 0
        if not Hm:
            return self._policy(p, obs[m], arch)[0]
        if self.h[j] is None:
            self.h[j] = np.zeros((*m.shape, Hm))
        H = self.h[j]
        h_in = np.where(reset[m][:, None], 0.0, H[m])
        a, h_new = self._policy(p, obs[m], arch, h_in)
        H[m] = np.where(alive[m][:, None], h_new, 0.0)
        return a

    @classmethod
    def load(cls, path, sim: Sim, mute=None):
        """``path`` may be one .npz file or several joined by '+' (one per group)."""
        from .policy_compat import LEGACY_INTERFACE, adapt_inputs

        paths = str(path).split("+")
        # sensors, the action interface and the world (geometry, grip) are part of what a policy
        # learned: run it with the values it was trained with (older checkpoints: the legacy
        # values, see policy_compat); all policies of one sim must share them. Traffic settings
        # (spawn_right, wreck_time, demand) stay those of the given sim.
        ifaces = []
        for p in paths:
            tc = load_npz(p)[2]
            ifaces.append({k: type(legacy)(tc.get(k, legacy)) for k, legacy in LEGACY_INTERFACE.items()})
        for p, iface in zip(paths[1:], ifaces[1:]):
            if iface != ifaces[0]:
                diff = {k: (ifaces[0][k], v) for k, v in iface.items() if v != ifaces[0][k]}
                raise ValueError(f"policies with different interfaces cannot share one sim: {paths[0]} vs {p} "
                                 f"differ in {diff}")
        for k, v in ifaces[0].items():
            setattr(sim.cfg, k, v)
        if sim.sig.shape[-1] != sim.cfg.n_signal:  # lamp count of the checkpoint (fresh sim: lamps are 0)
            sim.sig = np.zeros((*sim.sig.shape[:-1], sim.cfg.n_signal))
        plist, archs, sizes, tcfg, scale = [], [], None, None, None
        for p in paths:
            params, sizes, tcfg = load_npz(p)
            scale = load_in_scale(p)
            archs.append(load_arch(p))
            if sizes[0] != obs_dim(sim.cfg) and not is_set(archs[-1]):  # older observation layout: remap inputs by feature name
                d = adapt_inputs(export_json(p), sim.cfg)
                sizes = d["sizes"]
                params = np.concatenate([np.concatenate([np.ravel(la["W"]), la["b"]]) for la in d["layers"]])
                scale = np.asarray(d["in_scale"], dtype=float)
            plist.append(params)
        if mute is None:  # policies trained without signal never learned to use their lamp
            mute = [not load_npz(p)[2].get("signal_enabled", True) for p in paths]
        return cls(sim, plist, sizes, mute=mute, train_cfg=tcfg, in_scale=scale, archs=archs)

    def act(self) -> np.ndarray:
        obs = self.sim.observe()
        grp = self.sim.group
        out = np.zeros((*obs.shape[:-1], self.sizes[-1]))
        # deterministic: every slot (empty ones are ignored by the sim); stochastic: the driving cars only
        alive = self.sim.alive
        cars = np.ones(grp.shape, dtype=bool) if self.noise_std is None else alive
        uid = self.sim.uid
        # memory: a slot starts from 0 when it holds a new car (uid changed) or no driving car
        reset = ~alive if self.uid_prev is None else ~alive | (uid != self.uid_prev)
        for g, p in enumerate(self.params):
            m = (grp == g if len(self.params) > 1 else np.ones(grp.shape, dtype=bool)) & cars
            if not m.any():
                continue
            a = self._run(g, p, self.archs[g], m, obs, reset, alive)
            if self.mute[g]:
                a[:, 2:] = 0.0
            out[m] = a
        if len(self.params) > 1:  # groups beyond the given params use the last one
            m = (grp >= len(self.params)) & cars
            if m.any():
                out[m] = self._run(len(self.params), self.params[-1], self.archs[-1], m, obs, reset, alive)
        for H in self.h:
            if H is not None:
                H[~alive] = 0.0
        self.uid_prev = uid.copy()
        return out


# ---------------------------------------------------------------- persistence
def save_npz(path, params, sizes, cfg: SimConfig | dict, in_scale=None, arch=None, **extra):
    """arch: None for the MLP (no field written: the file format of all older checkpoints), else
    the policy_set arch dict (stored as JSON in the field ``arch``)."""
    cfg_d = cfg if isinstance(cfg, dict) else dict(cfg.__dict__)
    np.savez(
        path,
        **({"arch": np.array(json.dumps(arch))} if is_set(arch) else {}),
        in_scale=np.ones(sizes[0]) if in_scale is None else np.asarray(in_scale, dtype=float),
        params=np.asarray(params),
        sizes=np.asarray(sizes, dtype=np.int64),
        cfg=np.array(json.dumps(cfg_d)),
        extra=np.array(json.dumps(extra)),
    )


def load_npz(path):
    with np.load(path) as z:
        return z["params"].copy(), [int(s) for s in z["sizes"]], json.loads(str(z["cfg"]))


def load_arch(path) -> dict | None:
    """The checkpoint's architecture (policy_set arch dict), or None for the MLP (also every older file)."""
    with np.load(path) as z:
        return json.loads(str(z["arch"])) if "arch" in z.files else None


def load_in_scale(path):
    with np.load(path) as z:
        return z["in_scale"].copy() if "in_scale" in z.files else None


def export_json(npz_path, json_path=None) -> dict:
    """Weights as nested lists (W[in][out], b[out]) for the browser (site/src/js/sim.js); a set
    encoder instead carries its ``arch`` and the flat ``params`` (no ``layers``)."""
    params, sizes, cfg = load_npz(npz_path)
    arch = load_arch(npz_path)
    d = {
        "sizes": sizes,
        "in_scale": (scale if (scale := load_in_scale(npz_path)) is not None else np.ones(sizes[0])).tolist(),
        "activation": "tanh",
        "output": "tanh",
        **({"arch": arch, "params": params.tolist()} if is_set(arch) else
           {"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in unpack(params, sizes)]}),
        "cfg": cfg,
    }
    if json_path:
        Path(json_path).write_text(json.dumps(d), encoding="utf-8", newline="\n")
    return d


if __name__ == "__main__":
    import sys

    export_json(sys.argv[1], sys.argv[2])
