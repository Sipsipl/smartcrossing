"""Shared MLP driving policy (pure numpy).

All cars of one group share one flat parameter vector. The network maps the
per-car observation of :mod:`smartcrossing.sim` to (accel, curvature, signal...)
with tanh hidden layers and a tanh output.

``forward`` accepts either one parameter vector (D,) applied to every car, or one
vector per environment (B, D) so a whole ES population can be evaluated inside a
single batched Sim (environment b uses params[b]).
"""

import json
from pathlib import Path

import numpy as np

from .sim import K_NEIGHBOURS, OBS_DIM, Sim, SimConfig, act_dim, obs_dim


def layer_sizes(cfg: SimConfig, hidden=(32, 32)) -> list[int]:
    return [obs_dim(cfg), *hidden, act_dim(cfg)]


def default_in_scale(cfg: SimConfig) -> np.ndarray:
    """Fixed per-feature input scaling: positions in the sim obs are divided by
    L = 50 m (own) or 30 m (neighbours), which makes the intersection box tiny;
    rescale them to ~10 m units so the network can resolve the turning point."""
    s = np.ones(obs_dim(cfg))
    s[0:2] = 5.0  # canonical own position
    s[8:10] = 2.0  # goal vector
    blk = 6 + cfg.n_signal
    for k in range(K_NEIGHBOURS):
        s[OBS_DIM + k * blk : OBS_DIM + k * blk + 2] = 3.0  # neighbour relative position
    return s


def n_params(sizes) -> int:
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


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


def forward(params: np.ndarray, obs: np.ndarray, sizes, in_scale=None) -> np.ndarray:
    """params (D,) with obs (..., obs_dim), or params (B, D) with obs (B, N, obs_dim).
    in_scale (obs_dim,) multiplies the observation first (None = identity)."""
    h = obs if in_scale is None else obs * in_scale
    batched = params.ndim == 2
    for W, b in unpack(params, sizes):
        if batched:
            h = np.tanh(np.einsum("bni,bio->bno", h, W, optimize=True) + b[:, None, :])
        else:
            h = np.tanh(h @ W + b)
    return h


class PolicyDriver:
    """Drives every car of a Sim with one of several parameter vectors chosen by
    ``sim.group``. ``mute[g]`` forces the signal output of group g to zero."""

    def __init__(self, sim: Sim, params, sizes, mute=None, train_cfg: dict | None = None, in_scale=None):
        self.sim = sim
        self.in_scale = in_scale
        self.sizes = list(sizes)
        plist = [params] if isinstance(params, np.ndarray) and params.ndim == 1 else list(params)
        self.params = [np.asarray(p, dtype=float) for p in plist]
        self.mute = list(mute) if mute is not None else [False] * len(self.params)
        self.train_cfg = train_cfg or {}
        if self.sizes[0] != obs_dim(sim.cfg) or self.sizes[-1] != act_dim(sim.cfg):
            raise ValueError(f"policy sizes {self.sizes} do not match sim obs/act dims")

    @classmethod
    def load(cls, path, sim: Sim, mute=None):
        """``path`` may be one .npz file or several joined by '+' (one per group)."""
        plist, sizes, tcfg, scale = [], None, None, None
        for p in str(path).split("+"):
            params, sizes, tcfg = load_npz(p)
            scale = load_in_scale(p)
            plist.append(params)
        if mute is None:  # policies trained without signal never learned to use their lamp
            mute = [not load_npz(p)[2].get("signal_enabled", True) for p in str(path).split("+")]
        # sensors are part of what a policy learned: run it with the perception range it was
        # trained with (first policy wins; checkpoints from before the field existed: 30 m)
        sim.cfg.neighbour_vis = float(load_npz(str(path).split("+")[0])[2].get("neighbour_vis", 30.0))
        return cls(sim, plist, sizes, mute=mute, train_cfg=tcfg, in_scale=scale)

    def act(self) -> np.ndarray:
        obs = self.sim.observe()
        grp = self.sim.group
        out = np.zeros((*obs.shape[:-1], self.sizes[-1]))
        for g, p in enumerate(self.params):
            m = grp == g if len(self.params) > 1 else np.ones(grp.shape, dtype=bool)
            if not m.any():
                continue
            a = forward(p, obs[m], self.sizes, self.in_scale)
            if self.mute[g]:
                a[:, 2:] = 0.0
            out[m] = a
        if len(self.params) > 1:  # groups beyond the given params use the last one
            m = grp >= len(self.params)
            if m.any():
                out[m] = forward(self.params[-1], obs[m], self.sizes, self.in_scale)
        return out


# ---------------------------------------------------------------- persistence
def save_npz(path, params, sizes, cfg: SimConfig | dict, in_scale=None, **extra):
    cfg_d = cfg if isinstance(cfg, dict) else dict(cfg.__dict__)
    np.savez(
        path,
        in_scale=np.ones(sizes[0]) if in_scale is None else np.asarray(in_scale, dtype=float),
        params=np.asarray(params),
        sizes=np.asarray(sizes, dtype=np.int64),
        cfg=np.array(json.dumps(cfg_d)),
        extra=np.array(json.dumps(extra)),
    )


def load_npz(path):
    with np.load(path) as z:
        return z["params"].copy(), [int(s) for s in z["sizes"]], json.loads(str(z["cfg"]))


def load_in_scale(path):
    with np.load(path) as z:
        return z["in_scale"].copy() if "in_scale" in z.files else None


def export_json(npz_path, json_path=None) -> dict:
    """Weights as nested lists (W[in][out], b[out]) for the browser viewer."""
    params, sizes, cfg = load_npz(npz_path)
    d = {
        "sizes": sizes,
        "in_scale": (scale if (scale := load_in_scale(npz_path)) is not None else np.ones(sizes[0])).tolist(),
        "activation": "tanh",
        "output": "tanh",
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in unpack(params, sizes)],
        "cfg": cfg,
    }
    if json_path:
        Path(json_path).write_text(json.dumps(d))
    return d


if __name__ == "__main__":
    import sys

    export_json(sys.argv[1], sys.argv[2])
