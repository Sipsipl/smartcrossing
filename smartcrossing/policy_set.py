"""Set encoder over the neighbour slots (Roadmap B), pure numpy.

Same observation as the MLP (no new information), but the neighbour blocks are processed as a
SET with shared weights, so a reaction ("car ahead brakes") is learned once, whatever slot the
car lands in, and more neighbour slots (``SimConfig.n_neighbours``) cost no extra weights:

    x = [ego (n_ego) | K neighbour blocks (F each) | extra (n_extra, e.g. the critic's priv)]
    h_j    = phi(nb_j)                         per-neighbour tanh MLP, shared over j
    pooled = pool({h_j : nb_j present})        presence = the neighbour's ``present`` feature != 0
        attn: per head, q = [ego, extra] Wq + bq, k_j = h_j Wk, v_j = h_j Wv + bv,
              w = softmax_j(q . k_j / sqrt(dk)) over present neighbours, pooled = sum_j w_j v_j
        mean / max: masked mean / max of h_j (DeepSets baselines)
        no neighbour present -> pooled = 0
    out    = trunk([ego, extra, pooled])       tanh hidden layers, linear output

``arch`` (a JSON-able dict, stored in the checkpoint) fixes the layout; the parameters are one
flat vector in the order of ``param_shapes`` (each W row-major (in, out), then its bias), like
the MLP layout of policy.py. ``forward`` returns the linear output (PPO's pre-squash mean); the
deterministic policy is tanh of it (policy.forward). The torch twin is ppo_torch.set_forward and
the browser twin site/src/js/sim.js (parity tests: tests/test_policy_set.py).
"""

import numpy as np

from .sim import OBS_DIM, SimConfig, act_dim, nb_features

POOLS = ("attn", "mean", "max")
NB_PRESENT = 5  # index of the `present` flag inside a neighbour block
NEG = -1e9  # attention logit of absent neighbours


def make_arch(cfg: SimConfig, hidden=(64, 64), phi=(32, 32), pool="attn", heads=4, dk=8, dv=8, n_extra=0,
              n_out=None) -> dict:
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    return {"type": "set", "n_ego": OBS_DIM, "K": int(cfg.n_neighbours), "F": nb_features(cfg) + cfg.n_signal,
            "n_extra": int(n_extra), "present": NB_PRESENT, "phi": [int(h) for h in phi], "pool": pool,
            "heads": int(heads), "dk": int(dk), "dv": int(dv), "hidden": [int(h) for h in hidden],
            "n_out": int(act_dim(cfg) if n_out is None else n_out)}


def in_dim(arch) -> int:
    return arch["n_ego"] + arch["K"] * arch["F"] + arch["n_extra"]


def pooled_dim(arch) -> int:
    return arch["heads"] * arch["dv"] if arch["pool"] == "attn" else arch["phi"][-1]


def trunk_sizes(arch) -> list[int]:
    return [arch["n_ego"] + arch["n_extra"] + pooled_dim(arch), *arch["hidden"], arch["n_out"]]


def param_shapes(arch) -> list[tuple[str, tuple]]:
    """(name, shape) of every parameter block, in the order of the flat vector."""
    out = []
    dims = [arch["F"], *arch["phi"]]
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        out += [(f"phi{i}.W", (a, b)), (f"phi{i}.b", (b,))]
    if arch["pool"] == "attn":
        H, dk, dv, e = arch["heads"], arch["dk"], arch["dv"], arch["phi"][-1]
        q_in = arch["n_ego"] + arch["n_extra"]
        out += [("q.W", (q_in, H * dk)), ("q.b", (H * dk,)), ("k.W", (e, H * dk)), ("v.W", (e, H * dv)),
                ("v.b", (H * dv,))]
    ts = trunk_sizes(arch)
    for i, (a, b) in enumerate(zip(ts[:-1], ts[1:])):
        out += [(f"trunk{i}.W", (a, b)), (f"trunk{i}.b", (b,))]
    return out


def n_params(arch) -> int:
    return sum(int(np.prod(s)) for _, s in param_shapes(arch))


def unpack(p, arch) -> dict:
    """name -> view into the flat vector p (numpy array or torch tensor)."""
    out, i = {}, 0
    for name, shape in param_shapes(arch):
        n = int(np.prod(shape))
        out[name] = p[i : i + n].reshape(shape)
        i += n
    assert i == p.shape[-1], f"param size {p.shape[-1]} != {i}"
    return out


def init_params(arch, rng=None, out_scale=0.1) -> np.ndarray:
    """Scaled-normal init (1/sqrt(fan_in)), biases 0, small output layer (as policy.init_params)."""
    rng = np.random.default_rng(rng)
    last = f"trunk{len(arch['hidden'])}.W"
    parts = []
    for name, shape in param_shapes(arch):
        if name.endswith(".b"):
            parts.append(np.zeros(shape).ravel())
        else:
            s = 1.0 / np.sqrt(shape[0]) * (out_scale if name == last else 1.0)
            parts.append(rng.normal(0, s, shape).ravel())
    return np.concatenate(parts)


def forward(p: np.ndarray, arch, x: np.ndarray) -> np.ndarray:
    """Linear output (..., n_out) for (scaled) inputs x (..., in_dim(arch))."""
    P = unpack(p, arch)
    E, K, F = arch["n_ego"], arch["K"], arch["F"]
    lead = x.shape[:-1]
    ego = np.concatenate([x[..., :E], x[..., E + K * F :]], -1)
    nb = x[..., E : E + K * F].reshape(*lead, K, F)
    mask = nb[..., arch["present"]] != 0  # (..., K)
    h = nb.reshape(-1, F)  # 2-D matmuls (numpy's broadcast 3-D matmul is much slower)
    for i in range(len(arch["phi"])):
        h = np.tanh(h @ P[f"phi{i}.W"] + P[f"phi{i}.b"])
    any_ = mask.any(-1, keepdims=True)
    if arch["pool"] == "attn":
        H, dk, dv = arch["heads"], arch["dk"], arch["dv"]
        q = (ego @ P["q.W"] + P["q.b"]).reshape(*lead, H, dk)
        k = (h @ P["k.W"]).reshape(*lead, K, H, dk)
        v = (h @ P["v.W"] + P["v.b"]).reshape(*lead, K, H, dv)
        s = np.einsum("...hd,...khd->...hk", q, k) / np.sqrt(dk)
        s = np.where(mask[..., None, :], s, NEG)
        w = np.exp(s - s.max(-1, keepdims=True)) * mask[..., None, :]
        w = w / np.maximum(w.sum(-1, keepdims=True), 1e-30)
        pooled = np.einsum("...hk,...khd->...hd", w, v).reshape(*lead, H * dv)
    elif arch["pool"] == "mean":
        h = h.reshape(*lead, K, -1)
        pooled = (h * mask[..., None]).sum(-2) / np.maximum(mask.sum(-1, keepdims=True), 1)
    else:
        h = h.reshape(*lead, K, -1)
        pooled = np.where(any_, np.where(mask[..., None], h, -2.0).max(-2), 0.0)
    z = np.concatenate([ego, pooled], -1)
    n = len(arch["hidden"])
    for i in range(n + 1):
        z = z @ P[f"trunk{i}.W"] + P[f"trunk{i}.b"]
        if i < n:
            z = np.tanh(z)
    return z
