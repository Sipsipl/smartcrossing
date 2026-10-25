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

Optional recurrent memory (``arch["mem"] = {"type": "gru", "size": Hm}``, actor only; absent = stateless, so
every older checkpoint is unchanged): a GRU cell on the trunk input z = [ego, extra, pooled] (the same
observation, no new information) carries a per-car state h (Hm,), 0 for a new car (new uid in its slot):

    r = s(z Wx_r + bx_r + h Wh_r + bh_r), u = s(z Wx_u + bx_u + h Wh_u + bh_u)      (gate order r, u, n
    n = tanh(z Wx_n + bx_n + r * (h Wh_n + bh_n)),  h' = (1 - u) n + u h              as in PyTorch)
    trunk layer 0: z W0 + h' Wm + b0           (Wm = ``mem.out.W``: the new state is a residual input)

The memory blocks come AFTER all others in the flat vector, so a stateless brain's vector is a prefix of
its memory version; ``upgrade_params`` appends them with Wm = 0, which drives exactly like the old brain.
``forward`` is the stateless pass (it refuses memory archs); ``forward_mem(p, arch, x, h)`` -> (out, h').

``arch`` (a JSON-able dict, stored in the checkpoint) fixes the layout; the parameters are one
flat vector in the order of ``param_shapes`` (each W row-major (in, out), then its bias), like
the MLP layout of policy.py. ``forward`` returns the linear output (PPO's pre-squash mean); the
deterministic policy is tanh of it (policy.forward). The torch twin is ppo_torch.set_forward and
the browser twin site/src/js/sim.js (parity tests: tests/test_policy_set.py, tests/test_policy_mem.py).
"""

import numpy as np

from .sim import OBS_DIM, SimConfig, act_dim, nb_features

POOLS = ("attn", "mean", "max")
NB_PRESENT = 5  # index of the `present` flag inside a neighbour block
NEG = -1e9  # attention logit of absent neighbours
MEM_OUT = "mem.out.W"  # memory -> first trunk layer (0 at init / upgrade: the memory starts without effect)
UPDATE_GATE_BIAS = 1.0  # initial bias of the GRU update gate (u ~ 0.73: the state is kept rather than overwritten)


def make_arch(cfg: SimConfig, hidden=(64, 64), phi=(32, 32), pool="attn", heads=4, dk=8, dv=8, n_extra=0,
              n_out=None, mem=0) -> dict:
    """mem > 0: a GRU memory of that size (module doc); 0 = stateless (no "mem" key: the format of older files)."""
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    arch = {"type": "set", "n_ego": OBS_DIM, "K": int(cfg.n_neighbours), "F": nb_features(cfg) + cfg.n_signal,
            "n_extra": int(n_extra), "present": NB_PRESENT, "phi": [int(h) for h in phi], "pool": pool,
            "heads": int(heads), "dk": int(dk), "dv": int(dv), "hidden": [int(h) for h in hidden],
            "n_out": int(act_dim(cfg) if n_out is None else n_out)}
    return with_mem(arch, mem)


def mem_size(arch) -> int:
    """Hm of a memory arch; 0 for a stateless one (also None = MLP)."""
    m = (arch or {}).get("mem")
    if not m:
        return 0
    if m.get("type") != "gru":
        raise ValueError(f"unknown memory type {m}")
    return int(m["size"])


def base_arch(arch) -> dict:
    """The arch without its memory (the stateless brain whose params are the prefix of this one's)."""
    return {k: v for k, v in arch.items() if k != "mem"}


def with_mem(arch, size) -> dict:
    """arch with a GRU memory of ``size`` (0: without memory)."""
    out = base_arch(arch)
    if size:
        out["mem"] = {"type": "gru", "size": int(size)}
    return out


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
    if Hm := mem_size(arch):  # memory blocks last: the stateless vector is a prefix
        out += [("gru.Wx", (ts[0], 3 * Hm)), ("gru.Wh", (Hm, 3 * Hm)), ("gru.bx", (3 * Hm,)), ("gru.bh", (3 * Hm,)),
                (MEM_OUT, (Hm, ts[1]))]
    return out


def n_params(arch) -> int:
    return sum(int(np.prod(s)) for _, s in param_shapes(arch))


def out_bias_index(arch) -> int:
    """Index of the first output bias (trunk{n}.b[0]) in the flat vector."""
    return n_params(base_arch(arch)) - arch["n_out"]


def unpack(p, arch) -> dict:
    """name -> view into the flat vector p (numpy array or torch tensor)."""
    out, i = {}, 0
    for name, shape in param_shapes(arch):
        n = int(np.prod(shape))
        out[name] = p[i : i + n].reshape(shape)
        i += n
    assert i == p.shape[-1], f"param size {p.shape[-1]} != {i}"
    return out


def _init_block(name, shape, arch, rng, out_scale) -> np.ndarray:
    if name in (MEM_OUT, "gru.bh"):
        return np.zeros(shape).ravel()
    if name == "gru.bx":
        b = np.zeros(shape)
        Hm = mem_size(arch)
        b[Hm : 2 * Hm] = UPDATE_GATE_BIAS
        return b
    if name.endswith(".b"):
        return np.zeros(shape).ravel()
    s = 1.0 / np.sqrt(shape[0]) * (out_scale if name == f"trunk{len(arch['hidden'])}.W" else 1.0)
    return rng.normal(0, s, shape).ravel()


def init_params(arch, rng=None, out_scale=0.1) -> np.ndarray:
    """Scaled-normal init (1/sqrt(fan_in)), biases 0, small output layer (as policy.init_params); memory: GRU
    weights 1/sqrt(fan_in), update-gate bias UPDATE_GATE_BIAS, memory -> trunk weights 0."""
    rng = np.random.default_rng(rng)
    return np.concatenate([_init_block(name, shape, arch, rng, out_scale) for name, shape in param_shapes(arch)])


def upgrade_params(p, arch_old, arch_new, rng=None) -> np.ndarray:
    """Params of the memory brain ``arch_new`` that drives exactly like the stateless ``p`` (``arch_old``):
    p followed by fresh memory blocks (memory -> trunk weights 0, so the memory has no effect until trained)."""
    if mem_size(arch_old) or not mem_size(arch_new) or base_arch(arch_new) != arch_old:
        raise ValueError(f"cannot upgrade {arch_old} to {arch_new}: needs a stateless arch and the same arch + memory")
    if len(p) != n_params(arch_old):
        raise ValueError(f"param size {len(p)} != {n_params(arch_old)} of the stateless arch")
    rng = np.random.default_rng(rng)
    n_old = len(param_shapes(arch_old))
    new = [_init_block(name, shape, arch_new, rng, 1.0) for name, shape in param_shapes(arch_new)[n_old:]]
    return np.concatenate([np.asarray(p, dtype=float), *new])


def encode(P: dict, arch, x: np.ndarray) -> np.ndarray:
    """Trunk input z = [ego, extra, pooled] (..., trunk_sizes[0]) of (scaled) inputs x; P = unpack(p, arch)."""
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
    return np.concatenate([ego, pooled], -1)


def trunk(P: dict, arch, z: np.ndarray, g: np.ndarray | None = None) -> np.ndarray:
    """Linear output of the trunk on z; g: the new memory state (memory archs), a residual input of layer 0."""
    n = len(arch["hidden"])
    for i in range(n + 1):
        if i == 0 and g is not None:
            z = z @ P["trunk0.W"] + g @ P[MEM_OUT] + P["trunk0.b"]
        else:
            z = z @ P[f"trunk{i}.W"] + P[f"trunk{i}.b"]
        if i < n:
            z = np.tanh(z)
    return z


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_cell(P: dict, x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """New GRU state (..., Hm) for input x (..., Dz) and state h (..., Hm) (gate order r, u, n)."""
    Hm = h.shape[-1]
    gx = x @ P["gru.Wx"] + P["gru.bx"]
    gh = h @ P["gru.Wh"] + P["gru.bh"]
    r = _sigmoid(gx[..., :Hm] + gh[..., :Hm])
    u = _sigmoid(gx[..., Hm : 2 * Hm] + gh[..., Hm : 2 * Hm])
    n = np.tanh(gx[..., 2 * Hm :] + r * gh[..., 2 * Hm :])
    return (1.0 - u) * n + u * h


def forward(p: np.ndarray, arch, x: np.ndarray) -> np.ndarray:
    """Linear output (..., n_out) for (scaled) inputs x (..., in_dim(arch)); stateless archs only."""
    if mem_size(arch):
        raise ValueError("this set encoder has a memory: use forward_mem(p, arch, x, h)")
    P = unpack(p, arch)
    return trunk(P, arch, encode(P, arch, x))


def forward_mem(p: np.ndarray, arch, x: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(linear output (..., n_out), new state (..., Hm)) of a memory arch for inputs x and states h (zeros
    for a new car)."""
    P = unpack(p, arch)
    z = encode(P, arch, x)
    g = gru_cell(P, z, h)
    return trunk(P, arch, z, g), g
