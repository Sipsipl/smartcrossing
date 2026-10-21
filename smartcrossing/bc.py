"""Behaviour cloning of an MLP brain into a new actor (``ppo --init-bc TEACHER.npz``): a set-encoder actor
(``--arch set``) or an MLP of other sizes (``--arch mlp --hidden ...``).

There is no function-preserving map from the MLP to the set encoder (or a wider MLP), so the new actor imitates the
teacher instead: the teacher drives a numpy Sim in the training world (its own cars, so the states
are the ones it visits), every car's observation is recorded, and the student's linear output (the
PPO mean in pre-squash space) is fitted to the teacher's pre-tanh output by MSE (Adam, torch).

The student may see more neighbour slots than the teacher (``SimConfig.n_neighbours`` > 4): the
slots are sorted by distance, so the teacher's observation is the first 4 blocks of the student's.
Everything else of the interface (lamps, neighbour features) must match.
"""

import numpy as np

from .policy import load_arch, load_in_scale, load_npz, mlp_linear
from .policy_compat import INTERFACE_KEYS, LEGACY_INTERFACE
from .sim import OBS_DIM, Sim, SimConfig, nb_features


def collect(teacher: str, cfg: SimConfig, envs=32, seconds=120.0, demand=(0.1, 0.3), seed=0, every=2):
    """(student obs (M, obs_dim(cfg)), teacher mu (M, A)) from the teacher driving in cfg's world."""
    params, sizes, tcfg = load_npz(teacher)
    if load_arch(teacher):
        raise ValueError("the teacher must be an MLP checkpoint")
    for k in INTERFACE_KEYS:
        if k != "n_neighbours" and type(LEGACY_INTERFACE[k])(tcfg.get(k, LEGACY_INTERFACE[k])) != getattr(cfg, k):
            raise ValueError(f"teacher interface {k} differs from the training world")
    K_t = int(tcfg.get("n_neighbours", LEGACY_INTERFACE["n_neighbours"]))
    if K_t > cfg.n_neighbours:
        raise ValueError(f"teacher sees {K_t} neighbours, the student only {cfg.n_neighbours}")
    scale_t = load_in_scale(teacher)
    scale_t = np.ones(sizes[0]) if scale_t is None else scale_t
    n_t = OBS_DIM + K_t * (nb_features(cfg) + cfg.n_signal)
    if n_t != sizes[0]:
        raise ValueError(f"teacher input size {sizes[0]} does not match its interface ({n_t})")
    world = SimConfig(**{**cfg.__dict__, "demand_range": tuple(demand)})
    sim = Sim(world, envs, seed=seed)
    X, Y = [], []
    for t in range(int(round(seconds / cfg.dt))):
        obs = sim.observe()
        al = sim.alive
        act = np.zeros((*al.shape, sizes[-1]))
        if al.any():
            o = obs[al]
            mu = mlp_linear(params, sizes, o[:, :n_t] * scale_t)
            act[al] = np.tanh(mu)
            if t % every == 0:
                X.append(o)
                Y.append(mu)
        sim.step(act)
    return np.concatenate(X), np.concatenate(Y)


def student_forward(p, arch, x):
    """Linear output of the student: a set encoder (``arch`` dict) or an MLP (``arch`` = layer sizes)."""
    from .ppo_torch import linear_forward

    return linear_forward(p, None, x, arch) if isinstance(arch, dict) else linear_forward(p, list(arch), x)


def fit(arch, p0, x, y, steps=3000, batch=4096, lr=1e-3, device="cpu", seed=0, log=None):
    """Adam on MSE(student linear output, teacher mu); x already scaled with the student's in_scale.
    ``arch``: set-encoder arch dict or MLP layer sizes."""
    import torch

    dev = torch.device(device)
    X = torch.as_tensor(x, dtype=torch.float32, device=dev)
    Y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    p = torch.tensor(p0, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    gen = torch.Generator(device=dev).manual_seed(seed)
    n = len(X)
    for s in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=gen, device=dev)
        loss = torch.mean((student_forward(p, arch, X[idx]) - Y[idx]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and (s % 500 == 0 or s == steps - 1):
            log(f"bc step {s}: mse {float(loss):.4f}")
    with torch.no_grad():
        full = float(torch.mean((student_forward(p, arch, X) - Y) ** 2))
    return p.detach().cpu().double().numpy(), full


def bc_actor(teacher: str, cfg: SimConfig, arch, in_scale, p0, steps=3000, seconds=120.0, device="cpu", seed=0,
             log=print):
    """Actor params (set encoder: ``arch`` dict; MLP: ``arch`` = layer sizes) cloned from the MLP checkpoint
    ``teacher`` (see module doc)."""
    x, y = collect(teacher, cfg, seconds=seconds, seed=seed)
    log(f"bc: {len(x)} car-steps from {teacher}; teacher mu var {np.var(y, 0).round(3).tolist()}")
    p, mse = fit(arch, p0, x * in_scale, y, steps=steps, device=device, seed=seed, log=log)
    log(f"bc: final mse {mse:.4f}, R2 {1 - mse / float(np.mean(np.var(y, 0))):.3f}")
    return p
