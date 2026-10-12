"""Running checkpoints that were trained with an older sim.

- Observation layout: checkpoints trained with an older observation layout (listed in
  ``LEGACY_LAYOUTS``) are adapted by feature name (``adapt_inputs``): the inputs that did
  not exist yet get zero weights and rescaled inputs get their factor folded into
  ``in_scale``, so the network computes exactly what it computed before and simply
  ignores the new sensors.
- Interface and world: sensors (incl. the neighbour layout: brake light + hazard or the legacy
  acceleration), the lamp count, the action interface, the crossing geometry and tyre grip
  are part of what a policy learned. ``LEGACY_INTERFACE`` holds the values of checkpoints
  saved before a field existed (``PolicyDriver.load`` applies them); ``physics_of`` gives
  the physics the browser runs a policy with.
"""

import json
from pathlib import Path

import numpy as np

from . import sim as simmod
from .policy import export_json
from .sim import SimConfig, act_dim, obs_dim

# ------------------------------------------------------------------ observation layout
# Feature keys of sim.observe, in order (own block, then one block per neighbour followed
# by its lamps). The site's labels (site/obs_labels.py) are keyed by these.
OWN_KEYS = ["pos_x", "pos_y", "dir_x", "dir_y", "speed", "goal_right", "goal_straight", "goal_left", "exit_fwd",
            "exit_left", "mouth_fwd", "mouth_left", "ray_r60", "ray_r30", "ray_0", "ray_l30", "ray_l60", "kappa",
            "kappa_grip"]
NEIGHBOUR_KEYS = ["fwd", "left", "same", "cross", "speed", "present", "kappa", "brake", "hazard"]
# cfg.brake_light = False (checkpoints trained before 2026-10-11): continuous acceleration, no hazard
NEIGHBOUR_KEYS_LEGACY = ["fwd", "left", "same", "cross", "speed", "present", "kappa", "acc"]


def nb_keys(cfg: SimConfig) -> list[str]:
    """Per-neighbour feature keys (before the lamps) of the cfg's observation interface."""
    return NEIGHBOUR_KEYS if cfg.brake_light else NEIGHBOUR_KEYS_LEGACY

# Observation layouts of older sim versions, used to adapt old checkpoints by feature key.
# "factor": old value = new value * factor (folded into in_scale, so the network computes
# exactly what it computed before); keys missing from the old layout get weight 0.
_OLD_OWN = ["pos_x", "pos_y", "dir_x", "dir_y", "speed", "goal_right", "goal_straight", "goal_left", "exit_fwd",
            "exit_left", "ray_r60", "ray_r30", "ray_0", "ray_l30", "ray_l60"]
_OLD_NB = ["fwd", "left", "same", "cross", "speed", "present"]
# before the wide crossing: own position / L and goal / L with L = 50 m
_OLD_FACTOR = {"pos_x": simmod.POS_SCALE / 50.0, "pos_y": simmod.POS_SCALE / 50.0,
               "exit_fwd": simmod.GOAL_SCALE / 50.0, "exit_left": simmod.GOAL_SCALE / 50.0}
LEGACY_LAYOUTS = [
    {"own": _OLD_OWN + ["kappa"], "nb": _OLD_NB, "factor": _OLD_FACTOR},  # ppo*, es after the steering-rate limit
    {"own": _OLD_OWN, "nb": _OLD_NB, "factor": _OLD_FACTOR},  # es1-es5 (before the steering-rate limit)
]


def neighbour_keys(cfg: SimConfig, nb: list[str] | None = None) -> list[str]:
    keys = (nb or nb_keys(cfg)) + [f"lamp{q + 1}" for q in range(cfg.n_signal)]
    return [f"n{k + 1}_{f}" for k in range(cfg.n_neighbours) for f in keys]


def obs_keys(cfg: SimConfig, own: list[str] | None = None, nb: list[str] | None = None) -> list[str]:
    return (own or OWN_KEYS) + neighbour_keys(cfg, nb)


def adapt_inputs(d: dict, cfg: SimConfig) -> dict:
    """Map a policy trained on a legacy observation layout onto the current one by
    feature key (new inputs get zero weights and scale 1; rescaled inputs get their
    factor folded into in_scale). No-op if sizes match."""
    n_in, want = d["sizes"][0], obs_dim(cfg)
    if n_in == want:
        return d
    if "arch" in d:  # set encoders exist only with the current layout
        raise ValueError(f"set-encoder policy has {n_in} inputs; the sim gives {want}")
    cur = obs_keys(cfg)
    for legacy in LEGACY_LAYOUTS:
        old = obs_keys(cfg, legacy["own"], legacy["nb"])
        if len(old) != n_in:
            continue
        W0 = np.asarray(d["layers"][0]["W"], dtype=float)
        scale = np.asarray(d.get("in_scale") or np.ones(n_in), dtype=float)
        W = np.zeros((want, W0.shape[1]))
        s = np.ones(want)
        pos = {k: j for j, k in enumerate(cur)}
        for j, k in enumerate(old):
            W[pos[k]] = W0[j]
            s[pos[k]] = scale[j] * legacy["factor"].get(k, 1.0)
        missing = [k for k in cur if k not in set(old) and not k[1:2].isdigit()]
        missing_nb = sorted({k.split("_", 1)[1] for k in cur if k not in set(old) and k[1:2].isdigit()})
        d = {**d, "sizes": [want, *d["sizes"][1:]], "in_scale": s.tolist(),
             "layers": [{"W": W.tolist(), "b": d["layers"][0]["b"]}, *d["layers"][1:]],
             "adapted": f"trained without input(s) {', '.join(missing + ['neighbour ' + k for k in missing_nb])} (weights 0)"}
        return d
    raise ValueError(f"policy has {n_in} inputs; the sim gives {want} and no legacy layout matches")


def load_adapted_policy(path: str | Path) -> dict:
    """Policy dict (policy.export_json format) of a .npz checkpoint or .json export, mapped
    onto the current observation layout; raises ValueError if it cannot run in this sim."""
    path = Path(path)
    if path.suffix == ".npz":
        d = export_json(path)
    else:
        d = json.loads(path.read_text(encoding="utf-8"))
        d = d.get("policy", d)  # accept a {name, policy} wrapper
    sizes = d.get("sizes") or [len(d["layers"][0]["W"]), *[len(layer["b"]) for layer in d["layers"]]]
    cfg = interface_cfg(d.get("cfg") or {})  # the checkpoint's own lamp count / neighbour layout
    d["sizes"] = sizes
    if sizes[-1] != act_dim(cfg):
        raise ValueError(f"{path}: {sizes[-1]} outputs, the sim expects act_dim {act_dim(cfg)}")
    try:
        d = adapt_inputs(d, cfg)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    return d


# ------------------------------------------------------------------ interface and physics
# SimConfig fields a checkpoint is run with (value for checkpoints saved before the field
# existed): its sensors / action interface, and the crossing geometry + tyre grip it was
# trained on (checkpoints from before the wide crossing ran on the 7 m road without grip limit
# and without reverse gear)
# n_signal (lamp count) and brake_light (neighbour block layout) are interface values too: checkpoints
# from before 2026-10-11 have 2 lamps and see the neighbours' acceleration. Traffic / world settings
# that are not part of the interface (spawn_right, wreck_time, demand) are NOT taken from a checkpoint.
LEGACY_INTERFACE = {"neighbour_vis": 30.0, "speed_control": False, "n_signal": 2, "brake_light": False,
                    "n_neighbours": simmod.K_NEIGHBOURS, "L": 50.0, "W": 3.5, "Rc": 6.0, "a_lat_max": 0.0, "v_reverse": 0.0}
INTERFACE_KEYS = ("n_signal", "brake_light", "n_neighbours")  # what changes the observation / action sizes


def interface_cfg(train_cfg: dict) -> SimConfig:
    """SimConfig with the observation / action interface (lamps, neighbour layout) of a checkpoint cfg."""
    return SimConfig(**{k: type(LEGACY_INTERFACE[k])(train_cfg.get(k, LEGACY_INTERFACE[k])) for k in INTERFACE_KEYS})


PHYSICS_KEYS = ["v_max", "acc_max", "brake_max", "kappa_max", "kappa_rate", "neighbour_vis", "speed_control",
                "a_lat_max", "v_reverse", "L", "W", "Rc", "n_signal", "brake_light", "n_neighbours"]
# for the browser also kappa_rate: absent = trained before the steering-rate limit (steering acts instantly)
LEGACY_PHYSICS = {**LEGACY_INTERFACE, "kappa_rate": 1e6}


def physics_of(train_cfg: dict) -> dict:
    """Vehicle physics and observation interface a policy was trained with (to run it the way it
    learned). Numbers are floats, except n_signal (int) and brake_light (bool)."""
    fields = SimConfig.__dataclass_fields__
    out = {}
    for k in PHYSICS_KEYS:
        if k not in fields:
            continue
        v = train_cfg[k] if k in train_cfg else LEGACY_PHYSICS.get(k, getattr(SimConfig, k))
        out[k] = int(v) if k in ("n_signal", "n_neighbours") else bool(v) if k == "brake_light" else float(v)
    return out


# ------------------------------------------------------------------ warm start across interfaces
def transplant_actor(npz_path: str | Path, cfg: SimConfig, in_scale: np.ndarray, sizes: list[int]) -> np.ndarray:
    """Flat actor params (policy.py layout) for ``cfg``'s interface from an older checkpoint
    (e.g. 2 lamps + neighbour acceleration -> 1 lamp + brake light + hazard), for a PPO warm start.

    Inputs are matched by feature key (scale folded in so the first layer sees the same
    pre-activations); inputs the old brain did not have (brake, hazard, ...) start at weight 0
    (the old acceleration input is dropped, not remapped); outputs: accel, curvature and the
    first lamps are kept, extra old lamp outputs are dropped, missing ones start at 0."""
    from .policy import load_in_scale, load_npz, unpack

    params, old_sizes, old_cfg = load_npz(npz_path)
    old_s = load_in_scale(npz_path)
    old_s = np.ones(old_sizes[0]) if old_s is None else old_s
    if list(old_sizes[1:-1]) != list(sizes[1:-1]):
        raise ValueError(f"hidden sizes differ: {old_sizes} vs {sizes}")
    old_icfg = interface_cfg(old_cfg)
    old_keys, new_keys = obs_keys(old_icfg), obs_keys(cfg)
    assert len(old_keys) == old_sizes[0] and len(new_keys) == sizes[0]
    Ls = unpack(params, old_sizes)
    W0, b0 = np.asarray(Ls[0][0]), np.asarray(Ls[0][1])
    W = np.zeros((sizes[0], W0.shape[1]))
    pos = {k: j for j, k in enumerate(old_keys)}
    for j, k in enumerate(new_keys):
        if k in pos and in_scale[j] != 0:
            W[j] = W0[pos[k]] * old_s[pos[k]] / in_scale[j]
    WL, bL = np.asarray(Ls[-1][0]), np.asarray(Ls[-1][1])
    n = min(sizes[-1], old_sizes[-1])
    WN, bN = np.zeros((WL.shape[0], sizes[-1])), np.zeros(sizes[-1])
    WN[:, :n], bN[:n] = WL[:, :n], bL[:n]
    layers = [(W, b0), *[(np.asarray(w), np.asarray(b)) for w, b in Ls[1:-1]], (WN, bN)]
    return np.concatenate([x for w, b in layers for x in (w.ravel(), b.ravel())])
