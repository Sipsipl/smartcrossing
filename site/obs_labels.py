"""German labels / groups / display units for every entry of the observation vector
(site/build.py exports them as SC_DATA.layout for the pages).

The labels are keyed by the feature keys of ``policy_compat.OWN_KEYS`` /
``nb_keys(cfg)`` and checked against ``sim.OBS_DIM`` / ``obs_dim(cfg)``, so a layout
change in the sim fails loudly here instead of silently mislabelling. The neighbour block
depends on the policy's interface (cfg.brake_light: brake light + hazard lights, or the legacy
acceleration of the old checkpoints) and the lamp count (cfg.n_signal); ``obs_layout(cfg)``
builds the labels for one interface, ``interface_key(cfg)`` names it for the pages.
"""

import numpy as np

from smartcrossing import sim as simmod
from smartcrossing.policy_compat import OWN_KEYS, nb_keys, obs_keys
from smartcrossing.sim import SimConfig, nb_features, obs_dim

# Own block of sim.observe, in order: (key, group, label, display scale, unit).
# "scale" converts the (normalised) observation back into the unit shown on the site;
# a callable gets cfg.
OWN_FEATURES = [
    ("pos_x", "Position", "quer (rechts +)", simmod.POS_SCALE, "m"),
    ("pos_y", "Position", "längs (Richtung Mitte +)", simmod.POS_SCALE, "m"),
    ("dir_x", "Richtung", "Anteil nach rechts", 1, ""),
    ("dir_y", "Richtung", "Anteil nach vorn", 1, ""),
    ("speed", "Tempo", "Tempo", lambda c: c.v_max * 3.6, "km/h"),
    ("goal_right", "Ziel", "rechts abbiegen", 1, "bool"),
    ("goal_straight", "Ziel", "geradeaus", 1, "bool"),
    ("goal_left", "Ziel", "links abbiegen", 1, "bool"),
    ("exit_fwd", "Weg zur Ausfahrt", "vor mir", simmod.GOAL_SCALE, "m"),
    ("exit_left", "Weg zur Ausfahrt", "links von mir", simmod.GOAL_SCALE, "m"),
    ("mouth_fwd", "Weg zur Ausfahrtsstraße", "vor mir", simmod.GOAL_SCALE, "m"),
    ("mouth_left", "Weg zur Ausfahrtsstraße", "links von mir", simmod.GOAL_SCALE, "m"),
    ("ray_r60", "Abstand zum Fahrbahnrand", "60° rechts", simmod.RAY_RANGE, "m"),
    ("ray_r30", "Abstand zum Fahrbahnrand", "30° rechts", simmod.RAY_RANGE, "m"),
    ("ray_0", "Abstand zum Fahrbahnrand", "geradeaus", simmod.RAY_RANGE, "m"),
    ("ray_l30", "Abstand zum Fahrbahnrand", "30° links", simmod.RAY_RANGE, "m"),
    ("ray_l60", "Abstand zum Fahrbahnrand", "60° links", simmod.RAY_RANGE, "m"),
    ("kappa", "Lenkrad", "aktueller Lenkeinschlag (links +)", 100, "%"),
    ("kappa_grip", "Lenkrad", "möglicher Einschlag bei diesem Tempo (Haftung)", 100, "%"),
]
NEIGHBOUR_FEATURES = [
    ("fwd", "vor mir", lambda c: simmod.NEIGHBOUR_RANGE, "m"),
    ("left", "links von mir", lambda c: simmod.NEIGHBOUR_RANGE, "m"),
    ("same", "gleiche Richtung (+1) / entgegen (−1)", 1, ""),
    ("cross", "fährt quer: nach links (+) / rechts (−)", 1, ""),
    ("speed", "Tempo", lambda c: c.v_max * 3.6, "km/h"),
    ("present", "da?", 1, "bool"),
    ("kappa", "lenkt (links +)", 100, "%"),
    ("brake", "Bremslicht an", 1, "bool"),
    ("hazard", "Warnblinker an (Unfallwrack)", 1, "bool"),
]
# checkpoints from before 2026-10-11 (cfg.brake_light = False): the neighbour's acceleration instead
NEIGHBOUR_FEATURES_LEGACY = NEIGHBOUR_FEATURES[:7] + [
    ("acc", "beschleunigt (+) / bremst (−)", lambda c: c.brake_max, "m/s²"),
]


def interface_key(cfg) -> str:
    """Name of an observation interface, e.g. "1L-brake" (one lamp, brake light + hazard) or
    "2L-acc" (the old checkpoints); the pages pick the labels for a sim by this key."""
    n_signal = cfg["n_signal"] if isinstance(cfg, dict) else cfg.n_signal
    brake = cfg["brake_light"] if isinstance(cfg, dict) else cfg.brake_light
    return f"{int(n_signal)}L-{'brake' if brake else 'acc'}"


def _scale(s, cfg):
    return float(s(cfg)) if callable(s) else float(s)


def obs_layout(cfg: SimConfig) -> dict:
    """German labels for the observation vector, grouped for display."""
    if [k for k, *_ in OWN_FEATURES] != OWN_KEYS or len(OWN_FEATURES) != simmod.OBS_DIM:
        raise RuntimeError(
            f"site/obs_labels.py OWN_FEATURES has {len(OWN_FEATURES)} labels but sim.OBS_DIM is "
            f"{simmod.OBS_DIM}: the observation layout changed, update the labels"
        )
    nb_feats = NEIGHBOUR_FEATURES if cfg.brake_light else NEIGHBOUR_FEATURES_LEGACY
    blk = nb_features(cfg) + cfg.n_signal
    if ([k for k, *_ in nb_feats] != nb_keys(cfg) or len(nb_feats) + cfg.n_signal != blk
            or simmod.OBS_DIM + simmod.K_NEIGHBOURS * blk != obs_dim(cfg)):
        raise RuntimeError("neighbour block layout changed: update NEIGHBOUR_FEATURES in site/obs_labels.py")
    groups, i = [], 0

    def add(group, label, key, scale, unit, sub=None):
        nonlocal i
        if not groups or groups[-1]["name"] != group:
            groups.append({"name": group, "sub": sub, "items": []})
        groups[-1]["items"].append({"i": i, "key": key, "label": label, "scale": scale, "unit": unit})
        i += 1

    for key, group, label, scale, unit in OWN_FEATURES:
        add(group, label, key, _scale(scale, cfg), unit)
    for k in range(simmod.K_NEIGHBOURS):
        for key, label, scale, unit in nb_feats:
            add(f"Nachbar {k + 1}", label, f"n{k + 1}_{key}", _scale(scale, cfg), unit, sub="neighbour")
        for q in range(cfg.n_signal):
            add(f"Nachbar {k + 1}", "Lampe" if cfg.n_signal == 1 else f"Lampe {q + 1}", f"n{k + 1}_lamp{q + 1}", 1.0, "",
                sub="neighbour")
    assert i == obs_dim(cfg)
    return {
        "key": interface_key(cfg),
        "n_signal": cfg.n_signal,
        "brake_light": bool(cfg.brake_light),
        "dim": i,
        "own_dim": simmod.OBS_DIM,
        "neighbour_block": blk,
        "k_neighbours": simmod.K_NEIGHBOURS,
        "ray_angles_deg": [round(float(a), 3) for a in np.degrees(simmod.RAY_ANGLES)],
        "ray_range": float(simmod.RAY_RANGE),
        "neighbour_range": float(simmod.NEIGHBOUR_RANGE),
        "keys": obs_keys(cfg),
        "groups": groups,
        "actions": ["Gas / Bremse", "Lenkung"] + (["Lampe"] if cfg.n_signal == 1 else [f"Lampe {q + 1}" for q in range(cfg.n_signal)]),
    }
