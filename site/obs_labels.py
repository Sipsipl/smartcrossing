"""German and English labels / groups / display units for every entry of the observation vector
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

# Own block of sim.observe, in order: (key, (group de, group en), (label de, label en), display scale, unit).
# "scale" converts the (normalised) observation back into the unit shown on the site (speeds in km/h);
# a callable gets cfg.
OWN_FEATURES = [
    ("pos_x", ("Position", "Position"), ("quer (rechts +)", "sideways (right +)"), simmod.POS_SCALE, "m"),
    ("pos_y", ("Position", "Position"), ("längs (Richtung Mitte +)", "along the road (towards the middle +)"), simmod.POS_SCALE, "m"),
    ("dir_x", ("Richtung", "Heading"), ("Anteil nach rechts", "share pointing right"), 1, ""),
    ("dir_y", ("Richtung", "Heading"), ("Anteil nach vorn", "share pointing forward"), 1, ""),
    ("speed", ("Tempo", "Speed"), ("Tempo", "Speed"), lambda c: c.v_max * 3.6, "km/h"),
    ("goal_right", ("Ziel", "Goal"), ("rechts abbiegen", "turn right"), 1, "bool"),
    ("goal_straight", ("Ziel", "Goal"), ("geradeaus", "straight on"), 1, "bool"),
    ("goal_left", ("Ziel", "Goal"), ("links abbiegen", "turn left"), 1, "bool"),
    ("exit_fwd", ("Weg zur Ausfahrt", "Path to my exit"), ("vor mir", "ahead of me"), simmod.GOAL_SCALE, "m"),
    ("exit_left", ("Weg zur Ausfahrt", "Path to my exit"), ("links von mir", "left of me"), simmod.GOAL_SCALE, "m"),
    ("mouth_fwd", ("Weg zur Ausfahrtsstraße", "Path to the exit road"), ("vor mir", "ahead of me"), simmod.GOAL_SCALE, "m"),
    ("mouth_left", ("Weg zur Ausfahrtsstraße", "Path to the exit road"), ("links von mir", "left of me"), simmod.GOAL_SCALE, "m"),
    ("ray_r60", ("Abstand zum Fahrbahnrand", "Distance to the road edge"), ("60° rechts", "60° right"), simmod.RAY_RANGE, "m"),
    ("ray_r30", ("Abstand zum Fahrbahnrand", "Distance to the road edge"), ("30° rechts", "30° right"), simmod.RAY_RANGE, "m"),
    ("ray_0", ("Abstand zum Fahrbahnrand", "Distance to the road edge"), ("geradeaus", "straight ahead"), simmod.RAY_RANGE, "m"),
    ("ray_l30", ("Abstand zum Fahrbahnrand", "Distance to the road edge"), ("30° links", "30° left"), simmod.RAY_RANGE, "m"),
    ("ray_l60", ("Abstand zum Fahrbahnrand", "Distance to the road edge"), ("60° links", "60° left"), simmod.RAY_RANGE, "m"),
    ("kappa", ("Lenkrad", "Steering wheel"), ("aktueller Lenkeinschlag (links +)", "current steering angle (left +)"), 100, "%"),
    ("kappa_grip", ("Lenkrad", "Steering wheel"),
     ("möglicher Einschlag bei diesem Tempo (Haftung)", "possible angle at this speed (grip)"), 100, "%"),
]
# neighbour features: (key, (label de, label en), display scale, unit)
NEIGHBOUR_FEATURES = [
    ("fwd", ("vor mir", "ahead of me"), lambda c: simmod.NEIGHBOUR_RANGE, "m"),
    ("left", ("links von mir", "left of me"), lambda c: simmod.NEIGHBOUR_RANGE, "m"),
    ("same", ("gleiche Richtung (+1) / entgegen (−1)", "same direction (+1) / oncoming (−1)"), 1, ""),
    ("cross", ("fährt quer: nach links (+) / rechts (−)", "crossing: to the left (+) / right (−)"), 1, ""),
    ("speed", ("Tempo", "Speed"), lambda c: c.v_max * 3.6, "km/h"),
    ("present", ("da?", "there?"), 1, "bool"),
    ("kappa", ("lenkt (links +)", "steering (left +)"), 100, "%"),
    ("brake", ("Bremslicht an", "brake light on"), 1, "bool"),
    ("hazard", ("Warnblinker an (Unfallwrack)", "hazard lights on (crash wreck)"), 1, "bool"),
]
# checkpoints from before 2026-10-11 (cfg.brake_light = False): the neighbour's acceleration instead
NEIGHBOUR_FEATURES_LEGACY = NEIGHBOUR_FEATURES[:7] + [
    ("acc", ("beschleunigt (+) / bremst (−)", "accelerating (+) / braking (−)"), lambda c: c.brake_max, "m/s²"),
]


def interface_key(cfg) -> str:
    """Name of an observation interface, e.g. "1L-brake" (one lamp, brake light + hazard) or
    "2L-acc" (the old checkpoints); the pages pick the labels for a sim by this key."""
    n_signal = cfg["n_signal"] if isinstance(cfg, dict) else cfg.n_signal
    brake = cfg["brake_light"] if isinstance(cfg, dict) else cfg.brake_light
    k = int((cfg.get("n_neighbours") if isinstance(cfg, dict) else cfg.n_neighbours) or simmod.K_NEIGHBOURS)
    return f"{int(n_signal)}L-{'brake' if brake else 'acc'}" + (f"-K{k}" if k != simmod.K_NEIGHBOURS else "")


def _scale(s, cfg):
    return float(s(cfg)) if callable(s) else float(s)


def obs_layout(cfg: SimConfig) -> dict:
    """Labels (German; English in the *_en fields) for the observation vector, grouped for display."""
    if [k for k, *_ in OWN_FEATURES] != OWN_KEYS or len(OWN_FEATURES) != simmod.OBS_DIM:
        raise RuntimeError(
            f"site/obs_labels.py OWN_FEATURES has {len(OWN_FEATURES)} labels but sim.OBS_DIM is "
            f"{simmod.OBS_DIM}: the observation layout changed, update the labels"
        )
    nb_feats = NEIGHBOUR_FEATURES if cfg.brake_light else NEIGHBOUR_FEATURES_LEGACY
    blk = nb_features(cfg) + cfg.n_signal
    if ([k for k, *_ in nb_feats] != nb_keys(cfg) or len(nb_feats) + cfg.n_signal != blk
            or simmod.OBS_DIM + cfg.n_neighbours * blk != obs_dim(cfg)):
        raise RuntimeError("neighbour block layout changed: update NEIGHBOUR_FEATURES in site/obs_labels.py")
    groups, i = [], 0

    def add(group, label, key, scale, unit, sub=None):
        """group / label: (German, English)."""
        nonlocal i
        if not groups or groups[-1]["name"] != group[0]:
            groups.append({"name": group[0], "name_en": group[1], "sub": sub, "items": []})
        groups[-1]["items"].append({"i": i, "key": key, "label": label[0], "label_en": label[1], "scale": scale, "unit": unit})
        i += 1

    for key, group, label, scale, unit in OWN_FEATURES:
        add(group, label, key, _scale(scale, cfg), unit)
    for k in range(cfg.n_neighbours):
        nb = (f"Nachbar {k + 1}", f"Neighbour {k + 1}")
        for key, label, scale, unit in nb_feats:
            add(nb, label, f"n{k + 1}_{key}", _scale(scale, cfg), unit, sub="neighbour")
        for q in range(cfg.n_signal):
            add(nb, ("Lampe", "Lamp") if cfg.n_signal == 1 else (f"Lampe {q + 1}", f"Lamp {q + 1}"), f"n{k + 1}_lamp{q + 1}", 1.0, "",
                sub="neighbour")
    assert i == obs_dim(cfg)
    return {
        "key": interface_key(cfg),
        "n_signal": cfg.n_signal,
        "brake_light": bool(cfg.brake_light),
        "dim": i,
        "own_dim": simmod.OBS_DIM,
        "neighbour_block": blk,
        "k_neighbours": cfg.n_neighbours,
        "ray_angles_deg": [round(float(a), 3) for a in np.degrees(simmod.RAY_ANGLES)],
        "ray_range": float(simmod.RAY_RANGE),
        "neighbour_range": float(simmod.NEIGHBOUR_RANGE),
        "keys": obs_keys(cfg),
        "groups": groups,
        "actions": ["Gas / Bremse", "Lenkung"] + (["Lampe"] if cfg.n_signal == 1 else [f"Lampe {q + 1}" for q in range(cfg.n_signal)]),
        "actions_en": ["Accelerate / Brake", "Steering"] + (["Lamp"] if cfg.n_signal == 1 else [f"Lamp {q + 1}" for q in range(cfg.n_signal)]),
    }
