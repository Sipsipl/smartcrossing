"""Bundle the browser viewer (and shared helpers for the website build).

Writes ``viewer/policies.js`` (the agent types the page starts with) and one
self-contained HTML file ``viewer/dist/smartcrossing.html`` with sim.js, app.js
and the policy weights inlined (no external requests), e.g.::

    poetry run python -m smartcrossing.export_viewer \
        --policy "Evolved A=runs/es5/final_g310.npz" --policy "Early B=runs/es2/stop.npz"

Up to three ``NAME=PATH`` policies (.npz from es.py or .json from
``policy.export_json``). Without ``--policy`` two random-weight placeholder
policies are used so the page still runs.

Also used by ``site/build.py``:

- ``obs_layout(cfg)``: German labels / groups / display units for every entry of
  the observation vector. The label list mirrors ``sim.observe`` and is checked
  against ``sim.OBS_DIM`` / ``obs_dim(cfg)``, so a layout change in the sim fails
  loudly here instead of silently mislabelling.
- ``load_policy``: checkpoints trained with an older observation layout (listed
  in ``LEGACY_LAYOUTS``) are adapted by feature name: the inputs that did not
  exist yet get zero weights, so the network computes exactly what it computed
  before and simply ignores the new sensor.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import sim as simmod
from .policy import default_in_scale, export_json, init_params, layer_sizes, unpack
from .sim import SimConfig, act_dim, obs_dim

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
MAX_TYPES = 3

# ------------------------------------------------------------------ observation layout
# Own block of sim.observe, in order: (key, group, label, display scale, unit).
# "scale" converts the (normalised) observation back into the unit shown on the site;
# a str scale names a SimConfig field (e.g. v_max), a callable gets cfg.
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
    ("acc", "beschleunigt (+) / bremst (−)", lambda c: c.brake_max, "m/s²"),
]
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


def _scale(s, cfg):
    return float(s(cfg)) if callable(s) else float(s)


def own_keys() -> list[str]:
    return [k for k, *_ in OWN_FEATURES]


def neighbour_keys(cfg: SimConfig, nb: list[str] | None = None) -> list[str]:
    keys = (nb or [k for k, *_ in NEIGHBOUR_FEATURES]) + [f"lamp{q + 1}" for q in range(cfg.n_signal)]
    return [f"n{k + 1}_{f}" for k in range(simmod.K_NEIGHBOURS) for f in keys]


def obs_keys(cfg: SimConfig, own: list[str] | None = None, nb: list[str] | None = None) -> list[str]:
    return (own or own_keys()) + neighbour_keys(cfg, nb)


def obs_layout(cfg: SimConfig) -> dict:
    """German labels for the observation vector, grouped for display."""
    if len(OWN_FEATURES) != simmod.OBS_DIM:
        raise RuntimeError(
            f"export_viewer.OWN_FEATURES has {len(OWN_FEATURES)} labels but sim.OBS_DIM is "
            f"{simmod.OBS_DIM}: the observation layout changed, update the labels"
        )
    blk = simmod.NB_FEATURES + cfg.n_signal
    if len(NEIGHBOUR_FEATURES) + cfg.n_signal != blk or simmod.OBS_DIM + simmod.K_NEIGHBOURS * blk != obs_dim(cfg):
        raise RuntimeError("neighbour block layout changed: update export_viewer.NEIGHBOUR_FEATURES")
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
        for key, label, scale, unit in NEIGHBOUR_FEATURES:
            add(f"Nachbar {k + 1}", label, f"n{k + 1}_{key}", _scale(scale, cfg), unit, sub="neighbour")
        for q in range(cfg.n_signal):
            add(f"Nachbar {k + 1}", f"Lampe {q + 1}", f"n{k + 1}_lamp{q + 1}", 1.0, "", sub="neighbour")
    assert i == obs_dim(cfg)
    return {
        "dim": i,
        "own_dim": simmod.OBS_DIM,
        "neighbour_block": blk,
        "k_neighbours": simmod.K_NEIGHBOURS,
        "ray_angles_deg": [round(float(a), 3) for a in np.degrees(simmod.RAY_ANGLES)],
        "ray_range": float(simmod.RAY_RANGE),
        "neighbour_range": float(simmod.NEIGHBOUR_RANGE),
        "keys": obs_keys(cfg),
        "groups": groups,
        "actions": ["Gas / Bremse", "Lenkung"] + [f"Lampe {q + 1}" for q in range(cfg.n_signal)],
    }


# ------------------------------------------------------------------ physics
PHYSICS_KEYS = ["v_max", "acc_max", "brake_max", "kappa_max", "kappa_rate", "neighbour_vis", "speed_control", "speed_tau",
                "a_lat_max", "L", "W", "Rc"]
# Value a physics field had *before* it existed in SimConfig (checkpoints trained
# with an older sim do not store it): kappa_rate absent = steering acts instantly.
# Checkpoints from before the wide crossing: 7 m road (W 3.5, Rc 6), no grip limit (a_lat_max 0).
LEGACY_PHYSICS = {"kappa_rate": 1e6, "neighbour_vis": 30.0, "speed_control": 0.0, "speed_tau": 0.5,
                  "a_lat_max": 0.0, "L": 50.0, "W": 3.5, "Rc": 6.0}


def physics_of(train_cfg: dict) -> dict:
    """Vehicle physics a policy was trained with (to run it the way it learned)."""
    fields = SimConfig.__dataclass_fields__
    out = {}
    for k in PHYSICS_KEYS:
        if k not in fields:
            continue
        out[k] = float(train_cfg[k]) if k in train_cfg else float(LEGACY_PHYSICS.get(k, getattr(SimConfig, k)))
    return out


# ------------------------------------------------------------------ policies
def placeholder_policy(seed: int) -> dict:
    cfg = SimConfig()
    sizes = layer_sizes(cfg)
    params = init_params(sizes, seed)
    return {
        "sizes": sizes,
        "in_scale": default_in_scale(cfg).tolist(),
        "activation": "tanh",
        "output": "tanh",
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in unpack(params, sizes)],
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
    }


def adapt_inputs(d: dict, cfg: SimConfig) -> dict:
    """Map a policy trained on a legacy observation layout onto the current one by
    feature key (new inputs get zero weights and scale 1; rescaled inputs get their
    factor folded into in_scale). No-op if sizes match."""
    n_in, want = d["sizes"][0], obs_dim(cfg)
    if n_in == want:
        return d
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


def load_policy(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".npz":
        d = export_json(path)
    else:
        d = json.loads(path.read_text())
        d = d.get("policy", d)  # accept the {name, policy} wrapper written by this module
    sizes = d.get("sizes") or [len(d["layers"][0]["W"]), *[len(layer["b"]) for layer in d["layers"]]]
    n_signal = int((d.get("cfg") or {}).get("n_signal", SimConfig.n_signal))
    cfg = SimConfig(n_signal=n_signal)
    if n_signal != SimConfig.n_signal:
        raise ValueError(f"{path}: n_signal={n_signal}; the viewer runs with n_signal={SimConfig.n_signal}")
    d["sizes"] = sizes
    if sizes[-1] != act_dim(cfg):
        raise ValueError(f"{path}: {sizes[-1]} outputs, the sim expects act_dim {act_dim(cfg)}")
    try:
        d = adapt_inputs(d, cfg)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    return d


def round_policy(d: dict, digits: int = 5) -> dict:
    """Shrink the JSON: weights rounded to `digits` decimals."""
    r = lambda a: np.round(np.asarray(a, dtype=float), digits).tolist()  # noqa: E731
    return {**d, "in_scale": r(d["in_scale"]), "layers": [{"W": r(l["W"]), "b": r(l["b"])} for l in d["layers"]]}


def parse_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        p = Path(spec)
        return (p.parent.name or p.stem), spec
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def build_types(specs: list[str]) -> list[dict]:
    if len(specs) > MAX_TYPES:
        raise SystemExit(f"at most {MAX_TYPES} policies (got {len(specs)})")
    if not specs:
        return [
            {"name": "Placeholder A", "source": "random weights (seed 0)", "placeholder": True, "policy": placeholder_policy(0)},
            {"name": "Placeholder B", "source": "random weights (seed 1)", "placeholder": True, "policy": placeholder_policy(1)},
        ]
    types = []
    for spec in specs:
        name, path = parse_spec(spec)
        pol = load_policy(path)
        types.append({"name": name, "source": str(path), "placeholder": False, "policy": pol,
                      "physics": physics_of(pol.get("cfg") or {})})
    return types


def policies_js(types: list[dict]) -> str:
    return (
        "// Generated by `python -m smartcrossing.export_viewer`; agent types the viewer starts with.\n"
        "window.SC_POLICIES = " + json.dumps(types, separators=(",", ":")) + ";\n"
    )


def inline_html(index_html: str, scripts: dict[str, str]) -> str:
    """Replace <script src="name"></script> tags by inline scripts."""

    def repl(m):
        src = m.group(1)
        if src not in scripts:
            raise ValueError(f"index.html references unknown script {src!r}")
        body = scripts[src].replace("</script", "<\\/script")
        return f"<script>\n{body}\n</script>"

    out = re.sub(r'<script src="([^"]+)"></script>', repl, index_html)
    if re.search(r'\ssrc="https?:|\shref="https?:', out):
        raise ValueError("bundle would make external requests")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", action="append", default=[], metavar="NAME=PATH",
                    help="agent type: display name and .npz/.json policy (repeat up to 3 times)")
    ap.add_argument("--out", default=str(VIEWER / "dist" / "smartcrossing.html"))
    ap.add_argument("--policies-js", default=str(VIEWER / "policies.js"),
                    help="where to write the policy list for the unbundled page ('' to skip)")
    args = ap.parse_args(argv)

    types = build_types(args.policy)
    for t in types:
        t["policy"] = round_policy(t["policy"], 6)
    pjs = policies_js(types)
    if args.policies_js:
        Path(args.policies_js).write_text(pjs)
    scripts = {
        "sim.js": (VIEWER / "sim.js").read_text(),
        "policies.js": pjs,
        "app.js": (VIEWER / "app.js").read_text(),
    }
    html = inline_html((VIEWER / "index.html").read_text(), scripts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    names = ", ".join(t["name"] for t in types)
    print(f"wrote {out} ({len(html) / 1024:.0f} KiB) with agent types: {names}")
    if args.policies_js:
        print(f"wrote {args.policies_js}")


if __name__ == "__main__":
    main()
