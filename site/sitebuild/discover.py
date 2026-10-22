"""Checkpoints: discovery in runs/ and / or the committed snapshot, the Brain helper (MLP or set encoder), the
warm-start chains (args.json init / init_policy / init_bc) and writing the snapshot results/site/."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from smartcrossing.policy_compat import INTERFACE_KEYS, load_adapted_policy, physics_of

from .context import Context, method_of


def world_of(physics: dict) -> str:
    """'narrow' (original 7 m road) or 'wide' (14 m road) from a checkpoint's physics."""
    return "narrow" if physics.get("W", 3.5) < 5 else "wide"


def ckpt_meta(path: Path) -> dict:
    with np.load(path) as z:
        extra = json.loads(str(z["extra"])) if "extra" in z.files else {}
        cfg = json.loads(str(z["cfg"])) if "cfg" in z.files else {}
    gen = None
    for k in ("gen", "iter", "update", "step", "epoch"):
        if isinstance(extra.get(k), (int, float)):
            gen = int(extra[k])
            break
    if gen is None:  # e.g. final_g310.npz
        m = re.search(r"(?:g|gen|it|iter|u)(\d+)", path.stem)
        gen = int(m.group(1)) if m else None
    return {"gen": gen, "cfg": cfg, "extra": extra}


class Brain:
    """One brain of the site: a policy dict in the export_json / load_adapted_policy format, either an MLP
    (``layers``: W, b per layer) or a set encoder (smartcrossing/policy_set.py: ``arch`` + flat ``params``).
    All set / MLP branching of the build lives here."""

    def __init__(self, d: dict):
        self.d = d

    @property
    def is_set(self) -> bool:
        return bool(self.d.get("arch")) and self.d["arch"].get("type") == "set"

    def vector(self) -> np.ndarray:
        """The flat parameter vector (policy.py layout)."""
        if self.is_set:
            return np.asarray(self.d["params"], dtype=float)
        return np.concatenate([np.concatenate([np.ravel(np.asarray(l["W"], dtype=float)), np.asarray(l["b"], dtype=float)])
                               for l in self.d["layers"]])

    def hash(self) -> str:
        return hashlib.sha1(self.vector().tobytes() + json.dumps(self.d["in_scale"]).encode()).hexdigest()[:16]

    def payload(self) -> tuple:
        """Bench / replay payload for policy_driver: (params, sizes, in_scale[, arch])."""
        return (self.vector().tolist(), self.d["sizes"], self.d["in_scale"], *([self.d["arch"]] if self.is_set else []))

    def rounded(self, digits: int = 5) -> dict:
        """Shrink the JSON: weights rounded to `digits` decimals."""
        r = lambda a: np.round(np.asarray(a, dtype=float), digits).tolist()  # noqa: E731
        d = self.d
        if self.is_set:
            return {**d, "in_scale": r(d["in_scale"]), "params": r(d["params"])}
        return {**d, "in_scale": r(d["in_scale"]), "layers": [{"W": r(l["W"]), "b": r(l["b"])} for l in d["layers"]]}

    def snapshot(self) -> tuple[dict, dict]:
        """For results/site/brains/*.npz: (info fields n_layers + policy without weights,
        arrays in_scale + params (set encoder) or W<i> / b<i> (MLP))."""
        d = self.d
        info = {"n_layers": 0 if self.is_set else len(d["layers"]),
                "policy": {k: v for k, v in d.items() if k not in ("layers", "in_scale", "params")}}
        arrays = {"in_scale": np.asarray(d["in_scale"], dtype=float)}
        if self.is_set:  # set encoder: flat params, the arch stays in info["policy"]
            arrays["params"] = np.asarray(d["params"], dtype=float)
        for i, l in enumerate(d.get("layers", [])):
            arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(l["W"], dtype=float), np.asarray(l["b"], dtype=float)
        return info, arrays

    @classmethod
    def from_snapshot(cls, z, info: dict) -> "Brain":
        """Inverse of snapshot (z: the opened npz)."""
        pol = {**info["policy"], "in_scale": z["in_scale"].tolist()}
        if "params" in z.files:  # set encoder: flat params (its arch is in info["policy"])
            pol["params"] = z["params"].tolist()
        else:
            pol["layers"] = [{"W": z[f"W{i}"].tolist(), "b": z[f"b{i}"].tolist()} for i in range(info["n_layers"])]
        return cls(pol)


def policy_hash(pol: dict) -> str:
    return Brain(pol).hash()


def _entry(run: str, stem: str, file: str, method: str, gen, physics: dict, pol: dict) -> dict:
    return {"id": f"{run}/{stem}", "run": run, "file": file, "method": method, "gen": gen,
            "physics": physics, "policy": pol, "adapted": pol.get("adapted"), "hash": policy_hash(pol)}


def discover(ctx: Context, world: str = "any") -> dict:
    """All loadable checkpoints of the training runs in runs/ (or of the snapshot): id -> info (only those
    trained on `world` = narrow / wide crossing, unless 'any')."""
    found = {}
    if ctx.kind in ("snapshot", "merge"):
        for f in sorted((ctx.snapshot / "brains").glob("*/*.npz")):
            with np.load(f) as z:
                info = json.loads(str(z["info"]))
                pol = Brain.from_snapshot(z, info).d
            # snapshots written before the interface keys existed: lamp count / neighbour layout from the
            # checkpoint's own training cfg (old brains: 2 lamps, acceleration feature)
            phys = {**info["physics"], **{k: v for k, v in physics_of(pol.get("cfg") or {}).items() if k in INTERFACE_KEYS}}
            if world != "any" and world_of(phys) != world:
                continue
            if ctx.from_runs(info["run"]):
                continue
            e = _entry(info["run"], f.stem, info["file"], info["method"], info["gen"], phys, pol)
            found[e["id"]] = {**e, "snap": True}
        if ctx.kind == "snapshot":
            return found
    if not ctx.runs.exists():
        return found
    for run_dir in sorted(p for p in ctx.runs.iterdir() if p.is_dir()):
        method = method_of(run_dir.name)
        if not method or not ctx.from_runs(run_dir.name):
            continue
        for f in sorted(run_dir.glob("*.npz")):
            try:
                pol = load_adapted_policy(f)
            except Exception as e:  # incompatible / broken checkpoint: skip loudly
                print(f"  skip {f.relative_to(ctx.root)}: {e}")
                continue
            meta = ckpt_meta(f)
            if world != "any" and world_of(physics_of(meta["cfg"])) != world:
                continue
            e = _entry(run_dir.name, f.stem, f.relative_to(ctx.root).as_posix(), method, meta["gen"], physics_of(meta["cfg"]), pol)
            found[e["id"]] = e
    return found


def init_of(ctx: Context, run: str) -> tuple[str, int] | None:
    """(run, generation) of the checkpoint `run` was warm-started from, if it is a known run."""
    if not ctx.from_runs(run):
        r = ctx.manifest["runs"].get(run, {}).get("init")
        return tuple(r) if r else None
    p = ctx.runs / run / "args.json"
    args = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    init = args.get("init") or ""
    # actor transplanted from a checkpoint (init_policy) or behaviour-cloned from a teacher (init_bc): find it by its weights
    src = args.get("init_policy") or args.get("init_bc")
    if not init and src:
        ip = ctx.root / src
        if not ip.is_file():
            return None
        h = policy_hash(load_adapted_policy(ip))
        # a run whose first checkpoint is an exact copy of the same brain is not its origin: prefer gen > 0
        cands = [c for c in ctx.found.values() if c["hash"] == h and c["run"] != run]
        c = min(cands, key=lambda c: not c["gen"], default=None)
        return (c["run"], c["gen"] or 0) if c else None
    if not init:
        return None
    ip = ctx.root / init
    if not ip.exists() or not method_of(ip.parent.name):
        return None
    return ip.parent.name, ckpt_meta(ip)["gen"] or 0


def chain_of(ctx: Context, run: str) -> list[tuple[str, int]]:
    """Warm-start chain ending in `run`: [(run, gen offset), ...] oldest first."""
    chain, gens, seen = [], {}, set()
    cur = run
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        nxt = init_of(ctx, cur)
        if not nxt:
            break
        gens[cur] = nxt[1]
        cur = nxt[0]
    chain.reverse()
    out, offset = [], 0
    for i, r in enumerate(chain):
        if i > 0:
            offset = dict(out)[chain[i - 1]] + gens[r]
        out.append((r, offset))
    return out


def resolve(ctx: Context, spec: str) -> str:
    """Checkpoint id of `spec`: an id (<run>/<name>), its runs/ path, or any .npz / .json file
    whose weights equal a discovered checkpoint (e.g. results/wide_s1ft2_ckpt2650.npz)."""
    found = ctx.found
    cid = next((k for k, c in found.items() if c["file"] == spec or k == spec), None)
    if cid is None and (ctx.root / spec).is_file():
        h = policy_hash(load_adapted_policy(ctx.root / spec))
        cid = next((k for k, c in found.items() if c["hash"] == h), None)
    if cid is None:
        raise SystemExit(f"{spec}: not a loadable checkpoint in {ctx.kind} (see skip messages)")
    return cid


def run_world(ctx: Context, run: str) -> str | None:
    c = next((c for c in ctx.found.values() if c["run"] == run), None)
    if c is not None:
        return world_of(c["physics"])
    if ctx.manifest:
        return ctx.manifest["runs"].get(run, {}).get("world")
    return None


def write_snapshot(ctx: Context, used: list[str], chain_runs: list[str]):
    """results/site/: the used checkpoints (exact weights) + manifest (chains, worlds, thinned learning curves)."""
    from .curves import read_curve  # (curves imports this module)

    found = ctx.found
    brains = ctx.snapshot / "brains"
    old = {f.relative_to(brains).as_posix() for f in brains.glob("*/*.npz")} if brains.exists() else set()
    keep = set()
    for cid in used:
        c = found[cid]
        pol = c["policy"]
        rel = f"{c['run']}/{cid.split('/', 1)[1]}.npz"
        keep.add(rel)
        if c.get("snap") and (brains / rel).exists():  # unchanged copy from the snapshot itself
            continue
        binfo, arrays = Brain(pol).snapshot()
        info = {"run": c["run"], "file": c["file"], "method": c["method"], "gen": c["gen"], "physics": c["physics"],
                **binfo}
        path = brains / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, info=json.dumps(info, sort_keys=True), **arrays)
    for rel in old - keep:
        if ctx.kind == "merge":
            continue  # merge: the snapshot only grows (nothing old is deleted)
        (brains / rel).unlink()
    for d in brains.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    runs = sorted(set(chain_runs) | {found[cid]["run"] for cid in used})
    manifest = {
        "note": "written by site/build.py from runs/ (see its docstring); merged with runs/ or read alone",
        "runs": {r: {"init": list(init_of(ctx, r)) if init_of(ctx, r) else None, "world": run_world(ctx, r)} for r in runs},
        "curves": {r: read_curve(ctx, r, 0) for r in sorted(set(chain_runs))},
    }
    if ctx.kind == "merge":  # keep what the snapshot knew about runs not in runs/
        for k in ("runs", "curves"):
            manifest[k] = {**{r: v for r, v in ctx.manifest[k].items() if not ctx.from_runs(r)}, **manifest[k]}
    (ctx.snapshot / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
                                                encoding="utf-8", newline="\n")
    size = sum(f.stat().st_size for f in ctx.snapshot.rglob("*") if f.is_file())
    print(f"  snapshot: {len(keep)} brains + manifest in {ctx.snapshot.relative_to(ctx.root)}/ ({size / 1024:.0f} KiB)")
