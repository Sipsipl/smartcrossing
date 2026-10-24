"""Learning curves from the training logs (runs/*/log.jsonl or the snapshot) and the "Zeitreise" points along
a warm-start chain."""

import json
import re

import numpy as np

from .context import Context
from .discover import chain_of


def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def _pick(flat: dict, patterns: list[str]):
    for pat in patterns:
        for k, v in flat.items():
            if re.fullmatch(pat, k) and isinstance(v, (int, float)):
                return float(v)
    return None


def read_curve(ctx: Context, run: str, offset: int) -> list[dict]:
    """Learning curve of one run: x (cumulative generation / iteration), exit, crash,
    offroad (fractions of finished cars) and exit at demand 0.15 where evaluated.
    Robust to the field names: ES logs use centre.exit_ok etc."""
    if not ctx.from_runs(run):  # snapshot: stored with offset 0 (already thinned)
        rows = ctx.manifest["curves"].get(run, [])
        return [{**r, "x": round(r["x"] + offset, 4)} for r in rows]
    p = ctx.runs / run / "log.jsonl"
    if not p.exists():
        return []
    rows = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        f = dict(_flat(rec))
        x = _pick(f, [r"gen", r"iter(ation)?", r"update", r"step", r"epoch"])
        x = (x if x is not None else i) + offset
        row = {
            "x": x,
            "exit": _pick(f, [r"centre\.exit_ok", r"(train\.)?exit_ok(_rate)?", r".*\.exit_ok(_rate)?"]),
            "crash": _pick(f, [r"centre\.crash_rate", r"(train\.)?crash(_rate)?", r".*\.crash(_rate)?"]),
            "offroad": _pick(f, [r"centre\.offroad_rate", r"(train\.)?offroad(_rate)?", r".*\.offroad(_rate)?"]),
            "eval15": _pick(f, [r"eval\.d0\.15\.exit_ok", r"eval\..*0\.15.*exit_ok(_rate)?"]),
            "demand": _pick(f, [r"ceiling", r"demand"]),
            "steps": _pick(f, [r"total_steps"]),  # PPO: car decisions so far, cumulative along the warm-start chain
        }
        if row["exit"] is None and row["eval15"] is None:
            continue
        if _pick(f, [r"train\.finished"]) == 0 and row["eval15"] is None:  # no car finished: no rates
            continue
        rows.append({k: (int(v) if k == "steps" else round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if v is not None})
    # thin out long logs (keep every eval point)
    if len(rows) > 400:
        step = len(rows) / 400
        keep = {int(i * step) for i in range(400)} | {i for i, r in enumerate(rows) if "eval15" in r}
        rows = [r for i, r in enumerate(rows) if i in keep]
    return rows


def timeline(ctx: Context, best_id: str, n_max: int, keep: set = frozenset()) -> tuple[list[dict], list]:
    """Zeitreise points along the warm-start chain of `best_id` (at most n_max; checkpoints in
    `keep`, e.g. the previous main brain, always stay) and the chain."""
    found = ctx.found
    c = found[best_id]
    chain = chain_of(ctx, c["run"])
    pts = []
    for run, off in chain:
        for cid, cc in found.items():
            if cc["run"] == run and cc["gen"] is not None:
                pts.append((off + cc["gen"], cid))
    # dedupe equal cumulative generations (current == final), prefer the chosen one
    by_gen = {}
    for g, cid in sorted(pts):
        if g not in by_gen or cid == best_id:
            by_gen[g] = cid
    pts = sorted(by_gen.items())
    best_gen = next((g for g, cid in pts if cid == best_id), None)
    if best_gen is not None:  # nothing after the chosen checkpoint
        pts = [p for p in pts if p[0] <= best_gen]
    # drop near-duplicates (closer than 3% of the whole span to the next point)
    span = max((g for g, _ in pts), default=0) or 1
    kept = []
    for p in pts:  # keep a point only if it is far enough from the previously kept one
        if not kept or p[0] - kept[-1][0] > 0.03 * span:
            kept.append(p)
        elif p == pts[-1] or p[1] == best_id or p[1] in keep:  # the final / chosen / kept checkpoint stays
            kept[-1] = p
    pts = kept
    if len(pts) > n_max:
        idx = set(np.unique(np.linspace(0, len(pts) - 1, n_max).round().astype(int)).tolist())
        for j in (j for j, p in enumerate(pts) if p[1] in keep and j not in idx):
            # replace the nearest evenly spaced point (never the first / last one) by the kept one
            near = min((i for i in idx if 0 < i < len(pts) - 1 and pts[i][1] not in keep), key=lambda i: abs(i - j), default=None)
            if near is not None:
                idx = (idx - {near}) | {j}
        pts = [pts[i] for i in sorted(idx)]
    return [{"gen": g, "id": cid} for g, cid in pts], chain
