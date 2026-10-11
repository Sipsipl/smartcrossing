"""One table over the training runs in runs/: name, commit, date, key settings, the last log line's
training metrics and the last eval line.

Settings are shown as differences: ppo.py arguments that differ from ppo.py's defaults, and
SimConfig fields that differ from SimConfig's defaults (``cfg.``). Directories without an
args.json or log.jsonl (analysis folders, renders, ...) and plain files are skipped
(``--all`` lists them with empty columns).

  poetry run python scripts/runs_index.py                  # all runs, newest first
  poetry run python scripts/runs_index.py --filter r1_ --json
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smartcrossing.ppo import parse_args  # noqa: E402
from smartcrossing.sim import SimConfig  # noqa: E402

# bookkeeping arguments that are not settings
SKIP_ARGS = {"name", "actor_sizes", "critic_sizes", "cfg", "git", "eval_every", "render_every"}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _last_json_line(path: Path):
    """Last parseable JSON line of a .jsonl file (None if missing or empty)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, str) and ("/" in v or "\\" in v):
        return Path(v).name  # checkpoint paths: the file name is enough
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def settings(args: dict) -> str:
    """ppo.py arguments and SimConfig fields that differ from their defaults."""
    base = vars(parse_args(["--name", "x"]))
    out = [f"{k}={_fmt(v)}" for k, v in args.items() if k not in SKIP_ARGS and k in base and base[k] != v]
    cfg, dflt = args.get("cfg") or {}, SimConfig().__dict__
    for k, v in cfg.items():
        d = dflt.get(k)
        d = list(d) if isinstance(d, tuple) else d
        if k in dflt and d != v and not (isinstance(v, float) and isinstance(d, float) and abs(v - d) < 1e-12):
            out.append(f"cfg.{k}={_fmt(v)}")
    return " ".join(out)


def run_row(d: Path) -> dict | None:
    args = _read_json(d / "args.json")
    log = _last_json_line(d / "log.jsonl")
    if args is None and log is None:
        return None
    args = args or {}
    git = args.get("git") or {}
    commit = (git.get("commit") or "")[:8] + ("*" if git.get("dirty") else "")
    stamp = max((p.stat().st_mtime for p in (d / "log.jsonl", d / "args.json") if p.exists()), default=0)
    row = {"name": d.name, "commit": commit or "-", "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(stamp)),
           "settings": settings(args) if args else "(no args.json)", "log": "", "eval": ""}
    if log:
        tr = log.get("train", {})
        row["log"] = (f"it={log.get('iter')} ceil={log.get('ceiling')} exit={tr.get('exit_ok')} crash={tr.get('crash')} "
                      f"off={tr.get('offroad')} rew={log.get('rew_per_car_step')}")
    ev = _last_json_line(d / "eval.jsonl")
    if ev:
        parts = [f"it={ev.get('iter')}"]
        for k, m in ev.items():
            if k.startswith("d") and isinstance(m, dict):
                parts.append(f"{k}: thr={m.get('throughput_per_min')} ok={m.get('exit_ok_rate')} "
                             f"crash/1k={m.get('crash_cars_per_1000')}")
        row["eval"] = " ".join(parts)
    row["_stamp"] = stamp
    return row


def index(runs: Path, flt: str = "", all_dirs: bool = False) -> list[dict]:
    rows = []
    for d in sorted(runs.iterdir()) if runs.is_dir() else []:
        if not d.is_dir() or flt not in d.name:
            continue
        try:
            r = run_row(d)
        except Exception as e:  # a broken run folder must not break the index
            r = {"name": d.name, "commit": "-", "date": "", "settings": f"(unreadable: {e})", "log": "", "eval": "",
                 "_stamp": 0}
        if r is None and all_dirs:
            r = {"name": d.name, "commit": "-", "date": "", "settings": "(no run)", "log": "", "eval": "", "_stamp": 0}
        if r is not None:
            rows.append(r)
    rows.sort(key=lambda r: -r["_stamp"])
    for r in rows:
        del r["_stamp"]
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--filter", default="", help="only runs whose name contains this")
    p.add_argument("--all", action="store_true", help="also list folders that are not runs")
    p.add_argument("--json", action="store_true", help="one JSON object per run instead of the table")
    a = p.parse_args(argv)
    rows = index(Path(a.runs), a.filter, a.all)
    if a.json:
        for r in rows:
            print(json.dumps(r))
        return
    cols = ("name", "commit", "date", "settings", "log", "eval")
    w = {c: max([len(c)] + [len(r[c]) for r in rows]) for c in cols[:3]}
    header = {**dict(zip(cols, cols)), "log": "", "eval": ""}
    for r in [header] + rows:
        head = "  ".join(r[c].ljust(w[c]) for c in cols[:3])
        print(f"{head}  {r['settings']}")
        if r["log"] or r["eval"]:
            print(f"{'':{len(head)}}  log: {r['log']}" + (f" | eval: {r['eval']}" if r["eval"] else ""))


if __name__ == "__main__":
    main()
