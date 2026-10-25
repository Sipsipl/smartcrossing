"""Command line entry points (driver specs: reservation | light | none | policy:<file.npz>, drivers.py):

  python -m smartcrossing.cli bench    --driver ... [--demand 0.15 --envs 8 --seconds 120 --seed 1000]   (one JSON line)
  python -m smartcrossing.cli render   --driver ... --out runs/x   (sheet.png, episode.gif, spacetime.png)
  python -m smartcrossing.cli gate     --driver policy:<file> [--out cases.json]   (stop gate, gates.py)
  python -m smartcrossing.cli evalset  --driver ... [--driver ...] --out results/x_eval.jsonl --workers 4   (evalset.py)
  python -m smartcrossing.cli select   --run runs/x [--run ...] [--every N] [--gate-min 0.95] [--ref results/round7_best.npz]
                                       [--copy-to results/x.npz] --out results/x_select.jsonl --workers 4   (select.py)
  python -m smartcrossing.cli deadlock --driver ... --demands 0.3,0.5 --out runs/deadlock/x.jsonl
                                       [--noise policy|0.1] [--lamp zero|shuffle]   (deadlock.py)
  python -m smartcrossing.cli deadlock-summarize runs/deadlock/*.jsonl [--fundamental results/deadlock_fundamental.json
                                       --name round7_best=brain] [--json rows.json]   (table + fundamental diagram)
  python -m smartcrossing.cli cloud    --driver brain=policy:results/round7_best.npz --driver light --driver reservation
                                       --out results/fundamental_cloud.json --workers 4   (scatter cloud, fundamental_cloud.py)
  python -m smartcrossing.cli incentives [--driver policy:<file>] [--cfg JSON]   (reward accounting of a stuck car, incentives.py)

Defaults of --demands / --seeds / --envs / --seconds per command: smartcrossing/protocol.py.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import protocol
from .drivers import make_driver
from .metrics import summarize
from .render import SpaceTime, contact_sheet, draw_frame, save_gif
from .sim import Sim, SimConfig

def run(spec, cfg, n_envs, seconds, seed, frame_every=None, b_render=0, view=None, spacetime=None):
    sim = Sim(cfg, n_envs, seed=seed)
    drv = make_driver(spec, sim)
    frames = []
    st = SpaceTime(sim, b_render) if spacetime else None
    steps = int(seconds / cfg.dt)
    for i in range(steps):
        sim.step(drv.act())
        if st is not None:
            st.record()
        if frame_every and i % frame_every == 0:
            frames.append(draw_frame(sim, b_render, 400, view=view, label=spec.split("/")[-1]))
    if spacetime:
        st.image().save(spacetime)
    return sim, frames


def cfg_from_args(a) -> SimConfig:
    cfg = SimConfig(demand=a.demand)
    if a.cfg:
        cfg = SimConfig(**{**cfg.__dict__, **json.loads(a.cfg)})
    return cfg


def _csv(s: str, default, cast=float):
    """Comma-separated values of an option ('' = default)."""
    return [cast(x) for x in s.split(",")] if s else default


def _names(pairs) -> dict:
    """--name spec=key pairs -> {spec: key}."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--name {p}: expected spec=key, e.g. round7_best=brain")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=list(COMMANDS))
    p.add_argument("files", nargs="*", help="deadlock-summarize: sweep JSONL files (cli deadlock --out)")
    p.add_argument("--driver", action="append", default=None,
                   help="driver spec (default reservation); evalset / deadlock: repeatable; cloud: key=spec, repeatable")
    p.add_argument("--demand", type=float, default=0.15, help="bench / render: arrivals /s/arm")
    p.add_argument("--cfg", default="", help="JSON overrides for SimConfig")
    p.add_argument("--envs", type=int, default=None, help="envs per run (default: protocol.ENVS = 8)")
    p.add_argument("--seconds", type=float, default=None,
                   help="seconds per run (default: protocol.SECONDS = 120, for every command; the deadlock sweep "
                        "protocol of the fundamental diagram passes --seconds 180 = protocol.SWEEP_SECONDS)")
    p.add_argument("--seed", type=int, default=1000, help="bench / render: traffic seed")
    p.add_argument("--out", default=None, help="render: folder (default runs/render); gate: optional JSON file of "
                   "the cases; evalset / select / deadlock: JSONL file the lines are appended to; cloud: JSON file")
    p.add_argument("--view", type=float, default=None, help="render: half-width of the view in m")
    p.add_argument("--demands", default="", help="evalset / select / deadlock / cloud: comma-separated demands "
                   "(default: protocol.EVAL_DEMANDS / SELECT_DEMANDS / SWEEP_DEMANDS)")
    p.add_argument("--seeds", default="", help="evalset / select / deadlock: comma-separated traffic seeds "
                   "(default: protocol.SELECTION_SEEDS)")
    p.add_argument("--workers", type=int, default=1, help="evalset / select / deadlock / cloud: worker processes")
    p.add_argument("--noise", default=None, help="deadlock: action noise, 'policy' (trained std) or a fixed std")
    p.add_argument("--lamp", default="normal", choices=["normal", "zero", "shuffle"], help="deadlock: received lamps")
    p.add_argument("--run", action="append", default=None, help="select: run dir with ckpt_*.npz (repeatable)")
    p.add_argument("--every", type=int, default=1, help="select: every N-th checkpoint (first and last always)")
    p.add_argument("--gate-min", type=float, default=0.95, help="select: min safe_rate_feasible of the stop gate")
    p.add_argument("--crash-tol", type=float, default=1.1, help="select: crash+off/1000 at 0.3 <= ref * this + 1")
    p.add_argument("--ref", default=None, help="select: reference checkpoint (default: the run's first one)")
    p.add_argument("--copy-to", default=None, help="select: copy the overall selected checkpoint here")
    p.add_argument("--name", action="append", default=None, metavar="SPEC=KEY",
                   help="deadlock-summarize: driver key in the fundamental JSON, e.g. round7_best=brain (repeatable; "
                        "unnamed drivers keep their label)")
    p.add_argument("--fundamental", default=None, help="deadlock-summarize: write the fundamental-diagram JSON here")
    p.add_argument("--json", default=None, help="deadlock-summarize: write the aggregated rows here")
    return p


def cmd_select(a, o):
    from . import select

    select.select(a.run, a.every, a.gate_min, a.crash_tol, ref=a.ref, workers=a.workers, out=a.out,
                  copy_to=a.copy_to, cfg_d=o.cfg_d, envs=o.envs, seconds=o.seconds,
                  demands=_csv(a.demands, protocol.SELECT_DEMANDS), seeds=_csv(a.seeds, protocol.SELECTION_SEEDS, int))


def cmd_cloud(a, o):
    from .deadlock import label
    from .fundamental_cloud import cloud

    specs = dict(d.split("=", 1) if "=" in d else (label(d), d) for d in o.drivers)
    res = cloud(specs, _csv(a.demands, protocol.SWEEP_DEMANDS), workers=a.workers, cfg_d=o.cfg_d)
    Path(a.out).write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8", newline="\n")


def cmd_deadlock(a, o):
    from . import deadlock

    deadlock.sweep(o.drivers, _csv(a.demands, protocol.SWEEP_DEMANDS), _csv(a.seeds, protocol.SELECTION_SEEDS, int),
                   o.envs, o.seconds, a.noise, a.lamp, o.cfg_d, a.out, a.workers)


def cmd_deadlock_summarize(a, o):
    from . import deadlock

    if not a.files:
        raise SystemExit("deadlock-summarize: give the sweep JSONL files")
    deadlock.summarize_files(a.files, _names(a.name), a.fundamental, a.json)


def cmd_incentives(a, o):
    from .incentives import incentives

    train = {}
    if a.driver and (o.driver.startswith("policy:") or o.driver.endswith(".npz")):
        from .policy import load_npz

        train = load_npz(o.driver.removeprefix("policy:"))[2]
    print(json.dumps(incentives({**train, **(o.cfg_d or {})}), indent=1))


def cmd_evalset(a, o):
    from . import evalset

    evalset.evalset(o.drivers, a.out, o.cfg_d, demands=_csv(a.demands, protocol.EVAL_DEMANDS),
                    seeds=_csv(a.seeds, protocol.SELECTION_SEEDS, int), envs=o.envs, seconds=o.seconds,
                    workers=a.workers)


def cmd_gate(a, o):
    from .gates import stop_gate

    summary, rows = stop_gate(o.driver, o.cfg_d)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(summary))


def cmd_bench(a, o):
    from .evalset import bench_line

    print(json.dumps(bench_line(o.driver, cfg_from_args(a), a.seed, o.envs, o.seconds)))


def cmd_render(a, o):
    out = Path(a.out or "runs/render")
    out.mkdir(parents=True, exist_ok=True)
    sim, frames = run(o.driver, cfg_from_args(a), 1, o.seconds, a.seed, frame_every=2, view=a.view,
                      spacetime=out / "spacetime.png")
    pick = np.linspace(len(frames) * 0.1, len(frames) - 1, 9).astype(int)
    contact_sheet([frames[i] for i in pick]).save(out / "sheet.png")
    save_gif(frames[::2], out / "episode.gif", fps=10)
    print(json.dumps(summarize(sim, o.seconds)))


COMMANDS = {"bench": cmd_bench, "render": cmd_render, "gate": cmd_gate, "evalset": cmd_evalset, "select": cmd_select,
            "deadlock": cmd_deadlock, "deadlock-summarize": cmd_deadlock_summarize, "cloud": cmd_cloud,
            "incentives": cmd_incentives}


def main(argv=None):
    a = parser().parse_intermixed_args(argv)
    if a.files and a.cmd != "deadlock-summarize":
        raise SystemExit(f"{a.cmd}: unexpected arguments {a.files}")
    drivers = a.driver or ["reservation"]
    o = SimpleNamespace(drivers=drivers, driver=drivers[-1], cfg_d=json.loads(a.cfg) if a.cfg else None,
                        envs=protocol.ENVS if a.envs is None else a.envs,
                        seconds=protocol.SECONDS if a.seconds is None else a.seconds)
    COMMANDS[a.cmd](a, o)


if __name__ == "__main__":
    main()
