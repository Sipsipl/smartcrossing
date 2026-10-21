"""Command line entry points.

  python -m smartcrossing.cli bench   --driver reservation|light|none|policy:<file>
  python -m smartcrossing.cli render  --driver ... --out runs/x   (sheet.png, episode.gif, spacetime.png)
  python -m smartcrossing.cli gate    --driver policy:<file> [--out cases.json]   (stop gate, gates.py)
  python -m smartcrossing.cli deadlock --driver ... --demands 0.3,0.5 --out runs/deadlock/x.jsonl
                                       [--noise policy|0.1] [--lamp zero|shuffle]   (deadlock.py)
  python -m smartcrossing.cli cloud --driver brain=policy:results/round4_best.npz --driver light --driver reservation
                                       --out results/fundamental_cloud.json --workers 4   (scatter cloud, deadlock.py)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import LaneDriver
from .metrics import summarize
from .render import SpaceTime, contact_sheet, draw_frame, save_gif
from .sim import Sim, SimConfig


def make_driver(spec: str, sim: Sim):
    """light / reservation / none -> rule-based LaneDriver; policy:<file> or a plain .npz path -> PolicyDriver."""
    if spec in ("light", "reservation", "none"):
        return LaneDriver(sim, spec)
    from .policy import PolicyDriver

    return PolicyDriver.load(spec.removeprefix("policy:"), sim)


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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["bench", "render", "gate", "evalset", "deadlock", "cloud"])
    p.add_argument("--driver", action="append", default=None,
                   help="driver spec (default reservation); evalset: repeatable, one block of lines per driver")
    p.add_argument("--demand", type=float, default=0.15)
    p.add_argument("--cfg", default="", help="JSON overrides for SimConfig")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default=None, help="render: folder (default runs/render); gate: optional JSON file of "
                   "the cases; evalset: JSONL file the lines are appended to")
    p.add_argument("--view", type=float, default=None, help="render: half-width of the view in m")
    p.add_argument("--demands", default="", help="evalset: comma-separated demands (default evalset.DEMANDS)")
    p.add_argument("--seeds", default="", help="evalset: comma-separated traffic seeds (default evalset.SEEDS)")
    p.add_argument("--workers", type=int, default=1, help="evalset / deadlock: worker processes")
    p.add_argument("--noise", default=None, help="deadlock: action noise, 'policy' (trained std) or a fixed std")
    p.add_argument("--lamp", default="normal", choices=["normal", "zero", "shuffle"], help="deadlock: received lamps")
    a = p.parse_args(argv)
    drivers = a.driver or ["reservation"]
    driver = drivers[-1]
    cfg = cfg_from_args(a)
    if a.cmd == "cloud":
        from . import deadlock

        specs = dict(d.split("=", 1) if "=" in d else (deadlock.label(d), d) for d in drivers)
        res = deadlock.cloud(specs, [float(x) for x in a.demands.split(",")] if a.demands else deadlock.CLOUD_DEMANDS,
                             workers=a.workers, cfg_d=json.loads(a.cfg) if a.cfg else None)
        Path(a.out).write_text(json.dumps(res, separators=(",", ":")), encoding="utf-8", newline="\n")
    elif a.cmd == "deadlock":
        from . import deadlock

        deadlock.sweep(drivers, [float(x) for x in a.demands.split(",")] if a.demands else deadlock.DEMANDS,
                       [int(x) for x in a.seeds.split(",")] if a.seeds else deadlock.SEEDS, a.envs, a.seconds,
                       a.noise, a.lamp, json.loads(a.cfg) if a.cfg else None, a.out, a.workers)
    elif a.cmd == "evalset":
        from . import evalset

        evalset.evalset(drivers, a.out, json.loads(a.cfg) if a.cfg else None,
                        demands=[float(x) for x in a.demands.split(",")] if a.demands else evalset.DEMANDS,
                        seeds=[int(x) for x in a.seeds.split(",")] if a.seeds else evalset.SEEDS,
                        envs=a.envs, seconds=a.seconds, workers=a.workers)
    elif a.cmd == "gate":
        from .gates import stop_gate

        summary, rows = stop_gate(driver, json.loads(a.cfg) if a.cfg else None)
        if a.out:
            Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")
        print(json.dumps(summary))
    elif a.cmd == "bench":
        from .evalset import bench_line

        print(json.dumps(bench_line(driver, cfg, a.seed, a.envs, a.seconds)))
    else:
        out = Path(a.out or "runs/render")
        out.mkdir(parents=True, exist_ok=True)
        sim, frames = run(driver, cfg, 1, a.seconds, a.seed, frame_every=2, view=a.view,
                          spacetime=out / "spacetime.png")
        pick = np.linspace(len(frames) * 0.1, len(frames) - 1, 9).astype(int)
        contact_sheet([frames[i] for i in pick]).save(out / "sheet.png")
        save_gif(frames[::2], out / "episode.gif", fps=10)
        print(json.dumps(summarize(sim, a.seconds)))


if __name__ == "__main__":
    main()
