"""Command line entry points.

  python -m smartcrossing.cli bench   --driver reservation|light|none|policy:<file>
  python -m smartcrossing.cli render  --driver ... --out runs/x   (sheet.png, episode.gif, spacetime.png)
  python -m smartcrossing.cli gate    --driver policy:<file> [--out cases.json]   (stop gate, gates.py)
"""

import argparse
import json
import time
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
    p.add_argument("cmd", choices=["bench", "render", "gate"])
    p.add_argument("--driver", default="reservation")
    p.add_argument("--demand", type=float, default=0.15)
    p.add_argument("--cfg", default="", help="JSON overrides for SimConfig")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default="runs/render", help="render: folder; gate: optional JSON file of the cases")
    p.add_argument("--view", type=float, default=None, help="render: half-width of the view in m")
    a = p.parse_args(argv)
    cfg = cfg_from_args(a)
    if a.cmd == "gate":
        from .gates import stop_gate

        summary, rows = stop_gate(a.driver, json.loads(a.cfg) if a.cfg else None)
        if a.out != p.get_default("out"):
            Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8", newline="\n")
        print(json.dumps(summary))
    elif a.cmd == "bench":
        t0 = time.time()
        sim, _ = run(a.driver, cfg, a.envs, a.seconds, a.seed)
        res = {"driver": a.driver, "demand": cfg.demand, **summarize(sim, a.seconds)}
        res["wall_s"] = round(time.time() - t0, 1)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}))
    else:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        sim, frames = run(a.driver, cfg, 1, a.seconds, a.seed, frame_every=2, view=a.view,
                          spacetime=out / "spacetime.png")
        pick = np.linspace(len(frames) * 0.1, len(frames) - 1, 9).astype(int)
        contact_sheet([frames[i] for i in pick]).save(out / "sheet.png")
        save_gif(frames[::2], out / "episode.gif", fps=10)
        print(json.dumps(summarize(sim, a.seconds)))


if __name__ == "__main__":
    main()
