"""Command line of the site build (entry point: site/build.py; options: --help)."""

import argparse

from obs_labels import obs_layout
from smartcrossing.sim import SimConfig

from . import __doc__ as PACKAGE_DOC
from .collect import bench_all, collect
from .context import Context
from .discover import discover
from .fundamental import gate_of
from .pages import write_site

# the site's main brain (wide crossing) and the end of the older line whose training history is
# shown before it (narrow crossing); override with --main / --history
# Checkpoints are named <run>/<file stem> (found in runs/ or in the snapshot); a path to an
# .npz anywhere (e.g. results/wide_s1ft2_ckpt2650.npz) works too if its weights match one.
DEFAULT_MAIN = ["r7_DOB_s1/ckpt_1050"]  # = results/round7_best.npz (round 7: the round 6 brain trained on with road blockades, picked by stop-gate selection; not the run's last checkpoint)
DEFAULT_PREVIOUS = "r6_DO_s2/ckpt_0750"  # = results/round6_best.npz, the main brain before round 7
DEFAULT_HISTORY = "ppo17/ckpt_5250"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="site/build.py", description=PACKAGE_DOC,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", action="append", default=None, metavar="PATH",
                    help=f"the site's main brain (first one; drives start page, sensors, brain, Vergleich) and/or the best "
                         f"of its method (runs/<run>/x.npz); repeatable; default {DEFAULT_MAIN}")
    ap.add_argument("--history", default=DEFAULT_HISTORY, metavar="PATH",
                    help="checkpoint whose warm-start chain is shown before the main chain in the learning curve and "
                         "Zeitreise (older crossing / earlier line); '' = none")
    ap.add_argument("--bench", action="store_true", help="ignore the bench cache and re-run all benchmarks")
    ap.add_argument("--timeline", type=int, default=6, help="max checkpoints per Zeitreise")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--world", choices=["narrow", "wide", "any"], default="any",
                    help="only use checkpoints trained on this crossing (narrow = original 7 m road, wide = 14 m road)")
    ap.add_argument("--previous", default=DEFAULT_PREVIOUS, metavar="PATH",
                    help="the previous main brain, shown next to the main brain on Vergleich ('' = none)")
    ap.add_argument("--source", choices=["auto", "runs", "snapshot", "merge"], default="auto",
                    help="where the checkpoints / logs come from: runs/ only, the committed snapshot results/site/ only, "
                         "or merge (snapshot + training runs in runs/); runs and merge also refresh the snapshot; "
                         "auto = merge if both exist, else whichever exists")
    a = ap.parse_args(argv)
    if a.main is None:
        a.main = list(DEFAULT_MAIN)
    return a


def main(argv=None):
    a = parse_args(argv)
    # the run notes with {safe} / {stop} / {block} are about the default main brain's checkpoint
    ctx = Context(note_ckpt=DEFAULT_MAIN[0])
    ctx.setup(a.source)
    cfg = SimConfig()
    obs_layout(cfg)  # fails loudly if the sim's observation layout changed
    print("discovering checkpoints ...")
    ctx.found.update(discover(ctx, a.world))
    print(f"  {len(ctx.found)} loadable checkpoints")
    sel = collect(ctx, a)
    if gate_of(ctx, sel["main_id"]) is None:
        print("WARNING: no stop gate for the main brain (checkpoint not found); /vergleich shows no gate rates")
    bres = bench_all(ctx, sel, cfg, a)
    write_site(ctx, sel, bres, cfg, a)
