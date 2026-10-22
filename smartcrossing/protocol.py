"""Evaluation protocol constants: seeds, bench size and demand lists, defined once for evalset.py, select.py,
deadlock.py and site/build.py. Changing a value here changes measurements (and the site's bench cache keys).

Seeds by purpose:
- SELECTION_SEEDS: used while training rounds pick checkpoints (cli evalset / select, ppo evals).
- REPORT_SEEDS: held out from any selection; everything on the website (site bench, fundamental diagram,
  scatter cloud: traffic seeds REPORT_SEEDS[0] + k) uses them, so selection bias does not flatter the chosen
  brain.
"""

SELECTION_SEEDS = (1000, 2000, 3000)
REPORT_SEEDS = (4000, 5000, 6000)

# one bench run: envs x seconds in the default world (cli bench, evalset, select, site bench)
ENVS, SECONDS = 8, 120.0

EVAL_DEMANDS = (0.05, 0.15, 0.3, 0.5)  # cli evalset (the standard eval)
SELECT_DEMANDS = (0.15, 0.3, 0.5)  # cli select (needs 0.3 for the crash limit and 0.5 for the score)
SWEEP_DEMANDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)  # cli deadlock (fundamental diagram) and cli cloud
SWEEP_SECONDS = 180.0  # cli deadlock: longer runs than a bench (deadlocks need time to form and end)
SITE_DEMANDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)  # site bench, up to the start page slider maximum (120 cars/min)
