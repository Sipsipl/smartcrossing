# CLAUDE.md

Read `docs/PLAN.md` first (goals, fixed decisions, architecture). Log experiments in `docs/experiments.md`.

## Commands
```bash
poetry install
poetry run pytest -q
poetry run python -m smartcrossing.cli bench --driver reservation --envs 8 --demand 0.15
poetry run python -m smartcrossing.cli render --driver light --out runs/light   # sheet.png + episode.gif
```

## Notes
- CPU only (4 cores, no GPU). Keep the simulator vectorised over (envs, slots); no per-car Python loops in `sim.py` hot paths.
- `runs/` and `models/` are gitignored; commit small final artifacts under `results/` instead.
- The learning agents must never get lane/rule information; baselines may.
- `site/` is the only frontend. `site/src/js/sim.js` mirrors `sim.py`; its constants come from the generated
  `js/sim-config.js` (`sim.js_constants()`), parity is checked by `tests/test_sim_js_parity.py`.
