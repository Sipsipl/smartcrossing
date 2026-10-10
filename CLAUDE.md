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
- `legacy/` is the old pygame/DQN implementation, not used.
