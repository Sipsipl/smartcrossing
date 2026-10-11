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
- Keep the simulator vectorised over (envs, slots); no per-car Python loops in `sim.py` hot paths.
  The cloud container is CPU only (4 cores); the owner's PC has an RTX 3070 Ti (GPU port = Roadmap C).
- Training budget: <= ~1 h per round during development; long runs only before a merge to main.
- Workflow: work on a branch, never push to main; merge via PR only after the owner's OK
  (CI runs pytest; Cloudflare deploys main to https://smartcrossing.pages.dev, other branches get previews).
- `runs/` and `models/` are gitignored; commit small final artifacts under `results/` instead.
- The learning agents must never get lane/rule information; baselines may.
- `site/` is the only frontend. `site/src/js/sim.js` mirrors `sim.py`; its constants come from the generated
  `js/sim-config.js` (`sim.js_constants()`), parity is checked by `tests/test_sim_js_parity.py`.
