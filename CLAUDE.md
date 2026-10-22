# CLAUDE.md

Read `docs/PLAN.md` first (goals, fixed decisions, architecture). Log experiments in `docs/experiments.md`.

## Commands
```bash
poetry install
poetry install --with gpu   # optional: torch 2.6.0+cu118 (sim_torch, ppo --update-device cuda); CI uses the CPU wheel
poetry run pytest -q               # cuda test cases are skipped unless SC_TEST_CUDA=1 (never while a training runs)
poetry run python -m smartcrossing.cli bench --driver reservation --envs 8 --demand 0.15
poetry run python -m smartcrossing.cli render --driver light --out runs/light   # sheet.png + episode.gif
poetry run python -m smartcrossing.cli evalset --driver policy:results/round4_best.npz --driver reservation --out results/x_eval.jsonl --workers 4   # standard eval: 4 demands x 3 seeds + stop gate
poetry run python -m smartcrossing.cli select --run runs/<name> --every 2 --workers 4 --out results/x_select.jsonl   # model selection: checkpoints with stop gate >= 0.95, best throughput at 0.5 under a crash limit
poetry run python -m smartcrossing.cli deadlock --driver policy:results/round5_best.npz --driver light --driver reservation --seeds 4000,5000,6000 --seconds 180 --workers 4 --out runs/deadlock/x.jsonl   # demand sweep + deadlock detection
poetry run python -m smartcrossing.cli deadlock-summarize runs/deadlock/x.jsonl --name round5_best=brain --fundamental results/deadlock_fundamental.json   # table + fundamental diagram of the site
poetry run python -m smartcrossing.cli cloud --driver brain=policy:results/round5_best.npz --driver light --driver reservation --workers 4 --out results/fundamental_cloud.json   # scatter cloud of the site
poetry run python site/build.py    # website -> site/public/ (entry point; code in site/sitebuild/, see its docstring)
```

## Notes
- Keep the simulator vectorised over (envs, slots); no per-car Python loops in `sim.py` hot paths.
  The cloud container is CPU only (4 cores); the owner's PC has an RTX 3070 Ti (GPU port = Roadmap C).
- Local PC (Windows 11): Ryzen 5 3600 (6 cores / 12 threads), 32 GB RAM, RTX 3070 Ti (8 GB VRAM, driver 610).
  Python 3.13.2 via pyenv-win, Poetry 2.1, Node 25. Multiprocessing must work with `spawn` (no `fork` on
  Windows); text I/O always with `encoding="utf-8"`; `.gitattributes` forces LF because `site/build.py`
  hashes source bytes (bench cache keys, asset versions).
- Training budget: <= ~1 h per round during development; long runs only before a merge to main.
- Workflow: work on a branch, never push to main; merge via PR only after the owner's OK
  (CI runs pytest; Cloudflare deploys main to https://smartcrossing.pages.dev, other branches get previews).
- `runs/` and `models/` are gitignored; commit small final artifacts under `results/` instead.
- The learning agents must never get lane/rule information; baselines may.
- Seeds by purpose: SELECTION seeds 1000/2000/3000 (`select.py`, `evalset.py`, training rounds); REPORTING / held-out seeds 4000/5000/6000 for everything on the website (site bench, fundamental diagram, cloud). Both live in `smartcrossing/protocol.py` with the
  other protocol constants (bench envs / seconds, demand lists); change them only deliberately (bench cache keys, comparability).
- `site/` is the only frontend. `site/src/js/sim.js` mirrors `sim.py`; its constants come from the generated
  `js/sim-config.js` (`sim.js_constants()`), parity is checked by `tests/test_sim_js_parity.py`.
