"""Build the static website into site/public/ (commit the result; Cloudflare Pages serves that folder as is,
see docs/DEPLOY.md).

    poetry run python site/build.py                       # snapshot results/site/ + training runs in runs/
    poetry run python site/build.py --main runs/es5/final_g310.npz --bench

What it does (modules of site/sitebuild/):
1. context: where checkpoints and logs come from (below), passed explicitly to every phase,
2. discover: picks up the policies of runs/*/ (.npz) and / or the snapshot, follows the warm-start chains,
3. collect: the best checkpoint per method (ES: runs/es*, PPO: runs/ppo*, ...), the main / history / previous
   brains, a "Zeitreise" timeline along the warm-start chain of the best run, the other runs as side notes;
   curves: the training logs (runs/*/log.jsonl) as compact learning curves; texts: names and notes of the runs
   (German + English),
4. bench: benchmarks the reference drivers and the chosen policies in the Python sim (throughput vs demand,
   protocol.SITE_DEMANDS x REPORT_SEEDS). Results are cached in site/data/bench_cache.json, keyed by policy
   weights, physics and the sim source, so rebuilding is fast and reproducible; plus the side-by-side replays,
5. fundamental: the stop gate of the main / previous brain (measured) and the fundamental diagram + cloud
   (from results/deadlock_fundamental.json / fundamental_cloud.json),
6. pages: copies the hand-written pages and scripts from site/src/ (incl. the simulator js/sim.js), writes
   js/sim-config.js (SimConfig defaults and sim constants, from sim.js_constants) and everything the pages need
   into public/data/site-data.js and replays.js (no fetch needed).

runs/ is gitignored and usually holds only the latest runs, so by default (--source auto) the build
merges: the committed snapshot plus the training runs found in runs/ (a run in runs/ replaces the
snapshot's run of the same name). Training runs are runs/es*, runs/ppo*, runs/wide* and runs/r<N>_*
(training rounds, PPO); prof_*, smoke_* and other folders are ignored. Warm starts are followed via
args.json "init" (a run's state.npz), "init_policy" (a transplanted checkpoint, matched by weights) and
"init_bc" (the MLP teacher a brain was behaviour-cloned from, matched by weights). Brains are MLPs
(policy "layers") or set encoders (smartcrossing/policy_set.py: policy "arch" + flat "params").
A build from runs/ (or a merge) also writes a compact snapshot of everything it used
to results/site/ (committed): the chosen checkpoints (brains/<run>/<name>.npz, exact weights,
so the bench cache keys stay valid), the warm-start chain info and the thinned learning curves
(manifest.json). Without runs/ (a fresh clone) the build reads that snapshot instead
(--source), so `poetry run python site/build.py` works from a checkout. The committed
site/public/ is what gets deployed.
"""
