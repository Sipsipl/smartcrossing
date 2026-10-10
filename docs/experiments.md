# Experiments

Each entry: hypothesis, config, result, observations. Runs live in `runs/<name>/`
(`log.jsonl`, `args.json`, `best.npz`, `current.npz`). Training: OpenAI-ES
(`smartcrossing/es.py`), shared MLP 47-32-32-4 tanh (2724 params), 4 worker processes,
each evaluating the whole population (+ unperturbed centre) on its own 2 traffic
streams (common random numbers inside a worker), n_slots=16.

## Speed (CPU, 1 core per worker)
- `Sim.observe` is ~75-80% of step time (N x N neighbour argsort + 5x20 ray samples).
- n_slots=32: ~4.6-5.4k env-steps/s per core; n_slots=16: ~10-12k env-steps/s per core.
- ES generation, pop 48+1, 4 workers x 2 streams, 45 s episodes (450 steps): ~5.5-6 s.

## Baselines (cli bench, demand 0.15/arm/s, 8 envs x 120 s, sim at commit 5adcdbd)
- reservation: throughput 23.0/min, exit_ok 0.69, crash+offroad 310/1000 (62 crash, 103 offroad)
- light: throughput 14.9/min, exit_ok 0.51, 461/1000 (210 crash)
- Note: the baselines themselves are not yet crash-free in this sim version.

## es1 — default reward weights
- Hypothesis: default team reward (exit +1, crash/offroad -2, progress 0.01/m) with
  curriculum from demand 0.03 is enough to learn driving to the target.
- Config: pop 48, sigma 0.05, Adam lr 0.02, wd 0.005, 45 s episodes, signal on.
- Result (killed at gen 32): collapsed into **standing still**. After ~12 gens centre
  policy has 0 finished cars, fitness ~ +0.8 (bit of progress). Offroad costs -2 while
  standing costs nothing (max_age 60 s > episode 45 s, so no timeout penalty ever fires).
- Lesson: stopping must be penalised within the episode, progress must outweigh risk.

## es2 — stronger progress shaping, timeout inside the episode
- Hypothesis: with r_progress 0.03/m, r_offroad -1, r_timeout -1.5 and max_age 25 s
  (< 40 s episode) standing still is worse than driving, so cars learn to drive to the target.
- Config: `--seconds 40 --cfg {"r_progress":0.03,"r_offroad":-1.0,"r_timeout":-1.5,"max_age":25}`,
  pop 48, sigma 0.05, lr 0.02, curriculum from 0.03 (never raised: gate exit_ok > 0.85 not reached).
- Result (stopped at gen 106, ~7 s/gen): centre exit_ok rose 0 -> ~0.6 at demand 0.03; eval
  at 0.05: exit_ok 0.59-0.64, offroad 0.36-0.41; at 0.15: exit_ok 0.22, offroad 0.63, crash 0.12.
- Per-manoeuvre diagnosis (demand 0.05): **right turns 0% success (all offroad)**, straight 71%,
  left 61%. Cars drive at ~v_max (delay ~0.4 s) and never learned to brake for the tight right turn.
- Sheet: cars mostly travel straight along the arm at varying lateral offsets, no visible side preference.

## es3 — warm start from es2, bigger sigma, higher starting demand
- Hypothesis: more exploration (sigma 0.08, lr 0.03) and more cars per generation
  (demand 0.06, 3 streams/worker, 35 s episodes, max_age 20) break the right-turn plateau.
- Result (stopped at gen ~65, 7-14 s/gen with CPU contention from other jobs): eval at 0.05
  stuck at exit_ok 0.68 / offroad 0.32 (= right turns still failing), at 0.15 exit_ok 0.33,
  crash 0.11, offroad 0.56. Plateau, not a sample-size problem.
- Diagnosis: (a) the sim's euclidean gate-distance progress pulls right/left turners
  straight into the curb corner block; (b) own position is fed as x/L (L = 50 m), so the
  whole intersection box spans only +-0.19 in input space.
