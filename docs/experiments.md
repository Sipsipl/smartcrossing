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

## Sim update (commit fd6f9ae, by the sim agent)
spawn_headway=1.5 s (spawning considers closing speed), queued cars keep their arrival
attributes (better CRN), separate RNG for signal noise. Runs es1-es4 used the older sim;
numbers before/after are not directly comparable. Reference throughput (exits/min) at
demand 0.05/0.1/0.15/0.2/0.3: light 11.1/17.3/19.5/20.3/21.1, reservation
12.0/22.6/32.2/40.4/52.5 (both 0 crashes). (The baseline numbers above at 5adcdbd are obsolete.)

## es4 — fresh start with input scaling + route-aware shaping
- Changes: policy input scaling (`policy.default_in_scale`: own position x5, goal x2,
  neighbour positions x3, stored in the npz/json); sim progress off (`r_progress 0`) and
  replaced by es-side `--r-geo 0.03` = progress on a route-aware distance (target arm:
  remaining length; box: distance to target mouth centre; other arms: via their mouth).
- Result (stopped at gen 58): collapsed to **blocking the entrance**: cars stop near the
  spawn point, time out (-1.5) but keep the queue from entering, so far fewer cars can
  fail. 0 exits. Team reward without a cost for queued demand has this exploit.
- Fix: es-side `--r-queue` penalty per queued car per second.

## es5 — warm start from es3 + input scaling + route-aware shaping + queue penalty (new sim)
- Hypothesis: with the entrance-blocking exploit priced in (`--r-queue 0.1` per queued
  car-second) and the corner-cutting pull removed (`--r-geo 0.03`, sim `r_progress 0`),
  ES gets past the right-turn plateau. Warm start from es3 (first-layer weights divided by
  the new input scale so the network function is unchanged; gen-0 exit_ok 0.61 = es3).
- Config: `--init runs/es3/current.npz --seconds 35 --scen 3 --sigma 0.08 --lr 0.03
  --demand-start 0.05 --r-geo 0.03 --r-queue 0.1 --cfg {"r_progress":0,"r_offroad":-1,
  "r_timeout":-1.5,"max_age":20}`, signal on, pop 48, ~6.5-7 s/gen. Stopped at gen ~310
  (time budget); curriculum ceiling reached only 0.078.
- Per-manoeuvre (demand 0.05, 16 envs x 60 s): right 0.86 (was 0.00 in es2), straight 0.88,
  left 0.86 exit_ok; residual losses mostly offroad, some crashes.
- Bench (cli, 8 envs x 120 s), `runs/es5/final_g310.npz`:
  - demand 0.05: throughput 9.9/min, exit_ok 0.80, crash+offroad 201/1000 (16 crash, 24 offroad)
  - demand 0.15: throughput 17.6/min, exit_ok 0.52, 478/1000 (67 crash, 191 offroad)
  - reference (new sim): light 11.1 / 19.5, reservation 12.0 / 32.2 per min, 0 crashes.
  - `best.npz` (gen 160, best fixed-seed eval score) is slightly worse: 9.3/min @0.05, 15.6/min @0.15.
- Observations (`runs/es5/render_final_d015/sheet.png`): cars always drive at ~v_max
  (delay ~0.5 s, never brake or yield), turners cut diagonally through the box from the
  entry mouth to the exit mouth. At 0.15 the dominant failure is offroad, presumably cars
  swerving around each other at full speed. Roof lamps look manoeuvre-dependent (left
  turners mostly magenta) - unverified, needs MI(signal; manoeuvre) analysis.
- Driving side: none yet. Mean right-offset (relative to travel direction) on approach
  +0.11 m (50% on right), when leaving +0.64 m (57% on right) - a weak right bias on exit,
  well within what turn geometry alone explains.
- Open: no braking/yielding learned (no gradient yet - crashes are rare at the low
  curriculum demand); low-demand offroad ~10%; ES progress slow (~0.1 exit_ok per 100 gens).

## PPO (smartcrossing/ppo.py) — runs ppo1-ppo8 (reconstructed from runs/*/args.json + logs)
Pure-numpy PPO with parameter sharing (every car = one sample of the shared policy, own
per-car reward, alpha 0 = egoist), actor 64-64 tanh (exported in the policy.py format),
critic with 4 privileged inputs, GAE per car, curriculum: env demand ~ U(demand_lo *
ceiling, ceiling), ceiling x1.15 when training crash+offroad < safe_bad (0.03) and exit_ok
> 0.9. Eval = deterministic policy, cli-bench settings (8 envs x 120 s, seed 1000) at
demand 0.05 / 0.15 every 50 iters (`eval.jsonl`). All with signal on, gamma 0.995 (ppo1:
0.99), r_time 0.1, r_crash = r_offroad = -2, r_exit +1.
- ppo1a / ppo1b (n_slots 24, 4x24 envs, lr 3e-4 / 1e-3, demand_start 0.08): cars learned to
  stand / crawl (timeouts, queue grows); eval @0.05 exit_ok 0.25 / 0.19. Killed at ~100-150 it.
- ppo2 / ppo3 (2x24 envs, demand 0.03; ppo3 r_crash = r_offroad -3, r_time 0.2): **wrong exits**
  dominate (~55% of finished cars take the cheapest exit, r_wrong_exit -1 < cost of turning).
  @0.05 exit_ok 0.32 / 0.30. Killed at ~130 it.
- ppo4 (n_slots 16, 4x64 envs): same wrong-exit attractor; killed at 37 it.
- ppo5 (r_wrong_exit -2): wrong exits gone, replaced by offroad (59%); @0.05 exit_ok 0.29.
- ppo6 (+ r_progress 0.02 route-aware): breakthrough, @0.05 exit_ok 0.70 at it 100
  (crash 52 cars), @0.15 13.3/min. Killed at 120 it to restart with a smaller policy std.
- ppo7 = ppo6 config from scratch with logstd_init -1.6, lr_std 1e-2; ppo8 = identical but
  warm-started from ppo6 it 120 (`--init runs/ppo6/state.npz --reset-logstd`). Both stayed
  at the curriculum floor (ceiling 0.05-0.058, training demand U(0.015, 0.05)): training
  crash+offroad ~5% never got below the 3% gate.
  - ppo7 best eval it 650: @0.05 11.1/min, exit_ok 0.90, 101/1000; @0.15 20.1/min, 0.60, 397/1000.
  - ppo8 best eval it 700: @0.05 11.25/min, exit_ok 0.91, 91/1000; @0.15 22.3/min, 0.66, 342/1000
    (beats light's 19.5/min on throughput, but with 1/3 of the cars crashing). Evals fluctuate
    +-0.05 exit_ok between checkpoints, plateau from ~it 450.
  - Render (`runs/ppo8/render_0700`): every car drives at ~v_max all the way (spacetime lines
    straight, never a stop); turners cut diagonally; **no braking / yielding at all**.
    Crashes = two cars arriving at the same conflict point.
  - Diagnosis: at training demand ~0.03 a car meets another one in the box rarely (a few
    crash events per 15k-sample batch), so the gradient for yielding is tiny and noisy;
    the 3% safety gate keeps the curriculum at that density forever. Killed ppo7 (it 723)
    and ppo8 (it 842, snapshot `runs/ppo8/state_0842.npz`).
