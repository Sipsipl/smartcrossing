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

## ppo9 / ppo10 / ppo11 — denser training traffic (curriculum fix)
- Change (all three): warm start from `runs/ppo8/state_0842.npz`, new flag `--set-ceiling 0.15`,
  `--demand-lo 0.15` (env demand ~ U(0.0225, 0.15) instead of U(0.015, 0.05)), `--safe-bad 0.05`.
  Hypothesis: more car-car encounters per batch give the yielding gradient that was missing.
- ppo10 = ppo9 + r_crash = r_offroad = -4. Worse than ppo9 after 100 it (@0.05 exit_ok 0.90 vs
  0.95); killed early to free cores (inconclusive, short).
- ppo11 = ppo9 (it 950) + **perception range 60 m** (new SimConfig field `neighbour_vis`,
  default 30 = old behaviour; positions still scaled by NEIGHBOUR_RANGE=30; PolicyDriver.load
  applies the value a checkpoint was trained with; mirrored in viewer/sim.js and exported as a
  per-policy "physics" key with legacy value 30). Rationale: at 30 m two cars on perpendicular
  arms first see each other ~1.5 s before the box. Result: the input shift cost ~300 it of
  re-learning; at it 1300 still behind ppo9 (@0.15 24.8/min, exit_ok 0.74, 261/1000; delay 1.3 s
  - it brakes more). Killed at it 1343 for core budget; not conclusive.
- **ppo9** (r_crash -2): steady improvement, best evals
  - it 1350: @0.05 11.9/min, exit_ok 0.969, 31/1000; @0.15 26.6/min, 0.79, 212/1000
  - it 1700: @0.05 11.8/min, 0.959, 41/1000; @0.15 **27.7/min**, 0.83, 170/1000 (light 19.5)
  - plateau ~it 1500-1764 (training crash ~9% at the 0.15 ceiling, gate 5% never reached).
  Stopped at it 1764 (`runs/ppo9/state_final.npz`).
- Render (`runs/ppo9/render_1000b`): still almost no braking (delay ~0.5-0.8 s), crashes at
  the box entry. Policy means are not tanh-saturated (accel mu median 0.7), so it is not an
  exploration-collapse problem.
- `smartcrossing/analyze.py` (new): signal zero/shuffle interventions, MI(lamp; manoeuvre /
  braking / region), lamp correlations, driving side. ppo9 it 900 @0.15: zeroing received
  lamps does *not* hurt (exit_ok 0.73 vs 0.70 normal, shuffle 0.68) -> lamps not used yet;
  MI(lamp; manoeuvre) 0.29 bit (lamps do encode the manoeuvre, H(lamp) 2.6 bit); driving side:
  **left**-hand bias on all four arms (approach 26% right of centre, mean -0.52 m; leaving 19%
  right, -0.68 m).

## ppo12 / ppo13 — fine-tuning ppo9 (warm start `runs/ppo9/state_final.npz`, 1 worker x 64 envs)
- ppo12: r_crash = r_offroad = -5 (lr 1e-3). No gain: @0.15 0.76-0.79 exit_ok, 210-240/1000,
  slightly more braking (delay ~1 s). Stopped at it 2295.
- **ppo13**: lr 3e-4 / critic 1e-3, rewards unchanged. Slow steady gain, plateau from ~it 2700;
  training crash dipped < 5% so the ceiling rose to 0.198. Stopped at it 4402.
  - ckpt_2850: @0.05 11.9/min, exit_ok 0.970, 30/1000; @0.15 **29.3/min**, 0.875, 125/1000
  - ckpt_3200: @0.05 12.0/min, exit_ok 0.980, 20/1000; @0.15 27.9/min, 0.834, 166/1000
  - late evals (it 3700-4400): @0.05 0.94-0.95, @0.15 0.84-0.85 / 28.5/min.
- Crash anatomy (ppo13 it 3950, @0.15, 16 envs x 120 s, scratchpad script): 86% of crash pairs
  are in the box, both cars at 12-14 m/s (nobody brakes); 42% perpendicular cars from adjacent
  arms, 23% same-heading pairs from adjacent arms (turner merging into a straight car), 20%
  opposite-heading. Crashes happen ~4 s after spawn and the partner had been within the 30 m
  perception range for only **1.4 s (median; 85% < 2 s)** - less than the 2 s a car at 14 m/s
  needs to stop. With 30 m sensors the policy would have to approach slowly to be safe.

## Signal vs no signal, seed 0 (from scratch, the ppo9 curriculum, 1 worker x 64 envs, 2000 it)
- `sig_s0` vs `nosig_s0` (`--no-signal`: lamp outputs out of the likelihood, received lamps 0).
  Same seed, same budget. Eval trajectories are indistinguishable (both reach @0.05 ~0.95-0.98,
  @0.15 ~0.75-0.81 exit_ok by it 1500-2000; nosig marginally better on most evals).
- `analyze.py` on ckpt_1999 (@0.15, 16 envs x 120 s, seed 1000):
  - sig_s0: 26.3/min, exit_ok 0.81, 191/1000. Received lamps zeroed: 26.0/min, 0.805, 195/1000;
    shuffled across cars: 26.5/min, 0.816, 184/1000 -> **the policy ignores incoming lamps**.
  - Own lamp is informative though: H(lamp, 3x3 bins) 1.8 bit, MI(lamp; manoeuvre) 0.15 bit
    (0.09 bit on the approach), MI(lamp; region approach/box/leave) 0.15 bit, MI(lamp; braking)
    0.01 bit; lamp2 correlates with speed (+0.50) and distance to centre (-0.31); left turners
    show lamp1 -0.26 vs right +0.03, straight +0.13. A side effect of the shared hidden layer,
    not a used channel.
  - nosig_s0: 27.3/min, exit_ok 0.85, 153/1000.
- Driving side (offset to the right of travel, 10-40 m out on the arms):
  - sig_s0: approaching -0.01 m (50% right; right-turners -1.0, left-turners +0.64 = they swing
    wide), leaving -0.69 m (20% right; straight -0.68, left -1.03, right -0.34).
  - nosig_s0: approaching -0.65 m (14% right), leaving -1.02 m (3% right) on all four arms,
    all manoeuvres (straight -0.55 / -1.10).
  - Both seed-0 runs (and ppo9) drive **on the left**; straight cars, which have no turn-geometry
    reason, show it as well, so this is a convention. Needs seed 1 for the across-seed question.

## ppo14 / ppo15 — 60 m perception from scratch; ppo16 — target-speed action
- ppo14 (neighbour_vis 60, neighbour inputs still x3): stuck in the offroad/right-turn plateau
  (exit_ok 0.3 at it 550; far neighbours give inputs up to 6 and saturate the first layer).
  Killed. Fix: `default_in_scale` now scales neighbour positions by 3 * 30 / neighbour_vis.
- ppo15 (vis 60, scaled inputs, seed 0, otherwise = sig_s0): tracks sig_s0 almost exactly at
  equal iterations (it 850: @0.15 0.77 vs 0.78); at it 1800-1900 @0.05 0.95-0.96, @0.15
  0.80-0.83 / 27-28/min - at most marginally better than the 30 m runs. Crash anatomy with 60 m:
  partner is among the 4 observed neighbours in >85% of the steps 1.5 s before the crash and
  visible since spawn, yet both still hit at 12-14 m/s -> **not a perception problem**, the
  policy has not learned to yield.
- ppo16: new SimConfig `speed_control` (action[0] = target speed (a+1)/2 * v_max, acc =
  clip((v_t - v) / speed_tau 0.5 s, -brake_max, acc_max)); motivation: with the acceleration
  interface every action > 0 is identical at v_max, so PPO gets no gradient towards slowing.
  Mirrored in viewer/sim.js (+ parity case), export physics key, PolicyDriver applies it.
  Result (from scratch, seed 0, 2000 it): more cautious early (delay 1.8-2 s) but converges
  to the same place (@0.05 0.94-0.96, @0.15 0.77-0.79, delay ~1.1 s). Killed at it 1995. No gain.

## Signal vs no signal, seed 1 (same recipe, `--seed 1`, 2000 it)
- Mean of the last 5 evals (it 1800-1999), @0.05 / @0.15 exit_ok (bad per 1000):
  sig_s0 0.963 (37) / 0.762 (238); nosig_s0 0.955 (45) / 0.792 (208);
  sig_s1 0.924 (76) / 0.771 (230); nosig_s1 0.943 (57) / 0.781 (219).
  -> no measurable benefit of the signal channel (if anything no-signal is ~1-3 points better).
- sig_s1 ckpt_1999 @0.15 interventions: normal 25.9/min, 0.802, 198/1000; zeroed 26.3/min, 0.815,
  185/1000; shuffled 26.1/min, 0.809, 191/1000 -> lamps again ignored by receivers.
  Own lamp: H 1.9 bit, MI(lamp; manoeuvre) 0.09 bit, MI(lamp; region) 0.13, MI(lamp; brake) 0.002;
  lamps correlate with own speed (-0.36 / -0.33).
- **Driving side, seed 1: right-hand traffic** in both runs, on all four arms and for all
  manoeuvres: sig_s1 approaching +1.45 m (99% right), leaving +1.16 m (95% right);
  nosig_s1 approaching +0.88 m (90%), leaving +1.00 m (92%).
  Seed 0 (sig and nosig) drives on the left. So a driving-side convention emerges reliably
  (also without any communication channel), and which side is a symmetry breaking that depends
  on the seed - exactly as expected for an arbitrary convention.

## coop_a1 — fully cooperative reward from scratch (alpha 1.0, otherwise = sig_s0)
- Per-car reward = own + 1.0 * mean reward of the other cars on the road (per step).
- Result: **collapse to immediate offroad** (exit_ok 0 from it 0 to 1999; every car leaves the
  road within ~1 s). Reason: the alpha term is paid only while the car is alive, and early on
  the others' rewards are dominated by their -2 offroad/crash events, so living longer means
  absorbing more of the others' penalties; dying quickly is optimal. Per-step reward sharing
  among the *living* is a broken cooperative objective - a proper version needs a credit term
  that does not scale with the own lifetime (e.g. difference rewards, or sharing only terminal
  outcomes of cars that met). Lesson recorded; not pursued from scratch.

## coop_a05 — alpha 0.5 warm-started from sig_s0 it 1999 (1000 it)
- From a policy that already drives, the others' mean reward is positive on average, so no
  collapse. Result (ckpt_2999, @0.15, 16 envs x 120 s): 26.4/min, exit_ok 0.82, 183/1000
  (sig_s0 it 1999: 26.3/min, 0.81, 191/1000) - no measurable difference from the egoist; zeroing
  lamps changes nothing (0.811); still left-hand traffic (approach 27% right, leave 17% right).
  Usable as a "cooperative" type in the viewer mixing, but behaviourally ~identical to egoists.

## ppo15 (end), ppo17 — final main line
- ppo15 (vis 60) finished 4000 it: last evals @0.05 0.92-0.95, @0.15 0.79-0.82 / 27-28/min,
  best it 3250 (@0.15 29.2/min, 0.875, 125/1000) - same level as the 30 m runs.
- **ppo17**: warm start ppo13 it 4402, lam 0.98, lr 1.5e-4, 2 workers x 64 envs (curriculum
  ceiling 0.198 inherited). Slow drift up; stopped at it ~5400 (time budget).
- Robust bench (cli bench, 32 envs x 120 s, seed 1000), throughput/min, exit_ok, crash+offroad/1000:

  | demand | light | reservation | ppo13 ckpt_3200 | ppo13 ckpt_4400 | **ppo17 ckpt_5250** |
  |---|---|---|---|---|---|
  | 0.05 | 9.7 / 1.00 / 0 | 10.3 / 1.00 / 0 | 10.2 / 0.962 / 38 | 10.2 / 0.964 / 36 | 10.2 / 0.962 / 38 |
  | 0.10 | - | - | 19.9 / 0.887 / 113 | 20.1 / 0.894 / 106 | 20.5 / 0.912 / 88 |
  | 0.15 | 19.5 / 1.00 / 0 | 32.3 / 1.00 / 0 | 27.3 / 0.819 / 181 | 27.7 / 0.827 / 173 | 27.8 / 0.833 / 167 |
  | 0.20 | - | - | 33.9 / 0.77 / 230 | 34.4 / 0.777 / 223 | 35.5 / 0.803 / 197 |
  | 0.30 | 20.9 / 1.00 / 0 | 53.4 / 1.00 / 0 | 41.0 / 0.63 / 370 | 44.4 / 0.682 / 318 | 46.0 / 0.71 / 290 |

  The PPO cars beat the traffic light on throughput from demand 0.1 up (0.15: 27.8 vs 19.5/min,
  0.3: 46 vs 21/min) with ~0.5 s mean delay (light 21 s), but at a crash rate that is far from
  acceptable (167/1000 at 0.15, 38/1000 at 0.05; targets were <50 and <20). Reservation (AIM)
  remains better on every metric.
- ppo17 ckpt_5250 analysis (@0.15): zeroed lamps 28.6/min, 0.885 vs normal 28.6/min, 0.883 ->
  lamps unused by receivers; own lamp encodes manoeuvre (MI 0.28 bit, 0.41 bit on the approach),
  region (0.30) and speed (lamp2 corr with speed -0.52, with accel +0.56); left-hand traffic
  (approach 13% right, mean -0.93 m; leave 7% right, -1.19 m) on all arms.

## Summary / open problems (after ppo17)
- What mattered: progress shaping 0.02 (ppo6), and above all **dense training traffic**
  (env demand up to 0.15 from the start; ppo9 / sig_s*: from scratch to @0.05 0.95 in ~1000 it,
  ~20 min on 2 cores). Lower lr for fine-tuning (ppo13/17) gives the last few points.
- What did not matter: crash penalty -4/-5, 60 m perception, target-speed action, alpha 0.5.
- Open: all PPO policies plateau at ~15% crashed cars at demand 0.15; cars essentially never
  brake (brake_frac 1-4%) although the partner is visible and observed. Yielding is a
  coordination problem a shared deterministic policy has not solved with this reward;
  candidates: explicit time-to-collision features, recurrent policy, opponent shaping / a
  joint-crash penalty that is not shared by the victim, larger batches per conflict, or
  separately trained populations (asymmetric roles).
- Signal: no run uses the received lamp (zero/shuffle interventions change nothing, no-signal
  runs are as good). The lamp *output* carries manoeuvre / speed information (MI 0.1-0.4 bit) as
  a side effect; nobody listens.
- Driving side: a consistent convention per seed emerges with or without signal; seed 0 runs
  (sig_s0, nosig_s0, ppo9/13/17 lineage, coop_a05) drive on the left, seed 1 runs (sig_s1,
  nosig_s1) on the right.
