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

## Wide crossing, grip limit, reverse gear, new observation (runs wide_*)
Sim changes (all mirrored in `viewer/sim.js`, parity cases added: narrow geometry, strong grip,
reverse-heavy trajectory, grip/impact/reverse edge cases):
- **Geometry** default W = 7 (14 m roads, two cars side by side per direction), Rc = 9, L = 50.
  The old crossing is `SimConfig(**sim.NARROW)` (W 3.5, Rc 6, a_lat_max 0, v_reverse 0).
  Spawns: uniform over the whole width (both halves), as before.
- **Tyre grip**: actual curvature clipped to min(kappa_max, a_lat_max / v^2) after the steering-
  rate limit, a_lat_max = 4 m/s^2 (~0.4 g). Old physics allowed r = 5 m at 14 m/s (39 m/s^2).
  Turn feasibility: tightest baseline arc (right turn from the right lane, r = 10.75 m) at
  6.6 m/s, left turn (r = 17.75 m) at 8.4 m/s; at 14 m/s r_min = 49 m.
- **Reverse gear** (owner decision): speed in [-v_reverse, v_max], v_reverse = 2 m/s. Pedal < 0
  brakes a forward car (brake_max) and, once stopped, accelerates backwards (acc_max); pedal > 0
  the mirror image; speed never crosses 0 within a step. Heading uses the signed speed. Backing
  out of an arm = wrong exit. Baselines never reverse. (`speed_control` interface: no reverse.)
- **Crash / off-road cost = base + factor x impact speed**: r_crash = r_offroad = -1.5,
  -0.1 per m/s (car-car: |relative velocity| of the pair, max over partners; off-road: own
  speed); each car pays its own term (no blame). Range -1.5 (touch) ... -2.9 (14 m/s) ... -4.3
  (head-on 2 x 14 m/s); order exit +1 > waiting (r_time 0.1/s) > timeout -1 > crash kept.
  No standing-still / entrance-blocking relapse seen (timeouts 0 in all evals; with r_time 0.1/s a
  car standing 60 s pays -7).
- **Observation** (OBS_DIM 17 -> 19, neighbour block 6+2 -> 8+2, obs_dim 48 -> 59): own position
  in fixed 10 m units (was / L then x5: same values for L = 50), exit-gate vector in 25 m units
  (was / L x2: same for L = 50), NEW exit-mouth vector (centre of the target arm's mouth, 25 m
  units), NEW own grip-limited max curvature / kappa_max; per neighbour NEW curvature / kappa_max
  and actual acceleration / brake_max (brake lights; physically visible). Old checkpoints load
  through `export_viewer.adapt_inputs` (feature-name remap, rescaled inputs folded into in_scale,
  new inputs weight 0): ppo17 ckpt_5250 reproduces its bench exactly (28.56/min, 0.883 @0.15).
- `PolicyDriver.load` / viewer physics now also apply the checkpoint's L, W, Rc, a_lat_max,
  v_reverse (old checkpoints: the narrow values), so every policy runs on the crossing it learned.
- New analysis (`analyze.py`): `--breakdown` (solo car = 1 slot per env, and demands; outcome per
  manoeuvre), wobble (straight arm segments: mean |yaw rate|, curvature sign changes / s, heading
  std, |lateral speed|; split approach / leave), lane histograms, reverse usage; baselines too.

### Failure breakdown "before" (ppo17 ckpt_5250 on the narrow crossing, 16 envs x 120 s, % of finished cars)
| condition | n | exit ok | crash | offroad | timeout+wrong | R exit/off | S exit/off | L exit/off | thr/min |
|---|---|---|---|---|---|---|---|---|---|
| solo | 256 | 100.0 | 0.0 | 0.0 | 0.0 | 100.0/0.0 | 100.0/0.0 | 100.0/0.0 | 8.0 |
| d0.02 | 158 | 97.5 | 2.5 | 0.0 | 0.0 | 95.2/0.0 | 97.9/0.0 | 100.0/0.0 | 4.81 |
| d0.05 | 343 | 97.1 | 2.9 | 0.0 | 0.0 | 95.0/0.0 | 96.3/0.0 | 100.0/0.0 | 10.41 |
| d0.15 | 1035 | 88.3 | 11.4 | 0.3 | 0.0 | 85.0/0.3 | 85.2/0.6 | 95.4/0.0 | 28.56 |
| d0.3 | 2024 | 71.3 | 27.7 | 0.9 | 0.0 | 62.2/0.9 | 62.4/0.9 | 90.0/1.1 | 45.12 |

A solo ppo17 car never fails; essentially all failures are car-car crashes (offroad <= 1%).
Light and reservation: 100% exit at every condition (light: 1.1% timeouts at 0.3).

### Baselines on the wide crossing (cli bench, 16 envs x 120 s, seed 1000)
Two lanes per direction, right-hand traffic. Lessons from making them crash-free: with uniform
spawns across both halves, a rule-based car spawned in the oncoming half has to cross the
leaving traffic (gap acceptance / mutual stopping produced head-on crashes or deadlocks in every
variant tried). Final design: baselines set `cfg.spawn_right` (lane-disciplined arrivals; same
arrival times / manoeuvres / random draws, only the lateral offset maps to the right half);
light: right lane for right turns, left lane for left turns, straight = nearest lane, merging
with a heading-target law at <= 9 m/s, per-lane stop-line queues; reservation: every car keeps
the lane nearest its spawn offset (turns from either lane, the manager resolves conflicts),
reservation only with lateral error < 1 m, executes the plan capped by the path speed limit.

| demand | arrivals/min | light thr/min (delay s) | reservation thr/min (delay s) |
|---|---|---|---|
| 0.05 | 11.3 | 10.0 (6.6) | 10.6 (1.6) |
| 0.10 | 23.2 | 18.5 (12.3) | 21.5 (1.8) |
| 0.15 | 34.3 | 23.0 (16.8) | 31.6 (2.0) |
| 0.20 | 45.6 | 24.3 (19.7) | 41.5 (2.2) |
| 0.30 | 68.6 | 25.7 (21.5) | 60.9 (3.0) |
| 0.50 | 116.1 | 26.1 (23.1) | 81.1 (4.4) |

All with 0 crashes, 0 offroad, 0 timeouts, exit_ok 1.00. Narrow crossing @0.15 (regression
check): light 19.6/min, reservation 31.3/min, 0 crashes. The wide road lifts the light's
saturation from ~21 to ~26/min and reservation's from 53 to 61/min @0.3 (81/min @0.5).

### PPO on the wide crossing (wide_s0, wide_s1 -> wide_s0r / wide_s1r -> wide_s1ft)
- Recipe = sig_s0 (dense traffic from the start: ceiling 0.15, demand_lo 0.15, safe_bad 0.05;
  lr 1e-3 / critic 2e-3, logstd_init -1.6, gamma 0.995, r_progress 0.02 route-aware, r_time 0.1,
  r_wrong_exit -2), n_slots 24, 1 worker x 64 envs, signal on, from scratch (obs layout changed).
- wide_s0 / wide_s1 (seed 0 / 1, no reverse gear yet) ran to it ~700; the reverse gear was
  added then and both were **warm-started** from their state.npz with v_reverse = 2 (same obs
  layout, physics change only) as wide_s0r / wide_s1r. wide_s1r was switched at it 1067 to the
  fine-tuning phase (lr 3e-4 / critic 1e-3, 2 workers x 64 envs) = **wide_s1ft**; wide_s0r kept
  lr 1e-3. Stopped at it 1636 (s1ft) / 1936 (s0r) for the time budget (~1.3 h of training).
- Learning speed: from scratch to @0.05 exit_ok 0.85 in ~300 it (sig_s0 on the narrow crossing
  was at ~0.9 there); @0.15 plateau ~0.70-0.75 at lr 1e-3 (s0r until it 1900), fine-tuning
  lifted s1 to 0.81-0.83 within 300 it (still rising slowly when stopped).
- Ablation **wide_noman_s0** (`--zero-inputs 5,6,7`: manoeuvre one-hot hidden from actor and
  critic, otherwise = wide_s0, no reverse, killed at it 330): at equal iterations it trails wide_s0
  by 5-10 points (it 300: @0.05 0.78 vs 0.86, @0.15 0.59 vs 0.68). The goal vectors alone
  are enough in principle, but the one-hot speeds learning -> kept.

**Best: wide_s1ft ckpt_1400** (`results/wide_s1ft_ckpt1400.npz` / `.json`), cli-bench settings
16 envs x 120 s (throughput/min, exit_ok, crash+offroad cars per 1000):

| demand | light (wide) | reservation (wide) | **wide_s1ft 1400** | wide_s0r 1400 | ppo17 5250 (narrow, old physics) |
|---|---|---|---|---|---|
| 0.05 | 10.0 / 1.00 / 0 | 10.6 / 1.00 / 0 | 10.4 / 0.980 / 21 | 10.3 / 0.968 / 32 | 10.4 / 0.971 / 29 |
| 0.15 | 23.0 / 1.00 / 0 | 31.6 / 1.00 / 0 | 26.4 / 0.825 / 175 | 24.8 / 0.779 / 221 | 28.6 / 0.883 / 117 |
| 0.30 | 25.7 / 1.00 / 0 | 60.9 / 1.00 / 0 | 40.0 / 0.641 / 359 | 35.8 / 0.571 / 429 | 45.1 / 0.713 / 286 |
| 0.50 | 26.1 / 1.00 / 0 | 81.1 / 1.00 / 0 | 47.5 / 0.480 / 520 | - | - |

Failure breakdown wide_s1ft ckpt_1400 (% of finished cars):

| condition | n | exit ok | crash | offroad | timeout+wrong | R exit/off | S exit/off | L exit/off | thr/min |
|---|---|---|---|---|---|---|---|---|---|
| solo | 224 | 100.0 | 0.0 | 0.0 | 0.0 | 100.0/0.0 | 100.0/0.0 | 100.0/0.0 | 7.0 |
| d0.02 | 158 | 100.0 | 0.0 | 0.0 | 0.0 | 100.0/0.0 | 100.0/0.0 | 100.0/0.0 | 4.94 |
| d0.05 | 341 | 98.0 | 1.8 | 0.3 | 0.0 | 98.3/0.8 | 97.2/0.0 | 98.3/0.0 | 10.44 |
| d0.15 | 1024 | 82.5 | 17.2 | 0.3 | 0.0 | 92.0/0.3 | 81.0/0.3 | 73.8/0.3 | 26.41 |
| d0.3 | 1996 | 64.1 | 35.4 | 0.5 | 0.0 | 81.1/0.1 | 57.6/0.6 | 53.8/0.9 | 39.97 |
| d0.5 | 3166 | 48.0 | 51.1 | 0.9 | 0.0 | 70.2/0.4 | 41.9/1.0 | 33.4/1.2 | 47.53 |

- **Solo gate passed** (already at it 400 of wide_s0): a single car on the wide crossing with
  grip limit reaches its exit in 100% of the cases for all three manoeuvres, offroad 0. Offroad in
  traffic <= 1% -> failures are car-car crashes; left turners and straight cars suffer most.
- vs baselines: beats the light on throughput from 0.15 up (26.4 vs 23.0 @0.15, 40 vs 25.7 @0.3)
  and is better than ppo17 at low demand (crash+offroad 21/1000 @0.05 vs 29), but crashes more
  than ppo17 at 0.15+ (shorter training: ~1.6k vs 5.2k iterations; the wide box has more conflict
  points and more speed variety). Reservation remains far ahead (61/min @0.3, 0 crashes).
- **Reverse gear: never used** by the deterministic policies (0 of ~1100 cars at 0.15 in both
  seeds, frac car-steps 0.0). Cars do brake more than on the narrow road (brake_frac 0.13-0.14 vs
  0.01-0.04 before; the grip limit forces slowing for turns).
- **Wobble** (straight arm segments, 16 envs @0.15; "leave" = after the box, no spawn merging):

  | policy | mean abs yaw rate rad/s | kappa sign changes /s | heading std deg | abs lateral speed m/s |
  |---|---|---|---|---|
  | ppo17 5250 (narrow, no grip) all / leave | 0.139 / 0.127 | 1.01 / 1.03 | 4.9 / 3.2 | 0.77 / 0.59 |
  | sig_s0 800 (narrow, no grip) all / leave | 0.116 / 0.108 | 0.94 / 0.82 | 4.5 / 3.1 | 0.64 / 0.55 |
  | wide_s1ft 1400 (grip) all / leave | 0.194 / 0.190 | 0.83 / 0.80 | 9.4 / 6.8 | 1.29 / 1.06 |
  | wide_s0r 1400 (grip) all / leave | 0.168 / 0.188 | 0.76 / 0.75 | 9.3 / 6.8 | 1.24 / 1.08 |
  | reservation (wide) all / leave | 0.023 / 0.006 | 0.21 / 0.00 | 2.1 / 0.3 | 0.15 / 0.03 |

  The grip limit lowers the weaving frequency (curvature sign changes 1.0 -> 0.8 /s) and caps the
  yaw rate at speed (<= a_lat_max / v = 0.29 rad/s at 14 m/s), but it does **not** remove the
  weaving: on the 14 m road the cars swing in wider, slower arcs (heading std 6.8 deg vs 3.2,
  lateral speed ~1 m/s). With no cost for it (outcome-only reward) and room on the road, the
  policy has no reason to drive straight; the only physics lever left would be stronger limits
  (kappa_rate, a_lat_max) or a cost that weaving actually incurs physically (e.g. energy /
  tyre-scrub speed loss; not modelled). Visible in `runs/wide/render_s1ft_1400/sheet.png` (cars 10-20 deg off
  the arm axis).
- **Driving side: right-hand traffic in both seeds** (s1ft: approach +1.69 m, 83% right of
  centre; leave +2.65 m, 91%; s0r: +1.99 m / 86%, +2.86 m / 94%; all four arms alike). On the
  narrow road seed 0 drove on the left; on the wide road both seeds chose the right (2 seeds only;
  the conventions are still a symmetry breaking until >= 20 seeds say otherwise, roadmap D).
- **Lanes: no.** The lateral-offset histograms on the arms are unimodal (approach peak at +2.25 m,
  std 2.2 m; leave peak +2.75..3.75 m, std 1.4-1.7 m); Sarle's bimodality 0.43-0.48 (< 0.555
  = uniform). Cars use the right half as one wide "lane", the leaving cars hug the outer side.
  (`analyze.py --hist` prints the histograms per arm.)
- **Signal**: still not used by receivers. @0.15 16 envs: s1ft normal 0.825 / zeroed 0.826 /
  shuffled 0.811 exit_ok; s0r 0.779 / 0.772 / 0.785. (An 8-env run of s0r ckpt_1250 showed
  0.751 -> 0.704 when zeroed; not reproduced at ckpt_1400 with 16 envs -> noise.) Own lamp
  encodes manoeuvre (MI 0.35 bit s1ft, 0.12 s0r) and region (0.49 / 0.16 bit), not braking
  (0.02 bit).
- Not done (time): TTC / closing-speed neighbour features or a recurrent policy (open problem
  "cars never brake/yield" - they now do brake 13% of the time, but crashes at 0.15 are still 17%).

## Summary / open problems (after the wide crossing)
- Physics: grip limit + reverse gear + wide geometry in place, parity-tested; baselines crash-free
  on both geometries (wide: lane-disciplined arrivals via `spawn_right`).
- Learning agents: solo driving is perfect; all residual failures are crashes in traffic, rising
  steeply with demand (1.8% @0.05, 17% @0.15, 35% @0.3). Throughput beats the light from 0.15 up.
- Wobble persists as a slow weave (physics alone does not remove it on a wide road).
- No lanes, no signal use, reverse never used; right-hand traffic in both seeds.
- Next (roadmap B): set/attention encoder over all neighbours, optional memory; more seeds.
  (Longer fine-tuning was tried afterwards, see below: plateau at ~17% @0.15; best now
  `results/wide_s1ft2_ckpt2650`.)

## Fine-tune continuation on the wide crossing (wide_s1ft2, wide_s0ft)
- Owner cap for this round: ~50 min of training per line (development phase; long runs only
  before merges), then evaluate. No code or sim changes.
- **wide_s1ft2**: warm start `runs/wide_s1ft/state.npz` (it 1636), same recipe as wide_s1ft
  (lr 3e-4 / critic 1e-3, lam 0.95, 2 workers x 64 envs, seed 1, ceiling 0.15, demand_lo 0.15);
  it 1637 -> 2723 (stopped). **wide_s0ft**: warm start `runs/wide_s0r/state.npz` (it 1936),
  switched to the same fine-tuning lr (was 1e-3), 1 worker x 64 envs, seed 0; it 1937 -> 3507.
  Its curriculum ceiling rose 0.15 -> 0.1725 at it 2306 (s1 never raised).
- Training stats (mean of first / last 100 it): s1ft2 exit 0.876 -> 0.889, crash+off 0.124 ->
  0.111; s0ft 0.835 -> 0.841, 0.165 -> 0.159. Both lines are **on a plateau**.
- Every checkpoint (every 50 it) robust-benched @0.15 (cli bench 32 envs x 120 s, seed 1000):
  s1ft2 1650-2700: 165-199 crash+offroad/1000 (mean ~178; ckpt_1400 of wide_s1ft: 185);
  s0ft: 256 at it 1950, then 183-212 from it 2300 on. Finalists re-benched at 0.05 / 0.3 and at
  0.15 with seed 2000 (a different traffic set; checkpoint selection on seed 1000 is biased).

Robust bench (32 envs x 120 s; throughput/min, exit_ok, crash+offroad cars per 1000):

| demand | light | reservation | wide_s1ft 1400 | **wide_s1ft2 2650** | wide_s1ft2 2500 | wide_s0ft 2850 | ppo17 5250 (narrow) |
|---|---|---|---|---|---|---|---|
| 0.05 | 9.8 / 1.00 / 0 | 10.4 / 1.00 / 0 | 10.0 / 0.951 / 49 | 10.0 / 0.949 / 51 | 10.0 / 0.960 / 40 | 9.9 / 0.951 / 49 | 10.2 / 0.962 / 38 |
| 0.15 | 23.0 / 1.00 / 0 | 32.7 / 1.00 / 0 | 27.0 / 0.815 / 185 | 27.5 / 0.832 / 168 | 27.4 / 0.833 / 167 | 26.9 / 0.817 / 183 | 27.8 / 0.833 / 167 |
| 0.15 seed 2000 | - | - | 26.4 / 0.807 / 193 | 27.1 / 0.827 / 173 | 27.3 / 0.836 / 164 | 26.4 / 0.809 / 191 | 28.4 / 0.859 / 141 |
| 0.30 | 25.4 / 1.00 / 1 | 62.4 / 1.00 / 0 | 41.0 / 0.636 / 364 | 42.4 / 0.665 / 335 | 41.3 / 0.643 / 357 | 38.6 / 0.601 / 399 | 46.0 / 0.71 / 290 |

(The 16-env numbers quoted for ckpt_1400 above, 21 / 175 / 359, are a luckier traffic set; at
32 envs it is 49 / 185 / 364. The 0.05 rate is dominated by few events: ~10 crashed cars.)
- Result: ~1.1k more fine-tuning iterations bought ~15-20 / 1000 at 0.15 and ~30 / 1000 at 0.3,
  nothing at 0.05. On seed 1000 the wide policy now **matches ppo17 at 0.15 (168 vs 167)**, but
  on the second traffic set ppo17 is still better (141 vs 173), so "matched" is within noise at
  best. Throughput stays above the light from 0.15 up (27.5 vs 23.0; 42.4 vs 25.4 @0.3).
  2500 vs 2650 are equivalent (mean over the four cells 182 both); **2650 chosen** for fewer
  off-road cars (8-10 vs 28-32 @0.15) and the best 0.3 value.
- Seed 0 trails seed 1 by ~15-35 / 1000 at every demand despite more iterations (it had the
  higher lr until it 1936).

Failure breakdown wide_s1ft2 ckpt_2650 (`analyze --breakdown`, 16 envs x 120 s, % of finished cars):

| condition | n | exit ok | crash | offroad | timeout+wrong | R exit/off | S exit/off | L exit/off | thr/min |
|---|---|---|---|---|---|---|---|---|---|
| solo | 224 | 100.0 | 0.0 | 0.0 | 0.0 | 100.0/0.0 | 100.0/0.0 | 100.0/0.0 | 7.0 |
| d0.02 | 158 | 98.7 | 1.3 | 0.0 | 0.0 | 100.0/0.0 | 97.9/0.0 | 97.9/0.0 | 4.88 |
| d0.05 | 342 | 96.5 | 3.5 | 0.0 | 0.0 | 100.0/0.0 | 95.3/0.0 | 93.9/0.0 | 10.31 |
| d0.15 | 1016 | 86.2 | 13.4 | 0.4 | 0.0 | 95.7/0.0 | 81.7/0.9 | 80.9/0.3 | 27.38 |
| d0.3 | 1996 | 65.6 | 33.8 | 0.6 | 0.0 | 86.6/0.3 | 60.8/0.4 | 49.5/1.1 | 40.91 |
| d0.5 | 3033 | 50.4 | 48.3 | 1.2 | 0.0 | 77.1/0.6 | 42.8/1.2 | 32.7/1.9 | 47.81 |

Solo gate still 100%; offroad <= 1.2%; failures are car-car crashes of straight and left cars
(right turners 96% ok @0.15). (This 16-env set gives 138 / 1000 @0.15; the 48-env run below
201 - scenario variance is large, use the 32-env bench for comparisons.)

Behaviour (`analyze`, @0.15, 16 envs, ckpt_2650 / s0ft 2850):
- **Wobble** (straight arm segments; all / leave):

  | policy | mean abs yaw rate rad/s | kappa sign changes /s | heading std deg | abs lateral speed m/s |
  |---|---|---|---|---|
  | wide_s1ft 1400 | 0.194 / 0.190 | 0.83 / 0.80 | 9.4 / 6.8 | 1.29 / 1.06 |
  | wide_s1ft2 2650 | 0.228 / 0.223 | 1.02 / 1.13 | 9.7 / 6.7 | 1.31 / 1.02 |
  | wide_s0ft 2850 | 0.197 / 0.208 | 0.93 / 0.91 | 9.0 / 7.0 | 1.17 / 1.07 |

  Fine-tuning did not reduce the weave; s1 even steers a bit more often (sign changes 0.8 ->
  1.1 /s on the leave segments) at the same amplitude (heading std ~6.7 deg). As expected with
  an outcome-only reward.
- **Braking / reverse**: brake_frac 0.178 (s1) / 0.174 (s0), up from 0.13-0.14. Reverse gear:
  3 of 1090 cars in s1ft2 2650 reverse briefly (~1.7 s, 0.06% of car-steps); s0ft never.
- **Driving side**: right-hand traffic in both seeds, all four arms: s1 approach +2.08 m (90%
  right of centre), leave +3.24 m (94%); s0 approach +1.50 m (82%), leave +3.40 m (96%).
- **Lanes: no.** Approach histograms unimodal (s1 peak +2.25 m, std 2.1 m, bimodality 0.49; s0
  +1.25 m, 0.41). The s1 leave histogram (peak +4.0..+5.0 m, std 1.6 m) has a shoulder at
  +2..+3 m and a bimodality coefficient of 0.57 (just above 0.555, driven by skew; one mode);
  s0 leave 0.50. Not two lanes.
- **Lamps**: no reliable receiver use. 16 envs showed normal 138 / zeroed 162 / shuffled 192
  per 1000 for 2650, but with 48 envs: 2650 201 / 195 / 204, 2500 187 / 198 / 205, 2700
  191 / 193 / 209 -> zeroing does nothing; shuffling costs ~+10-18 / 1000 in all three
  (a hint at most; a shuffled lamp is out-of-distribution input). s0ft: 192 / 183 / 173 (16
  envs). Own lamp: MI manoeuvre 0.25 bit, region 0.38, braking 0.02 (s1); 0.14 / 0.13 / 0.01 (s0).
- Saved: `results/wide_s1ft2_ckpt2650.npz/.json` (best, seed 1), `results/wide_s0ft_ckpt2850.npz/.json`
  (best seed 0 by 32-env bench @0.15, for the driving-side comparison). The .json is
  `export_viewer.load_policy` output (same format as `wide_s1ft_ckpt1400.json`). Tests: 46 passed.
- Conclusion: the K-nearest MLP with this reward has plateaued at ~17% crashed cars @0.15 on the
  wide crossing (ppo17 needed 5.2k it for the same level on the narrow one). Further fine-tuning
  is unlikely to reach < 50 / 1000; next steps remain roadmap B (set encoder, memory) or
  TTC/closing-speed features.

## Refactor: one frontend, compat module, generated JS constants (no training, behaviour unchanged)
- Removed the standalone viewer (`viewer/app.js`, `index.html`, `policies.js`, `dist/`),
  `results/smartcrossing_viewer.html` and `legacy/`. `sim.js` now lives in `site/src/js/`; the parity
  test is `tests/test_sim_js_parity.py` (+ `tests/sim_js_parity_runner.js`).
- `export_viewer.py` split: checkpoint adaptation (feature-key remap, legacy interface/physics,
  `physics_of`, `load_adapted_policy`) -> `smartcrossing/policy_compat.py`; German observation labels ->
  `site/obs_labels.py`; `round_policy` -> `site/build.py`.
- sim.js takes every number from `js/sim-config.js`, generated by `site/build.py` from
  `sim.js_constants()` (checked by `test_generated_config`). Spawn literals got names
  (`SPAWN_BACK`, `SPAWN_CURB_GAP`, `SPAWN_RIGHT_MIN`).
- Dead experiment knobs removed: ES `--r-geo` / `--r-queue` (use `--cfg`), PPO `--alpha` (coop runs) and
  `--queue-coef` (never used); `speed_tau` is the constant `sim.SPEED_TAU`. Kept for site checkpoints:
  `neighbour_vis` (ppo11/14/15) and `speed_control` (ppo16).
- Check: cli bench (8 envs x 60 s, seed 1000, @0.15) and final-state hashes bit-identical before/after for
  wide_s1ft2 2650, ppo17 5250, ppo16, ppo15, es5, wide_s0ft 2850, light, reservation; full site re-bench
  (35 policies x demands, replays) identical. Tests 46 -> 47 passed.

## Higher demand: spawn capacity (no training)
- Owner: the start page slider should go to ~1 arrival/s/arm now that the road is 14 m wide.
- **Measured bottleneck (old spawn logic)**, 16 envs x 120 s, seed 1000 (spawned vs arrived cars/min, all 4 arms):

  | demand | arrivals/min | reservation spawned / thr | light spawned / thr | wide_s1ft2 2650 spawned / thr |
  |---|---|---|---|---|
  | 0.3 | 68.6 | 67.0 / 60.9 | 34.8 / 25.7 | 67.0 / 40.9 |
  | 0.5 | 116.1 | 90.3 / 81.1 | 35.2 / 26.1 | 101.5 / 47.8 |
  | 0.8 | 190.8 | 94.2 / 84.4 | 35.5 / 26.5 | 115.2 / 50.1 |
  | 1.0 | 239.5 | 94.1 / 84.6 | 35.8 / 26.6 | 116.0 / 50.4 |

  Pure entry capacity (all cars full throttle straight ahead, no driver limit): 0.69 /s/arm full
  width, 0.85 right half (165 / 204 cars/min). Cause: one car per arm and step at the lateral
  offset drawn at arrival; the queue head waits until exactly that spot is clear, and outside its
  own corridor every car ahead counts as standing (up to 7 + 1.5 x 8 = 19 m clearance), so a car
  slightly beside the previous one waits ~2 s. `n_slots` = 32 never binds (max mean 22 cars,
  never all slots full; 64 slots gave bit-identical results).
- **Change** (`Sim._arrivals`, mirrored in `sim.js`): each arrival draws u in [0,1) (replaces the
  lateral draw; same number of draws, CRN unchanged). The queue head enters at its own spot (u
  across the spawn range, on a 0.2 m grid `SPAWN_LAT_STEP`) if clear, else at the
  floor(u x n_clear)-th clear grid position (uniform among the free ones, both halves: no side
  preference). Up to `SPAWN_MAX_PER_STEP` = 3 cars per arm and step (side by side if the clearance
  permits); cars spawned this step count with their real velocity in the closing-speed check.
  Clearance rules unchanged; baselines keep `spawn_right` (same rule within the right half).
  Candidate check is restricted to cars within reach (clearance + headway x max closing speed),
  so the cost stays small: 64 envs, brain: arrivals 2.0 ms of 16 ms/step at 0.15, 10 of 27 ms at 1.0.
  Pure entry capacity now 1.7 /s/arm (full width), 1.2 /s/arm (right half).
- **After** (same setup as the table above):

  | demand | arrivals/min | reservation spawned / thr | light spawned / thr | wide_s1ft2 2650 spawned / thr / crash per 1000 |
  |---|---|---|---|---|
  | 0.3 | 68.6 | 68.2 / 61.9 | 37.8 / 28.2 | 68.5 / 39.0 / 386 |
  | 0.5 | 116.1 | 96.8 / 85.8 | 38.9 / 29.1 | 115.6 / 49.1 / 546 |
  | 0.8 | 190.8 | 101.0 / 90.3 | 40.0 / 29.7 | 185.8 / 53.2 / 697 |
  | 1.0 | 239.5 | 102.8 / 91.7 | 40.3 / 30.2 | 211.6 / 55.8 / 721 |

  The brain now gets (almost) the full inflow; its throughput saturates at ~55/min because the
  extra cars crash (72% at 1.0). Reservation and light are limited by their own intersection
  capacity now (their queues reach back to the entry), not by the spawner. Light had one crash
  pair at 0.3 (a lane merge in a queue; the old logic had 1 / 1000 at 0.3 in the robust bench too).
- **Brain degradation (trained on the old logic, not retrained per owner)**: robust bench (32 envs x
  120 s), crash+offroad per 1000, old -> new: 0.05: 51 -> 57; 0.15: 168 -> 200; 0.3: 335 -> 384.
  At 0.15 over four traffic sets (seeds 1000-4000): old 168/173/198/171 (mean 178), new
  200/181/194/196 (mean 193): **~+15 / 1000 (~9%)**, partly noise; no breakdown (throughput 26.5
  vs 27.5 /min @0.15). A first variant (always uniform among the clear spots, no own spot) was
  worse at 0.15 (207). A fine-tune on the new spawn logic would likely recover it (not done: owner).
- Site: demand slider 0 ... 1.0 /s/arm (240 cars/min); bench demands extended by 0.5 and 1.0.

## Roadmap C feasibility: torch/CUDA sim (2026-10-11, local RTX 3070 Ti)
- **Why**: decide with numbers whether a GPU port pays off before porting PPO.
- **Profile** (`ppo.py`, 4 workers x 24 envs, 8 iterations): sparse traffic rollout 0.3 s vs update
  0.95 s per iteration; dense traffic rollout 0.65 s vs update 3.6 s (numpy update single-threaded,
  ~45 us/sample). Inside a worker rollout: sim.step 54 %, observe 21 %, actor+critic forward 17 %.
- **Port**: `smartcrossing/sim_torch.py` (`TorchSim`, full `Sim` incl. arrivals/collisions/rewards/
  observe, optional CUDA-graph capture). Not ported: per-car `Stats.travel` lists, `events`; graph
  mode caps the arm queue at 256 (flag on overflow). `rng="torch"` gives different traffic streams
  (same distributions); paired-traffic comparisons need `rng="numpy"` (slow).
- **Parity** (`tests/test_sim_torch_parity.py`, skipped without torch): 2 configs x 300 steps, cpu +
  cuda float64: discrete state and stats identical, floats within 1.4e-14; graph replay == eager.
- **Throughput** (`scripts/bench_sim_torch.py`, env-steps/s incl. policy forward, one process;
  numpy: 8.7k at B=64, flat in B, ~35k with 4 workers):

  | B | cuda f32 graph | cuda f64 graph |
  |---|---|---|
  | 64 | 20.5k | 16.7k |
  | 1024 | 130k | 62k |
  | 4096 | 168k | 72k |
  | 16384 | 184k | 74k |

  Eager CUDA is launch-bound (~55 ms/step); torch on CPU is 3-5x slower than numpy.
- **Conclusion**: worth it, but the PPO update dominates (~85 % of wall time in dense traffic): port
  `ppo_update` to torch/GPU first, then roll out with `TorchSim` (graph, f32, B ~ 1024-4096, short
  horizon 64-128, buffers/GAE on GPU). Expected ~5x end-to-end. torch not yet in pyproject (CI CPU only).

## Roadmap C stage 1: PPO update in torch (2026-10-11)
- `smartcrossing/ppo_torch.py`, `ppo.py --update-device {numpy,cpu,cuda} --update-dtype`; default
  numpy unchanged. torch declared as optional poetry group `gpu` (cu118 source); CI does not install it.
- Parity (`tests/test_ppo_torch.py`): float64 params/log-std/Adam state within 1e-12 (cpu, cuda);
  80k-sample batch, 4 epochs: max deviation 1.1e-16 (f64), 1.5e-7 (f32).
- Update time (80k samples, 4 epochs x 8 minibatches): numpy 2.98 s, cuda f32 0.16 s (19x),
  cuda f64 0.23 s, torch cpu f32 0.62 s.
- Smoke runs (demand 0.3, 4 workers x 24 envs, ~110k car-steps/iter; runs/smoke_ppo_{torch,numpy}):
  per iteration cuda 1.10 s (rollout 0.89 / update 0.21) vs numpy 5.41 s (0.76 / 4.65): ~5x.
  Curves identical up to iter 6, then same trend (f32 rounding). The rollout is now the bottleneck
  -> stage 2: TorchSim rollouts on the GPU (CUDA graph, f32, 1024-4096 envs, GAE/buffers on GPU).

## Roadmap C stage 2: rollouts on the GPU (2026-10-11)
- `smartcrossing/rollout_torch.py` (`TorchRollout`: one TorchSim, CUDA-graph step, buffers/GAE on the
  GPU, batch handed to TorchPPO as device tensors); `ppo.py --rollout-device cuda --rollout-envs B
  --rollout-dtype`; eval stays on the numpy path (paired traffic); log/checkpoint format unchanged.
- Parity (`tests/test_ppo_gpu_rollout.py`): float64 with numpy traffic + injected noise, 3 rollouts incl.
  an episode reset: obs, actions, log-probs, values, advantages, returns, rewards identical (0.0) on cpu
  and cuda.
- Timing (demand 0.3, early training; runs/bench_ppo_gpu.jsonl):

  | config | car-steps/iter | s/iter | car-steps/s | GPU mem |
  |---|---|---|---|---|
  | numpy rollout + numpy update | 65k | 3.58 | 18k | - |
  | numpy rollout + cuda update | 65k | 0.79 | 82k | small |
  | GPU B1024 T32 | 158k | 0.46 | 343k | 1.0 GB |
  | GPU B2048 T32 | 312k | 0.62 | 502k | 2.1 GB |
  | GPU B2048 T64 | 661k | 1.19 | 557k | 4.1 GB |
  | GPU B4096 T32 | 617k | 1.70 | 364k | 4.4 GB |

  ~28x more car-steps/s than the old path. Bigger batches mean fewer updates per car-step, so
  car-steps/s is not learning speed; the first real round has to show the sample efficiency.
- Smoke run from scratch (runs/smoke_ppo_gpu, B1024 T32, 150 it, 65M car-steps in 88 s): like the numpy
  smoke run it settles into "stand and wait" early (few finished cars, crash shares of the few finished
  ones noisy); no qualitative check beyond that. A warm-start check was not possible locally (the
  committed checkpoints hold only the actor; full state.npz files were in the cloud runs/).
- Limits: torch RNG streams differ from numpy (same distributions); graph-mode arm queue capped at 256.
