# Smart Crossing — Plan & Decisions

Goal: a signal-free four-way intersection crossed only by learning autonomous cars.
Questions: (1) how high can throughput get without crashes, (2) does coordination emerge
when cars can emit a free, un-predefined "light" signal, (3) do conventions such as
right- vs left-hand traffic emerge on their own (driving side: answered by earlier runs, now
given by right-half spawning; lanes, priority, lamp meaning stay open), (4) what happens when differently
trained agent types share the road (ncase.me "Evolution of Trust" style).

## Fixed decisions (from the owner)
- No lanes, lane paths, traffic lights, priority rules or driving-side conventions for the
  learning agents. Steering is learned (2D kinematics), not just speed. Exception (owner,
  2026-10-11): cars enter the map on the right half of their road (see next training round).
- Old pygame/DQN code was replaced (deleted; it is in the git history before the refactor).
- Geometry (since the wide-crossing change): 4 arms, roads 14 m wide (W = 7 m half width: room
  for two cars side by side per direction), curb radius Rc = 9 m, arms L = 50 m. The original
  7 m crossing (W 3.5, Rc 6, no grip limit) is `SimConfig(**sim.NARROW)`; checkpoints trained on
  it still run on it (`PolicyDriver.load` / the site's physics apply the stored geometry).
- Tyre grip (physics, not a style penalty): actual curvature <= min(kappa_max, a_lat_max / v^2),
  a_lat_max = 4 m/s^2, applied after the steering-rate limit. Before, a 5 m turn radius at
  14 m/s (39 m/s^2) was allowed, which made cars wobble visibly.
- Reward principle: reward only outcomes (arriving, crashes/off-road, time/delay, waiting
  demand). Crash / off-road cost = base + factor x impact speed (car-car: relative speed of the
  pair; off-road: own speed), r_crash = r_offroad = -1.5, -0.1 per m/s; every car involved pays
  its own term (no blame). Physical limits (max speed, braking, turn radius, steering rate) are fine. No reward
  terms that prescribe driving style (e.g. no penalty for wobbling, no lane-keeping bonus).
  Progress shaping (route distance) is tolerated as a learning aid only; prefer removing it once
  agents can learn without it.
- Training budget (owner): during development keep trainings short (<= ~1 h per round); long
  trainings only right before a merge to main.
- Before every training run (owner, 2026-10-11): give a duration estimate and, if the scope can be
  fixed in advance, 2-3 options (short for fast iteration ... long for quality) with expected
  benefit; the owner chooses. Training runs on the owner's PC (GPU), which is busy meanwhile.
- Clean code (owner, 2026-10-11): insert refactoring steps regularly (dead code, duplication,
  stale docs/compat layers, unclear names) so no technical debt builds up. The coordinator decides
  when: between rounds, not while a sim change or training is in flight. Procedure: read-only audit
  first, then refactor with acceptance "pytest green + bench/parity identical before vs after".
- Deliverable: a live, in-browser visualisation with a few sliders (keep it simple),
  incl. mixing agent types.

## Architecture
- `smartcrossing/sim.py`: batched numpy simulator, state arrays (B envs, N slots).
  Kinematic unicycle with curvature, steering-rate and grip limits, dt=0.1 s. Cars spawn at a
  random lateral offset on the right half of the road in driving direction (`spawn_right`, default
  for baselines and learners; own spot if clear, else uniform among the clear spots; up to 3 per arm
  and step, side by side), must exit through target arm. Crashed cars stay 5 s as wrecks (hazard
  lights, driving into one = crash). Car-car collision via
  3 discs per car, off-road via rounded-corner distance. Removal on crash/offroad/exit/timeout.
  Observations are mirror-neutral: own state in a canonical frame (entry arm = south; position
  in fixed 10 m units, not scaled by L/W), ego-frame vectors to the exit gate and to the exit
  mouth (25 m units), 5 edge rays, own curvature and grip-limited max curvature, K=4 nearest
  neighbours (ego frame: position, heading, speed, curvature, brake light, hazard light, one lamp
  `n_signal=1`) but NOT their destination. Action = (accel, curvature, signal[n_signal]) in [-1,1].
  Common random numbers: env b uses traffic stream b % n_scenarios.
- `smartcrossing/baselines.py`: rule-based references with lanes (light / reservation(AIM) / none),
  1 lane per direction on the 7 m road, 2 on the 14 m road (right-hand traffic).
- `smartcrossing/render.py`, `metrics.py`, `cli.py` (bench / render).
- `smartcrossing/ppo.py`: numpy PPO with parameter sharing (main training method since ppo6;
  see experiments.md). `smartcrossing/analyze.py`: signal zero/shuffle interventions,
  MI(lamp; manoeuvre/braking/region), driving side per arm / manoeuvre, lateral-offset
  histograms (lanes?), wobble metrics, failure breakdown (solo / per demand / per manoeuvre).
  GPU path (optional `poetry install --with gpu`, torch): `sim_torch.py` (TorchSim, parity-tested
  against `sim.py`), `rollout_torch.py` (rollouts on the device, CUDA graph), `ppo_torch.py` (update);
  enabled by `ppo.py --update-device cuda --rollout-device cuda`. Warm start: `--init-policy`
  (actor from an older-interface checkpoint) + `--critic-warmup N` (first N iterations critic only).
- `smartcrossing/policy.py` (shared MLP, numpy, `PolicyDriver`). The old ES trainer was removed
  (history only); ES checkpoints still load via `policy_compat.py`.
  `smartcrossing/policy_compat.py`: running old checkpoints (observation-layout remap by
  feature name, legacy interface/world values, `physics_of` for the browser).
- SimConfig `neighbour_vis` (perception range, default 30 m; 60 in ppo11/14/15) and
  `speed_control` (action[0] as target speed, ppo16 only) are per-policy "interface" settings kept
  for those checkpoints (no benefit, see experiments.md): stored in the checkpoint cfg, applied by
  `PolicyDriver.load`, exported as the site's physics.
- `site/`: the website is the only frontend. `site/src/js/sim.js` is the JS port of sim + policy
  (checked against Python by `tests/test_sim_js_parity.py` via node); its constants come from
  `js/sim-config.js`, generated by `site/build.py` from `sim.js_constants()`.

## Self-evaluation workflow
- `pytest` invariants; `cli bench` prints one JSON line (throughput, crashes/1000, delay,
  Jain fairness, ...); `cli render` writes `sheet.png` (9-frame contact sheet) + `episode.gif`.
- Every experiment gets an entry in `docs/experiments.md` (hypothesis, config, result, commit).
- Subagents look at images themselves and report numbers; orchestrator keeps context small.
- Emergence claims require ablations: signal zeroed/shuffled, no-comm training, MI(signal; intent).

## Status (2026-10-11)
Done: simulator + baselines + metrics + tests; PPO training (numpy and GPU); emergence analysis
tooling; website (site/) incl. sliders, agent mixing, side-by-side comparison; wide crossing, grip
limit, reverse gear, impact-speed crash cost, wrecks / brake light / hazard / one lamp / right-half
spawning; round 1 (crash cost x3, warm start).
Main brain: round-1 brain results/round1_x3.npz (site main `r1_main_x3`; previous: wide_s1ft2_ckpt2650).
Open: cars see obstacles but don't brake to a stop; crashes in the box; weaving; lamps unused.
Next: round 2 (A/B/C, see Roadmap (E)), then (B), (C), (D).

## Roadmap (owner, after the wide-crossing change)
- (A) Driving quality: failure breakdown (offroad vs crash, solo vs traffic, per manoeuvre and
  demand), solo-driving gate (a car alone never leaves the road), impact-speed crash/offroad
  cost, physically visible neighbour features (curvature, acceleration).  <- done
- (B) Perception / network: set (attention) encoder over all neighbours instead of K nearest,
  optional recurrent memory (GRU).
- (C) Geometry-independent inputs throughout + randomized geometries during training; GPU port
  (owner has an RTX 3070 Ti, local training later).
- (D) Norm experiments: >= 20 seeds per configuration; lateral position (lanes; driving side is
  no longer open once spawning is right-only),
  priority patterns, signal MI, ablations, separately trained populations (cf. Pal et al.,
  ICLR 2022, "Emergent Road Rules").

- (E) Safety: "rather a jam than a crash" (owner, 2026-10-11). Target: almost no crashes even in dense
  traffic. Owner's choice: a fixed, clearly higher crash/off-road cost (instead of an adaptive crash
  budget); the right factor is found by short comparison runs (too high -> cars refuse to enter).
  Still outcome-only. Perception (B) and brake lights should help as well.
  Diagnosis 2026-10-11 (experiments.md): cars see the obstacle but don't brake to a stop. Stop gate
  (acceptance test for every new brain, like the solo-driving gate): a car alone with a wreck or a
  standing car on its path (several speeds/gaps) must stop without crashing in >= 95% of cases.
  The stop gate only measures; the fix belongs in training. Reward check: for a single car, stopping
  for a wreck is clearly better (~+2) than hitting it (~-6), so the reward is right in principle.
  Suspected causes: (1) no exploration of hard braking (warm start from a never-stopping brain,
  logstd -2); (2) multi-agent trap of the equal crash cost: a car that brakes gets rear-ended and pays
  as much as the hitter, so braking is punished while nobody brakes; (3) standing obstacles were rare.
  Round 2 (owner, 2026-10-11): three variants, equal training time, paired comparison:
  A = more exploration; B = A + obstacle scenarios (some envs start with a wreck/standing car on a
  path); C = B + crash cost by OWN impact speed (standing car pays only the base cost; still
  outcome-only physics). The owner decides on C after the numbers.
- (F) Geometry variety (long term): roundabout-like and non-right-angled crossings, then a whole
  village (road network; cars start/end at driveways or the map edge). Needs geometry-independent
  inputs (C), a road-graph representation instead of the hard-coded 4-arm crossing, and route
  information (next turn) as input instead of only the exit arm.
- Run tracking (owner, 2026-10-11), with the next refactoring step: git commit hash (+ dirty flag)
  in every run's args.json, full state.npz of important runs kept under results/ (cloud runs were
  lost), a small script listing runs/ (name, commit, settings, key metrics). The owner needs no
  tracking UI (the /lernen curve is enough); tooling is for the coordinator and agents, their call.
- Quick wins: small GIF of the crossing at the top of README.md (rendered from the main brain);
  short, plain explanation on /lernen of how training with many cars works (PPO: all cars share one
  brain, each learns from its own reward; ES: each mutation drives in its own environments, all
  cars there use it, fitness = sum over all cars).

## Decided for the next training round (owner)
- Spawn on the right half of the road in driving direction, always (`spawn_right` for the learners
  too; owner, 2026-10-11): fewer crashes / space problems at the map edge. This gives up the
  "does a driving side emerge on its own" question (earlier runs already showed the cars agree on a
  side); the other norm questions in (D) remain.
- Wrecks: crashed cars stay for 5 s as standing obstacles (driving into one = own crash) with hazard
  lights that other cars can see (a physically visible signal, no rule information), then vanish.
  Mirror in sim.js / site rendering.
- Higher fixed crash/off-road cost, see (E).
- One free lamp channel instead of two (n_signal = 1): easier to read and to analyse.
- Neighbour acceleration observation becomes a binary brake light (on when braking), as on real cars.
- No turn signals: a built-in blinker would leak the destination, which is deliberately hidden so
  that a lamp "language" has something to say. The free lamp is the place where one could emerge.
- Brake lights are drawn on the website now (visual only).
