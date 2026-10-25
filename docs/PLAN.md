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
- Honest perception of non-participants (owner, 2026-10-24): all obstacles show hazard lights (single
  obstacles and blockade bodies, wreck-type AND parked-type) - hazard = not a participant, hazard off = a
  participant. Brake light off, lamp 0. `obstacle_parked` only changes the drawing (undimmed body + hazard blink).
  Training semantics change only with obstacles on; obstacles off is bit-identical, old checkpoints unaffected.
- Deliverable: a live, in-browser visualisation with a few sliders (keep it simple),
  incl. mixing agent types.

## Architecture
- `smartcrossing/sim.py`: batched numpy simulator, state arrays (B envs, N slots).
  Kinematic unicycle with curvature, steering-rate and grip limits, dt=0.1 s. Cars spawn at a
  random lateral offset on the right half of the road in driving direction (`spawn_right`, default
  for baselines and learners; own spot if clear, else uniform among the clear spots; up to 3 per arm
  and step, side by side), must exit through target arm. Crashed cars stay 10 s as wrecks (was 5 s until round 7) (hazard
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
- `smartcrossing/render.py`, `metrics.py`, `cli.py` (all commands; one driver factory: `drivers.py`), `jobs.py`
  (spawn pool + JSONL appends of the measurement commands).
- `smartcrossing/ppo.py`: numpy PPO with parameter sharing (main training method since ppo6;
  see experiments.md; training loop + arguments; networks, rollout worker and numpy update in `ppo_numpy.py`,
  background mode in `background.py`). `smartcrossing/analyze.py`: signal zero/shuffle interventions,
  MI(lamp; manoeuvre/braking/region), driving side per arm / manoeuvre, lateral-offset
  histograms (lanes?), wobble metrics, failure breakdown (solo / per demand / per manoeuvre).
  GPU path (optional `poetry install --with gpu`, torch): `sim_torch.py` (TorchSim, parity-tested
  against `sim.py`), `rollout_torch.py` (rollouts on the device, CUDA graph), `ppo_torch.py` (update);
  enabled by `ppo.py --update-device cuda --rollout-device cuda`. Warm start: `--init-policy`
  (actor from an older-interface checkpoint) + `--critic-warmup N` (first N iterations critic only).
- `smartcrossing/policy_set.py` (Roadmap B, optional `ppo --arch set`): neighbour set encoder, same
  observation (no new information): shared per-neighbour MLP phi, pooling over the present neighbours
  (attention with the ego features as query, or DeepSets mean / max), MLP trunk on [ego, pooled].
  Permutation-invariant over the slots, so `SimConfig.n_neighbours` (default 4 = all older checkpoints;
  an interface value like n_signal) can grow without new weights. numpy (PolicyDriver, workers), torch
  (rollout_torch, ppo_torch via autograd) and JS (sim.js) forwards, parity-tested
  (tests/test_policy_set.py, test_sim_js_parity.py). Checkpoints carry an `arch` field (absent = MLP,
  older files unchanged); `policy.linear_forward` / `ppo_torch.linear_forward` and `policy.param_count` are the
  one dispatch between the two architectures. Site: sim.js runs set brains; the site build handles both (its `Brain`
  helper: snapshot, bench payload, rounding).
- `smartcrossing/bc.py` (`ppo --init-bc TEACHER.npz`): behaviour cloning of an MLP brain into a new actor (set
  encoder or MLP of other sizes; the student may see more neighbours). Other `ppo.py` options worth knowing:
  `--max-minutes` (stop after the iteration past the limit; the final checkpoint is always written and
  evaluated), `--init-policy` / `--critic-warmup` (above), `--arch set` / `--critic-arch`.
- `smartcrossing/gates.py`: stop gate (`cli gate`), a measurement: one car alone in front of a wreck / parked
  car (speeds x gaps x positions x types x distractors) must stop without crashing.
- `smartcrossing/protocol.py`: the protocol constants, defined once: SELECTION_SEEDS 1000/2000/3000 (checkpoint
  selection, evalset, training evals), REPORT_SEEDS 4000/5000/6000 (held out: everything on the website), ENVS x
  SECONDS = 8 x 120 s per bench run, the demand lists (EVAL / SELECT / SWEEP / SITE_DEMANDS), SWEEP_SECONDS 180.
- `smartcrossing/evalset.py` (`cli evalset`): the standard eval protocol = 4 demands (0.05, 0.15, 0.3, 0.5) x
  3 traffic seeds (1000, 2000, 3000) x 8 envs x 120 s in the default world + the stop gate, appended as JSON
  lines (format of results/round2_eval.jsonl / round4_eval.jsonl). Its `bench_one` / `KEEP_KEYS` are shared by
  `cli bench`, `ppo.eval_job` and the site bench.
- `smartcrossing/select.py` (`cli select`): model selection over the checkpoints of runs: stop gate >= 0.95 as a hard
  filter, then the best throughput at 0.5 under a crash limit at 0.3 (relative to the run's first checkpoint).
- `smartcrossing/deadlock.py` (PLAN (G), measurement): `cli deadlock` = demand sweep with per-step recording and an
  operational deadlock detector (stuck cars in the box, constellation, end cause, lamps; action-noise / lamp
  interventions); `cli deadlock-summarize` = table + the fundamental-diagram JSON (results/deadlock_fundamental.json,
  `--name round7_best=brain` names the site's curves); `cli cloud` (`smartcrossing/fundamental_cloud.py`) = scatter
  cloud of single 60-s windows after the 30-s warm-up (results/fundamental_cloud.json, seeds 4000 + k; `cloud_windows`
  is the definition, the start page's live dot uses the same one: site/src/js/fdwindow.js, node-tested);
  `cli incentives` (`smartcrossing/incentives.py`) = reward accounting of a stuck car.
- Obstacle scenarios (training option, round 2): SimConfig `obstacle_envs` (fraction of envs; 0 = off and
  bit-identical), `obstacle_rate`, `obstacle_time`, `obstacle_max`, `obstacle_parked`, `obstacle_box`: wrecks /
  parked cars appear on a path (never closer than a car could stop). `obstacle_block` (blockade scenarios, round 7; 0 = off, bit-identical):
  that share of the placements is a blockade of 3 bodies crosswise across the full road width of an arm (both
  directions), so only stopping avoids the crash; the stop gate reports its block cases as `block_*`.
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
  `js/sim-config.js`, generated by the site build from `sim.js_constants()`.
  Build: `python site/build.py` (thin entry point) runs the package `site/sitebuild/`: context (paths, source =
  runs/ and / or the committed snapshot results/site/, discovered checkpoints, gate cache; one explicit object, no
  module globals), discover (Brain, checkpoints, warm-start chains, snapshot writing), collect (which brains / timelines /
  side notes), bench (cached site bench on REPORT_SEEDS, side-by-side replays), curves, fundamental (stop gate, fundamental
  diagram + cloud from results/), texts (run names / notes, one entry per run with de + en), pages (frame, public/).
  results/site/ = compact snapshot of what the build used from runs/ (exact weights, chains, thinned curves), so a fresh
  clone builds the same site.
  i18n: German is the source text of the static HTML; English via data-i18n keys (js/i18n-en.js), generated sentences
  via SCX.t keys in js/i18n-de.js + i18n-en.js, build data carries *_en fields (SCX.tl); tests/test_site_i18n.py keeps
  both languages complete.

## Self-evaluation workflow
- `pytest` invariants; `cli bench` prints one JSON line (throughput, crashes/1000, delay,
  Jain fairness, ...); `cli render` writes `sheet.png` (9-frame contact sheet) + `episode.gif`.
- Every experiment gets an entry in `docs/experiments.md` (hypothesis, config, result, commit).
- Subagents look at images themselves and report numbers; orchestrator keeps context small.
- Emergence claims require ablations: signal zeroed/shuffled, no-comm training, MI(signal; intent).

## Status (2026-10-25)
Done: simulator + baselines + metrics + tests; PPO training (numpy and GPU); emergence analysis
tooling; website (site/, German + English) incl. sliders, agent mixing, side-by-side comparison, reward visual,
set-brain view, fundamental diagram; wide crossing, grip limit, reverse gear, impact-speed crash cost, wrecks /
brake light / hazard / one lamp / right-half spawning; rounds 1-7 (crash cost x3, obstacle scenarios,
set/attention encoder, dense traffic, blockade scenarios); stop gate incl. block cases; deadlock study;
multi-seed site bench on held-out report seeds; run tracking; refactoring steps (latest: refactor 4, 2026-10-25).
Main brain: round-7 brain results/round7_best.npz (site main `r7_DOB_s1@1050`; previous r6_DO_s2@750).
Crash rule: equal cost for all involved (relative impact speed); own-speed variant C was worse (round 3).
Open: deadlocks / near-gridlock in the box at high demand (cars removed after max_age 60 s); remaining
crashes in the box; lamp unused so far.

## Next up (ordered by the coordinator, owner 2026-10-14)
1. Site: make the brain view on /gehirn larger (zoom / full-width / fullscreen toggle; owner: too small).
   Same step: keep the start page's "Flackern (Störung)" slider (lamp noise, never used in training;
   an intervention tool: do cars rely on the lamp?) but explain it in one plain sentence (owner: option a).
2. [done: refactor 4, 2026-10-25] Refactoring step (due after rounds 2-4: set encoder + site build grew).
2b. [done 2026-10-21] English (owner, 2026-10-21): the whole website in German AND English with a language toggle top
   right (DE/EN; remembers the choice, default by browser language, German stays the primary text);
   all page texts incl. the data-driven sentences in pages.js, labels from obs_labels, run notes,
   chart labels. README.md in English only. Do it after the refactoring step (site build is cleaner
   then) and in parallel with the deadlock measurement (site-only vs sim/analysis: no file overlap).
2c. Site (with 2b): fundamental diagram (owner, 2026-10-21): throughput vs density for brain, light and
   reservation (traffic-flow "Fundamentaldiagramm": flow rises to capacity, then drops - capacity drop,
   spillback, gridlock), plus a live point on the start page showing where the running sim sits on the
   curve. Needs density over time from the bench (cars on the crossing / approaches). Same step: demand
   slider max 240 -> 120 cars/min (0.5 /s/arm: above every driver's capacity incl. reservation ~90/min,
   so the collapse stays visible; 240 is unrealistic); drop demand 1.0 from the site bench.
3. (G) Deadlock study, measurement first (see below): incl. a demand sweep 0.05 ... 0.5 for the main
   brain vs light vs reservation (throughput, deadlocks, timeouts, queue at the edges).
4. (G) Deadlock fixes, one variable at a time with 2+ seeds each. First candidate: train with dense
   traffic too (curriculum so far only up to 0.3 /s/arm = ~72 cars/min; above ~80/min the brain breaks
   down - out of its training range): demand range up to 0.5 /s/arm (= the new slider max, 120/min),
   compare with an arm up to 1.0 if cheap; keep sampling lower demands so normal traffic isn't forgotten.
   Second variable (owner, 2026-10-21: still too many crashes): crash/off-road cost x3 vs x6 with the
   set brain (round 1 showed x10 worse, x1 = x3, but without obstacles/attention). Measure crashes,
   throughput AND deadlocks/timeouts: a higher cost may trade crashes for gridlock. Goal: stay on the plateau above capacity instead of collapsing,
   then raise capacity towards the reservation's via emergent coordination (alternating streams,
   platoons, lamp negotiation).
4b. [2026-10-22] Status: round 6 (equal updates, dense traffic + obstacles 0.75) gave the main brain
   r6_DO_s2@750 (95 vs 61 cars/min at 120/min, half the crashes, deadlock time 50 -> 12 %). Open: the
   stop skill erodes during training (stop rate 0.71 -> 0.44; safe rate only just 0.955) - the stop gate
   still acts as a filter. Next: obstacle scenarios that cannot be passed around (whole carriageway
   blocked, e.g. two wrecks side by side or a wreck plus a standing queue), so only stopping avoids the
   crash - outcome-only, no style cost (NOT a penalty for passing close to obstacles). Then longer DO
   runs; then GRU memory and a shared reward component (owner decision) against remaining deadlocks.
4c. Curriculum rule (owner OK, 2026-10-24): the training demand ceiling grows with the brain. When the main
   brain is stuck (deadlock) less than ~5 % of the time at the current ceiling (`cli deadlock`, report
   seeds), raise the ceiling in the next round (keep sampling low demands). Not before: above capacity
   (~96/min) more demand mostly means longer queues, more training crashes, slower runs and the n_slots
   limit. Today (ceiling 0.5 /s/arm = 120/min) the brain is still stuck 12-23 % of the time there.
   Practice obstacles stay part of the normal training world (no separate obstacle training): deadlocks
   do not replace them (slow roll-in inside the box, they get rarer as the brain improves; the round-7
   control with dense traffic but no blockades lost block stop 1.00 -> 0.75). Open check (optional):
   a run with no artificial obstacles at all, real 10-s wrecks only.
5. Then (D) norm experiments (lamp language, priority patterns) - (G) feeds directly into them.

- (G) Deadlocks and decentralised negotiation (owner, 2026-10-14; the interesting part from a game-theory
  view: can individual agents find a convention / negotiation without any central instance?).
  Framing: a coordination game like "chicken" (go/yield; the problem is who yields). A traffic light is
  a correlation device; agents could create their own: (1) symmetry breaking - all cars share one brain
  and act deterministically in evaluation, so symmetric situations give symmetric actions (all wait /
  all go); randomness as in Ethernet backoff; (2) conventions from observable asymmetries (who was
  first, who comes from the right, who already stands); (3) communication via the free lamp ("I go" /
  "I have waited long"); (4) memory (GRU) to know waiting time / arrival order; (5) incentives: today a
  deadlock is cheap (removed after max_age 60 s: -1 plus waiting time vs crash from -4.5), and yielding
  mainly helps others (individual reward; shared brain gives only weak cooperation).
  Plan: measure first (deadlock frequency, duration, constellation; does a little action noise in
  evaluation already dissolve them? does the lamp play a role?), then single-variable experiments:
  deadlock cost (max_age handling / timeout cost, outcome-based), GRU memory, small shared reward
  component (throughput of all; owner decision), lamp analysis. Literature: Pal et al. 2022.

## Roadmap (owner, after the wide-crossing change)
- (A) Driving quality: failure breakdown (offroad vs crash, solo vs traffic, per manoeuvre and
  demand), solo-driving gate (a car alone never leaves the road), impact-speed crash/offroad
  cost, physically visible neighbour features (curvature, acceleration).  <- done
- (B) Perception / network: set (attention) encoder over all neighbours instead of K nearest
  <- done (round 4, clear win); optional recurrent memory (GRU) <- open, see (G).
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
- (F) Geometry variety (long term; owner 2026-10-24): after the two open core problems (stop skill,
  deadlocks) are solved. Order: (1) a non-right-angled crossing (closest to today's setup), (2) a motorway
  on-ramp / merge (Autobahnzufahrt: merging into a flowing stream, gap acceptance - owner's wish), (3) a
  roundabout, (4) later a whole village (road network; cars start/end at driveways or the map edge). Needs
  geometry-independent inputs (C), a road-graph representation instead of the hard-coded 4-arm crossing,
  and route information (next turn / merge) as input instead of only the exit arm; likely retraining from
  scratch per geometry family.
- Run tracking (owner, 2026-10-11), with the next refactoring step: git commit hash (+ dirty flag)
  in every run's args.json, full state.npz of important runs kept under results/ (cloud runs were
  lost), a small script listing runs/ (name, commit, settings, key metrics). The owner needs no
  tracking UI (the /lernen curve is enough); tooling is for the coordinator and agents, their call.
- Site bench over several traffic seeds (mean) instead of one (BENCH_SEED): single-seed numbers are
  noisy (round 2 best at 0.15: 12 on the site's seed vs 31 averaged over 3 seeds).
- Quick wins: small GIF of the crossing at the top of README.md (rendered from the main brain);
  short, plain explanation on /lernen of how training with many cars works (PPO: all cars share one
  brain, each learns from its own reward; ES: each mutation drives in its own environments, all
  cars there use it, fitness = sum over all cars).

## Decided for round 1 (owner; all implemented 2026-10-11)
- Spawn on the right half of the road in driving direction, always (`spawn_right` for the learners
  too; owner, 2026-10-11): fewer crashes / space problems at the map edge. This gives up the
  "does a driving side emerge on its own" question (earlier runs already showed the cars agree on a
  side); the other norm questions in (D) remain.
- Wrecks: crashed cars stay for 10 s (owner change 2026-10-24; was 5 s until round 7) as standing obstacles (driving into one = own crash) with hazard
  lights that other cars can see (a physically visible signal, no rule information), then vanish.
  Mirror in sim.js / site rendering.
- Higher fixed crash/off-road cost, see (E).
- One free lamp channel instead of two (n_signal = 1): easier to read and to analyse.
- Neighbour acceleration observation becomes a binary brake light (on when braking), as on real cars.
- No turn signals: a built-in blinker would leak the destination, which is deliberately hidden so
  that a lamp "language" has something to say. The free lamp is the place where one could emerge.
- Brake lights are drawn on the website now (visual only).
