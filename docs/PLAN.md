# Smart Crossing — Plan & Decisions

Goal: a signal-free four-way intersection crossed only by learning autonomous cars.
Questions: (1) how high can throughput get without crashes, (2) does coordination emerge
when cars can emit a free, un-predefined "light" signal, (3) do conventions such as
right- vs left-hand traffic emerge on their own, (4) what happens when differently
trained agent types share the road (ncase.me "Evolution of Trust" style).

## Fixed decisions (from the owner)
- No lanes, lane paths, traffic lights, priority rules or driving-side conventions for the
  learning agents. Steering is learned (2D kinematics), not just speed.
- Old pygame/DQN code was replaced (moved to `legacy/`, kept for reference only).
- Geometry: 4 arms, road 7 m wide (room for one car per direction), rounded curbs.
- Deliverable: a live, in-browser visualisation with a few sliders (keep it simple),
  incl. mixing agent types.

## Architecture
- `smartcrossing/sim.py`: batched numpy simulator, state arrays (B envs, N slots).
  Kinematic unicycle with curvature limit, dt=0.1 s. Cars spawn at a random lateral
  offset (so no side is preferred), must exit through target arm. Car-car collision via
  3 discs per car, off-road via rounded-corner distance. Removal on crash/offroad/exit/timeout.
  Observations are mirror-neutral: own state in a canonical frame (entry arm = south),
  goal vector, 5 edge rays, K=4 nearest neighbours (ego frame) incl. their signal but
  NOT their destination. Action = (accel, curvature, signal[n_signal]) in [-1,1].
  Common random numbers: env b uses traffic stream b % n_scenarios.
- `smartcrossing/baselines.py`: rule-based references with lanes (light / reservation(AIM) / none).
- `smartcrossing/render.py`, `metrics.py`, `cli.py` (bench / render).
- Planned: `policy.py` (shared MLP, numpy), `es.py` (OpenAI-ES, multiprocessing), `viewer/`
  (JS port of sim + policy, sliders, checked against Python via node test vectors).

## Self-evaluation workflow
- `pytest` invariants; `cli bench` prints one JSON line (throughput, crashes/1000, delay,
  Jain fairness, ...); `cli render` writes `sheet.png` (9-frame contact sheet) + `episode.gif`.
- Every experiment gets an entry in `docs/experiments.md` (hypothesis, config, result, commit).
- Subagents look at images themselves and report numbers; orchestrator keeps context small.
- Emergence claims require ablations: signal zeroed/shuffled, no-comm training, MI(signal; intent).

## Phases
1. Simulator, baselines, metrics, rendering, tests  ← in progress
2. ES training without/with signal; curriculum over demand
3. Emergence analysis (driving side, signal meaning), agent-type mixing
4. Browser viewer with sliders (demand, turn mix, signal noise/on-off, population mix)
