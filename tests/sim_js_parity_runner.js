// Node driver for tests/test_sim_js_parity.py.
// Usage: node sim_js_parity_runner.js cases.json sim-config.js  -> prints results JSON on stdout.
// sim-config.js is the file site/build.py writes (loaded first, as on the site).
"use strict";
const fs = require("fs");
const path = require("path");
require(path.resolve(process.argv[3]));
const SC = require(path.join(__dirname, "..", "site", "src", "js", "sim.js"));

const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const exported = {};
for (const k of ["RIGHT", "STRAIGHT", "LEFT", "DISC_R", "K_NEIGHBOURS", "NEIGHBOUR_RANGE", "OBS_DIM", "NB_FEATURES",
  "NB_FEATURES_LEGACY", "BRAKE_ACC", "POS_SCALE", "GOAL_SCALE", "RAY_RANGE", "DONE"]) exported[k] = SC[k];
const out = { defaults: SC.defaultConfig(), config: globalThis.SC_SIM_CONFIG, exported, cases: [], policies: [] };

for (const c of input.cases) {
  const sim = new SC.Sim(c.cfg, 0);
  sim.setState(c.state);
  const obs = sim.observe();
  const reward = sim.step(Float64Array.from(c.action.flat()));
  const D = SC.obsDim(sim.cfg);
  const rows = [];
  for (let n = 0; n < sim.N; n++) rows.push(Array.from(obs.subarray(n * D, (n + 1) * D)));
  out.cases.push({
    name: c.name,
    obs: rows,
    reward,
    car_reward: Array.from(sim.carReward),
    done_kind: Array.from(sim.doneKind),
    done_uid: Array.from(sim.doneUid),
    queue_cost: sim.queueCost,
    state: sim.getState(),
    events: sim.events.filter((e) => e.kind === "offroad" || e.kind === "crash").map((e) => [e.kind, e.x, e.y]),
  });
}

for (const p of input.policies || []) {
  const pol = SC.loadPolicy(p.policy);
  const D = p.obs[0].length;
  const flat = Float64Array.from(p.obs.flat());
  const rows = [];
  for (let n = 0; n < p.obs.length; n++) rows.push(Array.from(SC.forward(pol, flat, n * D)));
  let acted = null;
  if (p.driver) {
    const sim = new SC.Sim(p.driver.cfg, 0);
    sim.setState(p.driver.state);
    const pols = p.driver.policies.map(SC.loadPolicy);
    const A = SC.actDim(sim.cfg);
    const a = SC.act(sim, pols, p.driver.mute);
    acted = [];
    for (let n = 0; n < sim.N; n++) acted.push(Array.from(a.subarray(n * A, (n + 1) * A)));
  }
  out.policies.push({ out: rows, acted });
}

process.stdout.write(JSON.stringify(out));
