/* Measurement window of the live dot on the start page's fundamental diagram. It uses the definition of the scatter
 * cloud behind the curve (smartcrossing/fundamental_cloud.py cloud_windows, results/fundamental_cloud.json), so the dot and
 * the cloud are comparable: a WINDOW_S-second window lying entirely after WARMUP_S seconds from the empty start;
 * density = mean number of driving cars on the map over the window, throughput = exits in the window per minute.
 * Checked against the Python definition by tests/test_site_js.py (node). Browser: window.SCX.fdWindow. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else (root.SCX = root.SCX || {}).fdWindow = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const WARMUP_S = 30, WINDOW_S = 60, EPS = 1e-6;

  /** Sim time at which the first full window is available. */
  const firstAt = () => WARMUP_S + WINDOW_S;

  /**
   * The point at sim time t, or null while no full window after the warm-up exists.
   * samples: [[time, cars on the map], ...] taken before the step at that time (any spacing; the Python cloud
   *   samples every step), exitTimes: sim times of the exits (after their step), both in any order.
   * Window: samples with time in [t - WINDOW_S, t), exits with time in (t - WINDOW_S, t].
   */
  function point(samples, exitTimes, t) {
    const lo = t - WINDOW_S;
    if (lo < WARMUP_S - EPS) return null;
    let sum = 0, n = 0, k = 0;
    for (const [ts, cars] of samples) if (ts >= lo - EPS && ts < t - EPS) { sum += cars; n += 1; }
    if (!n) return null;
    for (const te of exitTimes) if (te > lo + EPS && te <= t + EPS) k += 1;
    return { cars: sum / n, thr: (k * 60) / WINDOW_S };
  }

  return { WARMUP_S, WINDOW_S, firstAt, point };
});
