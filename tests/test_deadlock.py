"""Deadlock study (deadlock.py): detector on synthetic recordings, noisy driver, cli line format, fundamental."""

import json
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import cli, deadlock
from smartcrossing.sim import DONE_TIMEOUT, Sim, SimConfig

ROOT = Path(__file__).resolve().parents[1]
ROUND4 = ROOT / "results" / "round4_best.npz"


def _rec(T, pos, moving=(), timeout_at=None):
    """Synthetic recording of one env: car n stands at pos[n] (rocking +-0.3 m) unless n in moving
    (then it drives at 5 m/s along x); car 0 is removed at max_age in step timeout_at."""
    N = len(pos)
    rec = {k: np.zeros((T, 1, N), np.float32) for k in ("x", "y", "th", "v", "age", "lamp")}
    rec.update(alive=np.ones((T, 1, N), bool), wreck=np.zeros((T, 1, N), bool),
               uid=np.tile(np.arange(N), (T, 1, 1)), arm=np.zeros((T, 1, N), np.int8),
               man=np.full((T, 1, N), 2, np.int8), done=np.zeros((T, 1, N), np.int8), queue=np.zeros((T, 1), np.int32))
    t = np.arange(T) * 0.1
    for n, (x, y, th) in enumerate(pos):
        rec["x"][:, 0, n] = x + (5 * t if n in moving else 0.3 * np.sin(t))
        rec["y"][:, 0, n] = y
        rec["th"][:, 0, n] = th
        rec["v"][:, 0, n] = 5.0 if n in moving else 0.3 * np.cos(t)
        rec["arm"][:, 0, n] = n % 4
    if timeout_at is not None:
        rec["done"][timeout_at, 0, 0] = DONE_TIMEOUT
        rec["alive"][timeout_at + 1:, 0, 0] = False
        rec["uid"][timeout_at + 1:, 0, 0] = -1
        rec["x"][timeout_at + 1:, 0, 0] = 1e4
    return rec


# three cars face each other inside the conflict area (a 3-cycle: each one's nearest car ahead is the next)
CYCLE = [(-3.0, -3.0, 0.0), (3.0, -3.0, np.pi / 2), (0.0, 2.0, -2.4)]


def test_detects_a_standing_cluster_and_its_end():
    cfg = SimConfig()
    rec = _rec(200, CYCLE, timeout_at=150)
    eps = deadlock.detect(rec, cfg)
    assert len(eps) == 1
    e = eps[0]
    assert e["onset"] == 50 and e["start"] == 0  # stuck after the 5 s window, back-dated
    assert e["cause"] == "timeout" and e["n_timeouts"] == 1 and not e["censored"]
    assert e["geo"]["n_inner"] == 3 and e["geo"]["kind"] == "cycle" and e["geo"]["cycle_len"] == 3
    assert e["n_cars"] == 3 and e["zone_exits"] == 0


def test_no_deadlock_when_moving_or_outside_the_conflict_area():
    cfg = SimConfig()
    assert deadlock.detect(_rec(200, CYCLE, moving=(1,)), cfg) == []  # only two stuck
    mouth = [(-3.0, -3.0, 0.0), (2.0, -14.0, np.pi / 2), (5.0, -14.0, np.pi / 2)]  # one inner, two at the mouth
    assert deadlock.detect(_rec(200, mouth), cfg) == []
    # the same cluster runs to the end of the recording: censored
    assert [e["cause"] for e in deadlock.detect(_rec(120, CYCLE), cfg)] == ["censored"]
    assert deadlock.detect(_rec(120, CYCLE), cfg, n_min=4) == []


def test_noisy_driver_matches_the_deterministic_policy_at_zero_noise():
    sims = [Sim(SimConfig(demand=0.3), 2, seed=5) for _ in range(3)]
    det = cli.make_driver(f"policy:{ROUND4}", sims[0])
    zero = deadlock.make_driver(f"policy:{ROUND4}", sims[1], noise=1e-12)
    noisy = deadlock.make_driver(f"policy:{ROUND4}", sims[2], noise="policy", seed=1)
    for _ in range(30):
        m = sims[0].alive.copy()
        acts = [det.act(), zero.act(), noisy.act()]
        for s, a in zip(sims, acts):
            s.step(a)
    assert np.allclose(acts[0][m], acts[1][m], atol=1e-9)
    assert not np.allclose(acts[0][m], acts[2][m])
    assert np.allclose(np.exp(deadlock.policy_logstd(ROUND4)), np.exp([-0.763, -0.774, -0.717]), atol=1e-3)
    with pytest.raises(ValueError):
        deadlock.make_driver("light", Sim(SimConfig(), 1), noise=0.1)


def test_cli_deadlock_line_aggregate_and_fundamental(tmp_path):
    out = tmp_path / "d.jsonl"
    cli.main(["deadlock", "--driver", "light", "--demands", "0.3", "--seeds", "1", "--envs", "2", "--seconds", "40",
              "--out", str(out)])
    cli.main(["deadlock", "--driver", f"policy:{ROUND4}", "--demands", "0.3", "--seeds", "1", "--envs", "2",
              "--seconds", "40", "--noise", "0.1", "--lamp", "zero", "--out", str(out)])
    lines = [json.loads(s) for s in out.read_text(encoding="utf-8").splitlines()]
    assert [(ln["noise"], ln["lamp"]) for ln in lines] == [("none", "normal"), ("0.1", "zero")]
    for k in ("throughput_steady_per_min", "cars_map_mean", "density_veh_per_km", "deadlocks_per_hour",
              "queue_end_per_env", "episodes", "sensitivity", "jam_time_frac"):
        assert k in lines[0]
    rows = deadlock.aggregate(lines)
    assert {r["driver"] for r in rows} == {"light", "round4_best"}
    assert "| light |" in deadlock.table(rows)
    f = deadlock.fundamental(lines, {"round4_best": "brain"})
    assert list(f["drivers"]) == ["light"]  # noisy / lamp-intervention runs are not part of it
    pt = f["drivers"]["light"]["points"][0]
    assert pt["demand"] == 0.3 and pt["density"] == lines[0]["cars_map_mean"] and pt["throughput"] > 0


def test_incentives_order():
    cfg = json.loads(str(np.load(ROUND4)["cfg"]))
    inc = deadlock.incentives(cfg)
    o = inc["options"]
    # waiting to the max_age removal costs less than any crash; discounting makes waiting almost free
    assert o["wait_to_timeout_age40"] > o["go_crash_2mps"] > o["go_crash_10mps"]
    assert o["go_exit_4s"] > o["wait_to_timeout_age20"]
    assert o["wait_to_timeout_age20_disc"] > o["wait_to_timeout_age20"]
    assert 0 < inc["break_even_p_crash"] < 1 and inc["horizon_s"] == 10.0


def test_cloud_windows_and_json_shape():
    """Cloud cell: 60-s sliding windows every 10 s per env, only after the 30 s warm-up; [density, throughput, demand]."""
    pts = deadlock.cloud_job(("light", None, 0.1, 7, 2, 110.0))
    assert len(pts) == 2 * 3  # windows ending at 90, 100, 110 s, 2 envs
    assert all(len(p) == 3 and p[2] == 0.1 and p[0] > 0 and p[1] >= 0 for p in pts)
    out = deadlock.cloud({"light": "light"}, demands=(0.1,), envs=2, seconds=110.0, per_demand=4, log=lambda *_: None)
    assert len(out["drivers"]["light"]["points"]) == 4  # thinned
    assert {"density", "throughput", "points", "protocol"} <= set(out["meta"])


def test_cloud_window_formula():
    """cloud_windows on synthetic counts: 4 cars all the time, one exit every 2 s -> density 4, throughput 30/min;
    the first window ends at warm-up + window (90 s), then one every 10 s."""
    dt, T = 0.1, 1200
    cars = np.full((T, 2), 4.0)
    cars[:300, 1] = 100.0  # the warm-up is never part of a window
    exits = np.repeat((np.arange(T + 1) // 20)[:, None], 2, 1).astype(float)
    win = deadlock.cloud_windows(cars, exits, dt)
    assert [round(t, 6) for t, *_ in win] == [90.0, 100.0, 110.0, 120.0]
    for _, dens, thr in win:
        assert np.allclose(dens, 4.0) and np.allclose(thr, 30.0)
    assert deadlock.cloud_windows(cars[:899], exits[:900], dt) == []  # no full window after the warm-up yet


def test_cli_deadlock_summarize_names_and_meta(tmp_path, capsys):
    out = tmp_path / "d.jsonl"
    cli.main(["deadlock", "--driver", "light", "--demands", "0.3", "--seeds", "1,2", "--envs", "1", "--seconds", "35",
              "--out", str(out)])
    fund, rows = tmp_path / "f.json", tmp_path / "rows.json"
    cli.main(["deadlock-summarize", str(out), "--name", "light=ampel", "--fundamental", str(fund), "--json", str(rows)])
    assert "| light |" in capsys.readouterr().out
    f = json.loads(fund.read_text(encoding="utf-8"))
    assert list(f["drivers"]) == ["ampel"] and "per_seed" not in f["drivers"]["ampel"]
    assert f["meta"]["protocol"].startswith("envs [1] x seconds [35.0], traffic seeds [1, 2]")
    assert json.loads(rows.read_text(encoding="utf-8"))[0]["n_seeds"] == 2
    with pytest.raises(SystemExit):
        cli.main(["deadlock-summarize", str(out), "--name", "light"])  # not spec=key


def test_cli_incentives(capsys):
    cli.main(["incentives", "--driver", f"policy:{ROUND4}"])
    assert json.loads(capsys.readouterr().out) == deadlock.incentives(json.loads(str(np.load(ROUND4)["cfg"])))
