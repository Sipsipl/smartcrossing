"""Stop gate (gates.py): feasibility vs a full-braking driver, a real policy, and the cli gate command."""

import json
from pathlib import Path

import numpy as np

from smartcrossing import gates
from smartcrossing.sim import act_dim

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = ROOT / "results" / "round1_x3.npz"


class Brake:
    """Full braking, straight on: the reference for the stop gate's feasibility."""

    def __init__(self, sim):
        self.sim = sim

    def act(self):
        a = np.zeros((self.sim.B, self.sim.cfg.n_slots, act_dim(self.sim.cfg)))
        a[..., 0] = -1.0
        return a


def test_stop_gate_with_a_braking_driver(monkeypatch):
    from smartcrossing import cli

    monkeypatch.setattr(cli, "make_driver", lambda spec, sim: Brake(sim))
    summary, rows = gates.stop_gate("brake")
    assert summary["cases"] == len(gates.cases()) == 108
    for r in rows:
        assert r["outcome"] == ("stopped" if r["feasible"] else "crash"), r
    assert summary["stop_rate_feasible"] == 1.0 and 0 < summary["feasible"] < summary["cases"]


def test_stop_gate_policy_runs_fast():
    s, _ = gates.stop_gate(f"policy:{ROUND1}")
    assert sum(s[o] for o in gates.OUTCOMES) == s["cases"] and s["wall_s"] < 30


def test_cli_gate_writes_cases_json(tmp_path, monkeypatch, capsys):
    """cli gate prints the summary and writes the per-case rows to --out (any path)."""
    from smartcrossing import cli

    monkeypatch.setattr(cli, "make_driver", lambda spec, sim: Brake(sim))
    out = tmp_path / "cases.json"
    cli.main(["gate", "--driver", "brake", "--out", str(out)])
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert summary["cases"] == len(rows) == 108 and summary["stop_rate_feasible"] == 1.0
    cli.main(["gate", "--driver", "brake"])  # without --out: no file, still the summary
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["cases"] == 108
