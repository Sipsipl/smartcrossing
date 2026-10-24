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
    assert len(rows) == 108 + len(gates.block_cases()) and summary["block_cases"] == len(gates.block_cases()) == 36
    for r in rows:
        assert r["outcome"] == ("stopped" if r["feasible"] else "crash"), r
    assert summary["stop_rate_feasible"] == 1.0 and 0 < summary["feasible"] < summary["cases"]
    assert summary["block_stop_rate_feasible"] == 1.0 and 0 < summary["block_feasible"] < summary["block_cases"]


class Straight:
    """Keeps its speed and lane: nothing can be passed in a blockade, every block case must crash."""

    def __init__(self, sim):
        self.sim = sim

    def act(self):
        return np.zeros((self.sim.B, self.sim.cfg.n_slots, act_dim(self.sim.cfg)))


class Swerve:
    """Swerves hard left (towards the oncoming half) at full speed: a single obstacle can be passed, a
    blockade cannot."""

    def __init__(self, sim):
        self.sim = sim

    def act(self):
        a = np.zeros((self.sim.B, self.sim.cfg.n_slots, act_dim(self.sim.cfg)))
        a[..., 1] = np.where(self.sim.x > -2.0, 0.6, -0.6)  # left until 2 m left of the centre
        return a


def test_block_cases_cannot_be_passed(monkeypatch):
    from smartcrossing import cli

    for drv in (Straight, Swerve):
        monkeypatch.setattr(cli, "make_driver", lambda spec, sim, drv=drv: drv(sim))
        summary, rows = gates.stop_gate(drv.__name__)
        block = [r for r in rows if r["pos"] == "block"]
        assert block and all(r["outcome"] in ("crash", "offroad") for r in block), drv
        assert summary["block_stop_rate"] == 0.0 and summary["block_outcomes"]["passed"] == 0


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
    assert summary["cases"] == 108 and len(rows) == 108 + 36 and summary["stop_rate_feasible"] == 1.0
    cli.main(["gate", "--driver", "brake"])  # without --out: no file, still the summary
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["cases"] == 108
