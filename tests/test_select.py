import json
import shutil
from pathlib import Path

from smartcrossing import select as sel
from smartcrossing.cli import main

BEST = Path(__file__).resolve().parent.parent / "results" / "round4_best.npz"


def row(it, gate, t5, c3):
    return {"ckpt": f"r/ckpt_{it:04d}.npz", "iter": it, "safe_rate_feasible": gate, "thr": {"0.3": 50, "0.5": t5},
            "crash": {"0.3": c3}}


def test_pick_rule():
    rows = [row(0, 1.0, 50, 10), row(1, 0.9, 80, 10), row(2, 0.96, 60, 10), row(3, 0.97, 70, 40), row(4, 0.99, 55, 11)]
    s = sel.pick(rows, rows[0])
    assert s["selected"]["iter"] == 2  # 1 fails the gate, 3 the crash limit (10 * 1.1 + 1 = 12)
    assert s["unconstrained"]["iter"] == 1
    assert sel.pick(rows, rows[0], gate_min=1.01)["selected"] is None


def test_checkpoints_every(tmp_path):
    for i in (0, 25, 50, 75, 99):
        (tmp_path / f"ckpt_{i:04d}.npz").touch()
    assert [c.stem for c in sel.checkpoints(tmp_path, 2)] == ["ckpt_0000", "ckpt_0050", "ckpt_0099"]


def test_select_end_to_end(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for i in (0, 25, 50):
        shutil.copyfile(BEST, run / f"ckpt_{i:04d}.npz")
    out, cp = tmp_path / "s.jsonl", tmp_path / "best.npz"
    main(["select", "--run", str(run), "--demands", "0.15,0.3,0.5", "--seeds", "1", "--envs", "1", "--seconds", "4",
          "--gate-min", "0.5", "--out", str(out), "--copy-to", str(cp)])
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert sum(x["kind"] == "row" for x in lines) == 3 and cp.exists()
    assert next(x for x in lines if x["kind"] == "selection")["selected"]["iter"] in (0, 25, 50)
