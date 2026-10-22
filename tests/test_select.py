import json
import shutil
from pathlib import Path

import pytest

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


def _run(tmp_path, n):
    run = tmp_path / "run"
    run.mkdir()
    for i in range(n):
        shutil.copyfile(BEST, run / f"ckpt_{i * 25:04d}.npz")
    return run


TINY = ["--seeds", "1", "--envs", "1", "--seconds", "2"]


def test_select_needs_demands_0_3_and_0_5(tmp_path):
    run = _run(tmp_path, 1)
    with pytest.raises(SystemExit, match="0.3 and 0.5"):
        main(["select", "--run", str(run), "--demands", "0.15,0.3", *TINY])
    with pytest.raises(SystemExit, match="no ckpt"):
        main(["select", "--run", str(tmp_path / "empty"), *TINY])


def test_select_single_checkpoint_without_0_15_and_empty_selection(tmp_path):
    """One checkpoint, no 0.15 in the demands (table shows '-'), a gate nobody passes: no selection, nothing copied."""
    run = _run(tmp_path, 1)
    out, cp = tmp_path / "s.jsonl", tmp_path / "best.npz"
    res = main(["select", "--run", str(run), "--demands", "0.3,0.5", "--gate-min", "1.01", "--out", str(out),
                "--copy-to", str(cp), *TINY])
    assert res is None and not cp.exists()
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [x["kind"] for x in lines] == ["row", "selection", "overall"]
    assert lines[1]["selected"] is None and lines[1]["unconstrained"]["iter"] == 0 and lines[2]["selected"] is None
