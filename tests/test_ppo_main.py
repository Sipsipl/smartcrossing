"""Smoke test of the whole PPO training entry point (ppo.main) on the CPU with tiny sizes:
fresh run, resume (--init, --critic-warmup), warm start (--init-policy), the torch rollout path,
run tracking (args.json, checkpoint extra, scripts/runs_index.py) and the eval job's world."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import ppo
from smartcrossing.policy import load_npz

ROOT = Path(__file__).resolve().parents[1]
TINY = ["--workers", "1", "--envs", "2", "--horizon", "16", "--hidden", "8,8", "--minibatches", "2", "--epochs", "1",
        "--eval-every", "0", "--cfg", json.dumps({"n_slots": 8, "wreck_time": 1.0}), "--demand-start", "0.3"]


def _log(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "log.jsonl").read_text(encoding="utf-8").splitlines()]


def _runs_index():
    spec = importlib.util.spec_from_file_location("runs_index", ROOT / "scripts" / "runs_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ppo_main_fresh_resume_and_index(tmp_path):
    runs = tmp_path / "runs"
    ppo.main(["--name", "smoke", "--iters", "8", "--runs-dir", str(runs), *TINY])
    run = runs / "smoke"
    args = json.loads((run / "args.json").read_text(encoding="utf-8"))
    assert set(args["git"]) == {"commit", "dirty"} and args["cfg"]["n_slots"] == 8 and args["init"] == ""
    _, sizes, cfg = load_npz(run / "current.npz")
    with np.load(run / "current.npz") as z:
        extra = json.loads(str(z["extra"]))
    assert extra["iter"] == 7 and "commit" in extra and "dirty" in extra and cfg["wreck_time"] == 1.0
    # resume: continues the iteration count, keeps args.json, writes args.1.json; critic warm-up keeps the actor
    actor_before = np.load(run / "state.npz")["actor"].copy()
    ppo.main(["--name", "smoke", "--iters", "12", "--runs-dir", str(runs), "--init", str(run / "state.npz"),
              "--critic-warmup", "4", *TINY])
    assert json.loads((run / "args.json").read_text(encoding="utf-8"))["init"] == ""
    resumed = json.loads((run / "args.1.json").read_text(encoding="utf-8"))
    assert resumed["init"].endswith("state.npz")
    log = _log(run)
    assert [r["iter"] for r in log] == list(range(12))
    assert all(r.get("critic_warmup") for r in log[8:12])
    np.testing.assert_array_equal(np.load(run / "state.npz")["actor"], actor_before)
    assert log[-1]["total_steps"] == sum(r["car_steps"] for r in log)
    # the runs index lists the run and skips folders that are not runs
    (runs / "junk").mkdir()
    (runs / "notes.txt").write_text("x", encoding="utf-8")
    rows = _runs_index().index(runs)
    assert [r["name"] for r in rows] == ["smoke"]
    assert "it=11" in rows[0]["log"] and "cfg.n_slots=8" in rows[0]["settings"]


def test_ppo_main_init_policy(tmp_path):
    """Warm start from a checkpoint (transplanted actor), a few iterations."""
    old = ROOT / "results" / "wide_s1ft2_ckpt2650.npz"
    tiny = [x if x != "8,8" else "64,64" for x in TINY]  # the transplant keeps the hidden sizes
    ppo.main(["--name", "warm", "--iters", "2", "--runs-dir", str(tmp_path), "--init-policy", str(old), *tiny])
    args = json.loads((tmp_path / "warm" / "args.json").read_text(encoding="utf-8"))
    assert args["init_policy"] == str(old) and len(_log(tmp_path / "warm")) == 2


def test_ppo_main_torch_rollout_cpu(tmp_path):
    pytest.importorskip("torch")
    ppo.main(["--name", "gpu_path", "--iters", "2", "--runs-dir", str(tmp_path), "--rollout-device", "cpu",
              "--rollout-envs", "3", *TINY])
    log = _log(tmp_path / "gpu_path")
    assert len(log) == 2 and log[-1]["car_steps"] > 0


def test_eval_job_uses_the_run_world(tmp_path, monkeypatch):
    """eval_job benches in the run's world (n_slots, wreck_time, spawn_right, ...) at fixed demands."""
    from smartcrossing import cli
    from smartcrossing.policy import layer_sizes, save_npz
    from smartcrossing.sim import SimConfig

    cfg = SimConfig(n_slots=8, wreck_time=1.0, spawn_right=False, demand_range=(0.1, 0.3))
    sizes = layer_sizes(cfg, (8,))
    save_npz(tmp_path / "p.npz", np.zeros(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))), sizes, cfg)
    seen = []
    real_run = cli.run

    def spy(spec, c, *args, **kw):
        seen.append(c)
        return real_run(spec, c, *args, **kw)

    monkeypatch.setattr(cli, "run", spy)
    ppo.eval_job(str(tmp_path / "p.npz"), str(tmp_path / "eval.jsonl"), 0, (0.05, 0.15), 1, 2.0, None, dict(cfg.__dict__))
    assert [c.demand for c in seen] == [0.05, 0.15]
    assert all(c.n_slots == 8 and c.wreck_time == 1.0 and not c.spawn_right and c.demand_range is None for c in seen)
    rec = json.loads((tmp_path / "eval.jsonl").read_text(encoding="utf-8"))
    assert set(rec) == {"iter", "ckpt", "d0.05", "d0.15", "gate"} and 0 <= rec["gate"]["stop_rate"] <= 1
