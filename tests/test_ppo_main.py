"""Smoke test of the whole PPO training entry point (ppo.main) on the CPU with tiny sizes:
fresh run, resume (--init, --critic-warmup), warm start (--init-policy, --logstd-init), the torch rollout
path, --arch set, --max-minutes with --eval-every 0, run tracking (git_info, write_args / args.json, checkpoint
extra, scripts/runs_index.py) and the eval job's world."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import ppo
from smartcrossing.policy import PolicyDriver, load_arch, load_npz
from smartcrossing.ppo_common import N_PRIV
from smartcrossing.sim import Sim, SimConfig

ROOT = Path(__file__).resolve().parents[1]
ROUND1 = ROOT / "results" / "round1_x3.npz"
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
    assert not any("duty" in r or "t_sleep" in r for r in log)  # no background mode: log unchanged
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


def test_background_mode_duty(tmp_path, monkeypatch):
    """--gpu-duty pauses t_iter * (1/F - 1) after each iteration; runs/<name>/duty.txt overrides it while running;
    --eval-workers 0 skips the background evals but the final checkpoint is still evaluated."""
    assert ppo.duty_sleep(2.0, 0.5) == 2.0 and ppo.duty_sleep(2.0, 1.0) == 0.0 and ppo.duty_sleep(1.0, 0.25) == 3.0
    assert ppo.read_duty(tmp_path / "missing.txt", 0.7) == 0.7
    (tmp_path / "d.txt").write_text(" 0.4 ", encoding="utf-8")
    assert ppo.read_duty(tmp_path / "d.txt", 1.0) == 0.4
    (tmp_path / "d.txt").write_text("junk", encoding="utf-8")
    assert ppo.read_duty(tmp_path / "d.txt", 0.9) == 0.9
    slept = []
    monkeypatch.setattr(ppo.time, "sleep", slept.append)
    jobs = []
    real_ctx = ppo.mp.get_context("spawn")

    class Ctx:  # eval processes are only recorded; rollout workers run for real
        def Process(self, target, args, **kw):
            if target is ppo.eval_job:
                jobs.append(args[2])
                return type("P", (), {"start": lambda s: None, "join": lambda s, timeout=None: None,
                                      "is_alive": lambda s: False})()
            return real_ctx.Process(target=target, args=args, **kw)

        def Pipe(self, *a, **kw):
            return real_ctx.Pipe(*a, **kw)

    monkeypatch.setattr(ppo.mp, "get_context", lambda method: Ctx())
    ppo.main(["--name", "bg", "--iters", "3", "--runs-dir", str(tmp_path), "--gpu-duty", "0.5", "--eval-workers", "0",
              *TINY, "--eval-every", "1"])
    log = _log(tmp_path / "bg")
    assert [r["duty"] for r in log] == [0.5] * 3 and len(slept) == 2  # (no pause after the final iteration)
    assert jobs == [2]  # --eval-workers 0: only the final checkpoint is evaluated
    (tmp_path / "bg" / "duty.txt").write_text("1", encoding="utf-8")
    ppo.main(["--name", "bg", "--iters", "5", "--runs-dir", str(tmp_path), "--gpu-duty", "0.5",
              "--init", str(tmp_path / "bg" / "state.npz"), *TINY])
    log = _log(tmp_path / "bg")
    assert [r["duty"] for r in log[3:]] == [1.0, 1.0] and all(r["t_sleep"] == 0 for r in log[3:]) and len(slept) == 2


def test_eval_job_uses_the_run_world(tmp_path, monkeypatch):
    """eval_job benches (evalset.bench_one) in the run's world (n_slots, wreck_time, spawn_right, ...) at fixed demands."""
    from smartcrossing import evalset
    from smartcrossing.policy import layer_sizes, save_npz

    cfg = SimConfig(n_slots=8, wreck_time=1.0, spawn_right=False, demand_range=(0.1, 0.3))
    sizes = layer_sizes(cfg, (8,))
    save_npz(tmp_path / "p.npz", np.zeros(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))), sizes, cfg)
    seen = []
    real = evalset.bench_one

    def spy(spec, c, *args, **kw):
        seen.append(c)
        return real(spec, c, *args, **kw)

    monkeypatch.setattr(evalset, "bench_one", spy)
    ppo.eval_job(str(tmp_path / "p.npz"), str(tmp_path / "eval.jsonl"), 0, (0.05, 0.15), 1, 2.0, None, dict(cfg.__dict__))
    assert [c.demand for c in seen] == [0.05, 0.15]
    assert all(c.n_slots == 8 and c.wreck_time == 1.0 and not c.spawn_right and c.demand_range is None for c in seen)
    rec = json.loads((tmp_path / "eval.jsonl").read_text(encoding="utf-8"))
    assert set(rec) == {"iter", "ckpt", "d0.05", "d0.15", "gate"} and 0 <= rec["gate"]["stop_rate"] <= 1
    assert set(rec["d0.05"]) == {*evalset.KEEP_KEYS, "queued_end"}


def test_git_info_and_write_args(tmp_path):
    """git_info: commit + dirty flag in the repo, None values outside a repo; write_args: args.json, and
    a resumed run (--init) keeps it and writes args.<n>.json."""
    from smartcrossing.ppo_common import git_info

    g = git_info()
    assert set(g) == {"commit", "dirty"} and len(g["commit"]) == 40 and isinstance(g["dirty"], bool)
    assert git_info(tmp_path) == {"commit": None, "dirty": None}
    a = ppo.parse_args(["--name", "x", "--hidden", "8,8"])
    st = ppo.build_state(a)
    p = ppo.write_args(tmp_path, a, st, g)
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert p.name == "args.json" and rec["git"] == g and rec["actor_sizes"] == st.a_sizes and rec["hidden"] == "8,8"
    a2 = ppo.parse_args(["--name", "x", "--init", "state.npz"])
    assert ppo.write_args(tmp_path, a2, st, g).name == "args.1.json"
    assert ppo.write_args(tmp_path, a2, st, g).name == "args.2.json"
    assert json.loads(p.read_text(encoding="utf-8"))["init"] == ""


def test_runs_index_cli(tmp_path, capsys):
    """scripts/runs_index.py lists a minimal run folder (args.json + log.jsonl) and skips the rest."""
    a = ppo.parse_args(["--name", "r1_t", "--lr", "1e-4"])
    run = tmp_path / "r1_t"
    run.mkdir()
    ppo.write_args(run, a, ppo.build_state(a), {"commit": "abc1234def", "dirty": False})
    (run / "log.jsonl").write_text(json.dumps({"iter": 3, "train": {"exit_ok": 0.9, "crash": 0.01}}) + "\n",
                                   encoding="utf-8")
    (tmp_path / "other").mkdir()
    mod = _runs_index()
    rows = mod.index(tmp_path)
    assert [r["name"] for r in rows] == ["r1_t"] and "lr=0.0001" in rows[0]["settings"]
    mod.main(["--runs", str(tmp_path), "--json"])
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["name"] for r in out] == ["r1_t"] and out[0]["commit"] == "abc1234d" and "it=3" in out[0]["log"]
    mod.main(["--runs", str(tmp_path)])  # the table
    assert "r1_t" in capsys.readouterr().out


def test_logstd_init_with_warm_start(tmp_path):
    """Variant A: --logstd-init applies with --init-policy, and with --init only via --reset-logstd."""
    a = ppo.parse_args(["--name", "x", "--init-policy", str(ROUND1), "--logstd-init", "-0.7"])
    st = ppo.build_state(a)
    assert np.all(st.logstd == -0.7)
    np.savez(tmp_path / "state.npz", actor=st.actor.p, critic=st.critic.p, logstd=np.full(3, -2.0), ceiling=0.3, iter=5)
    keep = ppo.build_state(ppo.parse_args(["--name", "x", "--init", str(tmp_path / "state.npz"), "--logstd-init", "-0.7"]))
    assert np.all(keep.logstd == -2.0) and keep.it0 == 6
    reset = ppo.build_state(ppo.parse_args(["--name", "x", "--init", str(tmp_path / "state.npz"), "--logstd-init", "-0.7",
                                            "--reset-logstd"]))
    assert np.all(reset.logstd == -0.7)
    np.testing.assert_array_equal(reset.actor.p, st.actor.p)


def test_ppo_main_arch_set(tmp_path):
    pytest.importorskip("torch")
    from smartcrossing import ppo

    runs = tmp_path / "runs"
    common = ["--workers", "1", "--envs", "2", "--horizon", "16", "--hidden", "8,8", "--minibatches", "2", "--epochs",
              "1", "--eval-every", "0", "--demand-start", "0.3", "--arch", "set", "--set-phi", "8", "--set-heads", "2",
              "--set-dk", "4", "--cfg", json.dumps({"n_slots": 8, "n_neighbours": 6})]
    with pytest.raises(SystemExit, match="torch update"):
        ppo.main(["--name", "bad", "--iters", "1", "--runs-dir", str(runs), *common])
    ppo.main(["--name", "s", "--iters", "3", "--runs-dir", str(runs), "--update-device", "cpu", *common])
    ppo.main(["--name", "t", "--iters", "2", "--runs-dir", str(runs), "--rollout-device", "cpu", *common])
    # behaviour-cloning warm start from the MLP brain (K 4 -> 6)
    ppo.main(["--name", "b", "--iters", "1", "--runs-dir", str(runs), "--rollout-device", "cpu", "--init-bc",
              str(ROOT / "results" / "round2_best.npz"), "--bc-steps", "5", "--bc-seconds", "3", *common])
    for name in ("s", "t", "b"):
        run = runs / name
        arch = load_arch(run / "current.npz")
        assert arch["type"] == "set" and arch["K"] == 6 and arch["phi"] == [8]
        args = json.loads((run / "args.json").read_text(encoding="utf-8"))
        assert args["actor_arch"] == arch and args["critic_arch"]["n_extra"] == N_PRIV
        sim = Sim(SimConfig(n_slots=8, demand=0.5), 2, seed=0)
        drv = PolicyDriver.load(run / "current.npz", sim)
        for _ in range(20):
            sim.step(drv.act())
    from smartcrossing.gates import stop_gate

    g, _ = stop_gate(f"policy:{runs / 's' / 'current.npz'}", {})
    assert 0.0 <= g["safe_rate_feasible"] <= 1.0
