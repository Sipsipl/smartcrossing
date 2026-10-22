"""The site build (site/sitebuild/) without a real build: checkpoint discovery in runs/, the warm-start chain
(args.json init / init_policy / init_bc, matched by weights), the snapshot round trip of MLP and set brains, on
synthetic checkpoints in tmp_path (an explicit build Context rooted there); the fundamental-diagram data and the
thin entry point site/build.py."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import policy_set
from smartcrossing.policy import init_params, layer_sizes, save_npz
from smartcrossing.sim import SimConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "site")]

from sitebuild import discover as D  # noqa: E402
from sitebuild import fundamental as F  # noqa: E402
from sitebuild.bench import policy_driver  # noqa: E402
from sitebuild.context import Context  # noqa: E402


def _mlp(path, seed, gen):
    cfg = SimConfig()
    sizes = layer_sizes(cfg, (8,))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, init_params(sizes, seed, out_scale=1.0), sizes, cfg, iter=gen)


def _set(path, seed, gen):
    cfg = SimConfig()
    arch = policy_set.make_arch(cfg, hidden=(8,), phi=(6,), heads=2, dk=3, dv=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path, policy_set.init_params(arch, seed, 1.0), layer_sizes(cfg, (8,)), cfg, arch=arch, iter=gen)


def _args(run_dir, **kw):
    (run_dir / "args.json").write_text(json.dumps(kw), encoding="utf-8")


def test_discovery_chain_and_snapshot(tmp_path):
    runs = tmp_path / "runs"
    _mlp(runs / "r1_a" / "ckpt_0001.npz", 1, 1)
    _mlp(runs / "r1_a" / "ckpt_0002.npz", 2, 2)
    _mlp(runs / "r1_b" / "ckpt_0003.npz", 3, 3)  # resumed from r1_a/ckpt_0002 (args.json init)
    _args(runs / "r1_b", init="runs/r1_a/ckpt_0002.npz")
    _mlp(tmp_path / "results" / "copy.npz", 2, 0)  # = r1_a/ckpt_0002's weights, used as init_policy
    _mlp(runs / "r1_c" / "ckpt_0001.npz", 4, 1)
    _args(runs / "r1_c", init_policy="results/copy.npz")
    _set(runs / "r1_d" / "ckpt_0005.npz", 5, 5)  # set brain behaviour-cloned from r1_a/ckpt_0001
    _args(runs / "r1_d", init_bc="runs/r1_a/ckpt_0001.npz")
    (runs / "smoke_x").mkdir()  # not a training run
    _mlp(runs / "smoke_x" / "ckpt_0001.npz", 9, 1)

    ctx = Context(root=tmp_path)
    assert ctx.setup("auto") == "runs" and ctx.train_runs == {"r1_a", "r1_b", "r1_c", "r1_d"}
    found = D.discover(ctx, "any")
    ctx.found.update(found)
    assert sorted(found) == ["r1_a/ckpt_0001", "r1_a/ckpt_0002", "r1_b/ckpt_0003", "r1_c/ckpt_0001", "r1_d/ckpt_0005"]
    assert all(c["method"] == "ppo" and c["gen"] == int(c["id"][-1]) for c in found.values())
    d = found["r1_d/ckpt_0005"]["policy"]
    assert D.Brain(d).is_set and not D.Brain(found["r1_a/ckpt_0001"]["policy"]).is_set
    assert D.chain_of(ctx, "r1_b") == [("r1_a", 0), ("r1_b", 2)]
    assert D.chain_of(ctx, "r1_c") == [("r1_a", 0), ("r1_c", 2)]  # init_policy matched by weights
    assert D.chain_of(ctx, "r1_d") == [("r1_a", 0), ("r1_d", 1)]  # init_bc matched by weights
    assert D.resolve(ctx, "results/copy.npz") == "r1_a/ckpt_0002"  # a file elsewhere, matched by its weights

    # the Brain payload drives a PolicyDriver for both architectures
    from smartcrossing.sim import Sim

    for cid in ("r1_a/ckpt_0001", "r1_d/ckpt_0005"):
        sim = Sim(SimConfig(demand=0.3), 1, seed=0)
        drv = policy_driver(sim, D.Brain(found[cid]["policy"]).payload())
        for _ in range(5):
            sim.step(drv.act())

    # snapshot round trip: written from runs/, read back alone with the same weights / hashes / chain
    used = sorted(found)
    D.write_snapshot(ctx, used, ["r1_a", "r1_b", "r1_c", "r1_d"])
    brains = tmp_path / "results" / "site" / "brains"
    assert sorted(p.relative_to(brains).as_posix() for p in brains.glob("*/*.npz")) == [f"{c}.npz" for c in used]
    manifest = json.loads((tmp_path / "results" / "site" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"]["r1_d"]["init"] == ["r1_a", 1] and manifest["runs"]["r1_a"]["init"] is None
    shutil.rmtree(runs)  # a fresh clone: only the snapshot
    snap_ctx = Context(root=tmp_path)
    assert snap_ctx.setup("auto") == "snapshot" and snap_ctx.manifest == manifest
    snap = D.discover(snap_ctx, "any")
    snap_ctx.found.update(snap)
    assert sorted(snap) == used
    for cid in used:
        assert snap[cid]["hash"] == found[cid]["hash"] and snap[cid].get("snap")
    np.testing.assert_array_equal(D.Brain(snap["r1_d/ckpt_0005"]["policy"]).vector(), D.Brain(d).vector())
    assert snap["r1_d/ckpt_0005"]["policy"]["arch"] == d["arch"]
    assert D.chain_of(snap_ctx, "r1_d") == [("r1_a", 0), ("r1_d", 1)]
    # rounding keeps the format of each architecture
    r = D.Brain(d).rounded(2)
    assert "params" in r and "layers" not in r and r["params"][0] == round(d["params"][0], 2)
    # the stop gate of a snapshot brain (no checkpoint file) runs from its weights; unknown ids have none
    g = F.gate_of(snap_ctx, "r1_a/ckpt_0001")
    assert set(g) == {"safe", "stop", "cases"} and 0 <= g["stop"] <= g["safe"] <= 1
    assert F.gate_of(snap_ctx, "nope/x") is None and F.gate_text(snap_ctx, "{stop} %", "nope/x") == "? %"


def test_fundamental_data_missing_files_and_start_demand(tmp_path):
    ctx = Context(root=tmp_path)
    assert F.fundamental_data(ctx) is None and F.cloud_data(ctx) == {}
    res = tmp_path / "results"
    res.mkdir()
    pts = [{"demand": d, "density": 2.0 * d, "throughput": t, "density_veh_per_km": 1.0, "demand_per_min": 240 * d,
            "deadlock_time_frac": 0.0, "extra": 1} for d, t in ((0.1, 23.9), (0.3, 70.79), (0.5, 60.0))]
    (res / "deadlock_fundamental.json").write_text(json.dumps({"drivers": {"brain": {"points": pts}, "other": {"points": pts}}}),
                                                   encoding="utf-8")
    out = F.fundamental_data(ctx)
    assert list(out["drivers"]) == ["brain"] and "extra" not in out["drivers"]["brain"][0]
    assert out["cloud"] == {}  # no cloud file
    assert out["max_throughput"] == 70.8 and out["start_demand"] == 0.29  # 70.79 / 240 = 0.29496 -> 0.29
    assert F.start_demand(120.0) == 0.5 and F.start_demand(500.0) == 0.5  # at most the slider maximum
    assert F.start_demand(30.0) == 0.12  # 0.125 -> 0.12 (round half to even on the 0.01 grid)
    (res / "fundamental_cloud.json").write_text(json.dumps({"drivers": {"light": {"points": [[1.0, 2.0, 0.05]]},
                                                                         "x": {"points": [[3.0, 4.0, 0.1]]}}}),
                                                encoding="utf-8")
    assert F.cloud_data(ctx) == {"light": [[1.0, 2.0]]}


def test_entry_point_help():
    """site/build.py stays the entry point (thin wrapper around sitebuild.main)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # the entry point sets its own paths
    res = subprocess.run([sys.executable, str(ROOT / "site" / "build.py"), "--help"], capture_output=True, text=True,
                         encoding="utf-8", env=env)
    assert res.returncode == 0, res.stderr
    assert "--source" in res.stdout and "site/sitebuild/" in res.stdout


@pytest.mark.parametrize("text, expect", [("{safe} / {stop}", "? / ?"), (None, None), ("plain", "plain")])
def test_gate_text_without_checkpoint(text, expect):
    assert F.gate_text(Context(), text, None) == expect
