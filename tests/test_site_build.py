"""site/build.py without a real build: checkpoint discovery in runs/, the warm-start chain (args.json init /
init_policy / init_bc, matched by weights) and the snapshot round trip of MLP and set brains, on synthetic
checkpoints in tmp_path."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from smartcrossing import policy_set
from smartcrossing.policy import init_params, layer_sizes, save_npz
from smartcrossing.sim import SimConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def build(tmp_path, monkeypatch):
    """site/build.py as a fresh module whose ROOT / runs / snapshot live in tmp_path."""
    spec = importlib.util.spec_from_file_location("site_build_t", ROOT / "site" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(mod, "SNAPSHOT", tmp_path / "results" / "site")
    mod.SOURCE.update(kind="runs", manifest=None, runs=set())
    mod.FOUND.clear()
    return mod


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


def test_discovery_chain_and_snapshot(build, tmp_path):
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

    found = build.discover("any")
    build.FOUND.update(found)
    assert sorted(found) == ["r1_a/ckpt_0001", "r1_a/ckpt_0002", "r1_b/ckpt_0003", "r1_c/ckpt_0001", "r1_d/ckpt_0005"]
    assert all(c["method"] == "ppo" and c["gen"] == int(c["id"][-1]) for c in found.values())
    d = found["r1_d/ckpt_0005"]["policy"]
    assert build.Brain(d).is_set and not build.Brain(found["r1_a/ckpt_0001"]["policy"]).is_set
    assert build.chain_of("r1_b") == [("r1_a", 0), ("r1_b", 2)]
    assert build.chain_of("r1_c") == [("r1_a", 0), ("r1_c", 2)]  # init_policy matched by weights
    assert build.chain_of("r1_d") == [("r1_a", 0), ("r1_d", 1)]  # init_bc matched by weights

    # the Brain payload drives a PolicyDriver for both architectures
    from smartcrossing.sim import Sim

    for cid in ("r1_a/ckpt_0001", "r1_d/ckpt_0005"):
        sim = Sim(SimConfig(demand=0.3), 1, seed=0)
        drv = build.policy_driver(sim, build.Brain(found[cid]["policy"]).payload())
        for _ in range(5):
            sim.step(drv.act())

    # snapshot round trip: written from runs/, read back alone with the same weights / hashes / chain
    used = sorted(found)
    build.write_snapshot(found, used, ["r1_a", "r1_b", "r1_c", "r1_d"])
    brains = tmp_path / "results" / "site" / "brains"
    assert sorted(p.relative_to(brains).as_posix() for p in brains.glob("*/*.npz")) == [f"{c}.npz" for c in used]
    manifest = json.loads((tmp_path / "results" / "site" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["runs"]["r1_d"]["init"] == ["r1_a", 1] and manifest["runs"]["r1_a"]["init"] is None
    build.SOURCE.update(kind="snapshot", manifest=manifest, runs=set())
    build.FOUND.clear()
    snap = build.discover("any")
    assert sorted(snap) == used
    for cid in used:
        assert snap[cid]["hash"] == found[cid]["hash"] and snap[cid].get("snap")
    np.testing.assert_array_equal(build.Brain(snap["r1_d/ckpt_0005"]["policy"]).vector(), build.Brain(d).vector())
    assert snap["r1_d/ckpt_0005"]["policy"]["arch"] == d["arch"]
    assert build.chain_of("r1_d") == [("r1_a", 0), ("r1_d", 1)]
    # rounding keeps the format of each architecture
    r = build.Brain(d).rounded(2)
    assert "params" in r and "layers" not in r and r["params"][0] == round(d["params"][0], 2)
