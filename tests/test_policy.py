import json

import numpy as np

from smartcrossing.policy import (
    PolicyDriver,
    export_json,
    forward,
    init_params,
    layer_sizes,
    load_npz,
    n_params,
    save_npz,
)
from smartcrossing.sim import Sim, SimConfig, act_dim, obs_dim


def _setup(B=3, N=5):
    cfg = SimConfig(n_slots=N)
    sizes = layer_sizes(cfg, (8, 6))
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(B, N, obs_dim(cfg)))
    params = np.stack([init_params(sizes, i, out_scale=1.0) for i in range(B)])
    return cfg, sizes, obs, params


def test_forward_shapes():
    cfg, sizes, obs, params = _setup()
    assert params.shape[1] == n_params(sizes)
    out = forward(params[0], obs, sizes)
    assert out.shape == (3, 5, act_dim(cfg))
    assert np.all(np.abs(out) <= 1)
    assert forward(params[0], obs[0], sizes).shape == (5, act_dim(cfg))


def test_save_load_roundtrip_and_driver(tmp_path):
    cfg, sizes, _, params = _setup()
    path = tmp_path / "p.npz"
    save_npz(path, params[0], sizes, cfg, gen=3)
    p, s, c = load_npz(path)
    np.testing.assert_array_equal(p, params[0])
    assert s == sizes and c["n_slots"] == cfg.n_slots
    sim = Sim(SimConfig(n_slots=5, demand=0.5), 2, seed=0)
    for _ in range(30):
        sim.step(np.zeros((2, 5, act_dim(sim.cfg))))
    drv = PolicyDriver.load(path, sim)
    a = drv.act()
    assert a.shape == (2, 5, act_dim(sim.cfg))
    np.testing.assert_allclose(a, forward(params[0], sim.observe(), sizes))


def test_groups_and_mute(tmp_path):
    cfg, sizes, _, params = _setup()
    sim = Sim(SimConfig(n_slots=5, demand=0.5), 2, seed=0, group_probs=[0.5, 0.5])
    for _ in range(40):
        sim.step(np.zeros((2, 5, act_dim(sim.cfg))))
    drv = PolicyDriver(sim, [params[0], params[1]], sizes, mute=[False, True])
    a = drv.act()
    obs = sim.observe()
    g1 = sim.group == 1
    np.testing.assert_allclose(a[~g1], forward(params[0], obs[~g1], sizes))
    assert np.all(a[g1][:, 2:] == 0)
    np.testing.assert_allclose(a[g1][:, :2], forward(params[1], obs[g1], sizes)[:, :2])


def test_json_export(tmp_path):
    cfg, sizes, obs, params = _setup()
    save_npz(tmp_path / "p.npz", params[0], sizes, cfg)
    d = export_json(tmp_path / "p.npz", tmp_path / "p.json")
    d2 = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert d2["sizes"] == sizes and len(d2["layers"]) == len(sizes) - 1
    h = obs[0]
    for layer in d["layers"]:
        h = np.tanh(h @ np.array(layer["W"]) + np.array(layer["b"]))
    np.testing.assert_allclose(h, forward(params[0], obs[0], sizes))


def test_load_several_policies_needs_one_interface(tmp_path):
    import dataclasses

    import pytest

    cfg, sizes, _, params = _setup()
    save_npz(tmp_path / "a.npz", params[0], sizes, cfg)
    save_npz(tmp_path / "b.npz", params[1], sizes, cfg)
    save_npz(tmp_path / "far.npz", params[1], sizes, dataclasses.replace(cfg, neighbour_vis=60.0))
    sim = Sim(SimConfig(n_slots=5), 2, seed=0, group_probs=[0.5, 0.5])
    drv = PolicyDriver.load(f"{tmp_path / 'a.npz'}+{tmp_path / 'b.npz'}", sim)
    assert len(drv.params) == 2
    with pytest.raises(ValueError, match="neighbour_vis"):
        PolicyDriver.load(f"{tmp_path / 'a.npz'}+{tmp_path / 'far.npz'}", Sim(SimConfig(n_slots=5), 2, seed=0))
