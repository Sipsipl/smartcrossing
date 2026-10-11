"""Warm start of a new-interface actor from an old checkpoint (policy_compat.transplant_actor)."""

from pathlib import Path

import numpy as np

from smartcrossing.policy import default_in_scale, forward, layer_sizes, load_in_scale, load_npz
from smartcrossing.policy_compat import interface_cfg, obs_keys, transplant_actor
from smartcrossing.sim import SimConfig

OLD = Path(__file__).resolve().parents[1] / "results" / "wide_s1ft2_ckpt2650.npz"


def test_transplant_keeps_accel_and_curvature():
    params, old_sizes, old_cfg = load_npz(OLD)
    old_s = load_in_scale(OLD)
    cfg = SimConfig()  # new interface: 1 lamp, brake light + hazard
    in_scale = default_in_scale(cfg)
    sizes = layer_sizes(cfg, tuple(old_sizes[1:-1]))
    new_p = transplant_actor(OLD, cfg, in_scale, sizes)
    assert new_p.size == sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

    old_keys, new_keys = obs_keys(interface_cfg(old_cfg)), obs_keys(cfg)
    rng = np.random.default_rng(0)
    o_old = rng.normal(0, 1, (256, len(old_keys)))
    for j, k in enumerate(old_keys):  # features without a counterpart in the new interface: 0
        if k not in new_keys:
            o_old[:, j] = 0.0
    pos = {k: j for j, k in enumerate(old_keys)}
    o_new = np.zeros((256, len(new_keys)))
    for j, k in enumerate(new_keys):  # brake / hazard (absent in the old layout) stay 0
        if k in pos:
            o_new[:, j] = o_old[:, pos[k]]
    y_old = forward(params, o_old, old_sizes, old_s)
    y_new = forward(new_p, o_new, sizes, in_scale)
    assert y_new.shape[1] == 3 and y_old.shape[1] == 4
    np.testing.assert_allclose(y_new[:, :3], y_old[:, :3], atol=1e-12)  # accel, curvature, lamp
    assert np.abs(y_old[:, :2]).mean() > 1e-3  # non-trivial outputs


def test_transplant_new_inputs_start_at_zero():
    _, old_sizes, _ = load_npz(OLD)
    cfg = SimConfig()
    in_scale = default_in_scale(cfg)
    sizes = layer_sizes(cfg, tuple(old_sizes[1:-1]))
    W = transplant_actor(OLD, cfg, in_scale, sizes)[: sizes[0] * sizes[1]].reshape(sizes[0], sizes[1])
    for j, k in enumerate(obs_keys(cfg)):
        if k.endswith("_brake") or k.endswith("_hazard"):
            assert not W[j].any()
