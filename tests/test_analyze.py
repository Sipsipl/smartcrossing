import numpy as np

from smartcrossing.analyze import mutual_info, region, side_stats
from smartcrossing.sim import LEFT, RIGHT, STRAIGHT, SimConfig


def test_mutual_info():
    a = np.array([0, 1] * 500)
    assert abs(mutual_info(a, a) - 1.0) < 1e-9
    assert mutual_info(a, np.zeros_like(a)) == 0.0


def test_region_and_side_sign():
    cfg = SimConfig()
    # rows: arm, man, x, y, v, acc  (south arm = 0, travelling north when approaching)
    R = np.array([
        [0, STRAIGHT, 1.5, -25.0, 10, 0],  # approaching from the south, 1.5 m right of the centre line
        [0, STRAIGHT, -1.0, 25.0, 10, 0],  # leaving north (travel +y): right is +x, so offset -1
        [1, LEFT, 0.0, -25.0, 10, 0],  # from the east, left turn -> target arm south (leaving, travel -y)
        [3, RIGHT, -20.0, 0.0, 10, 0],  # from the west, approaching
    ], float)
    reg = region(cfg, R[:, 0].astype(int), R[:, 1].astype(int), R[:, 2], R[:, 3])
    assert reg.tolist() == [0, 2, 2, 0]
    s = side_stats(R, cfg)
    assert s["approach"]["per_arm"][0] == 1.5
    assert s["leave"]["per_arm"][2] == -1.0
