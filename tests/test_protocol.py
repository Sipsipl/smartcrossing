"""The evaluation protocol constants (smartcrossing/protocol.py) are measurements' identity: pin the values and
check every module uses them (no second copy)."""

from smartcrossing import deadlock, evalset, protocol, select


def test_values_are_fixed():
    assert protocol.SELECTION_SEEDS == (1000, 2000, 3000)
    assert protocol.REPORT_SEEDS == (4000, 5000, 6000)
    assert (protocol.ENVS, protocol.SECONDS, protocol.SWEEP_SECONDS) == (8, 120.0, 180.0)
    assert protocol.EVAL_DEMANDS == (0.05, 0.15, 0.3, 0.5)
    assert protocol.SELECT_DEMANDS == (0.15, 0.3, 0.5)
    assert protocol.SWEEP_DEMANDS == (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
    assert protocol.SITE_DEMANDS == (0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def test_modules_use_the_protocol():
    assert evalset.evalset.__defaults__[2:4] == (protocol.EVAL_DEMANDS, protocol.SELECTION_SEEDS)
    kw = dict(zip(select.select.__code__.co_varnames[1:], select.select.__defaults__))
    assert (kw["demands"], kw["seeds"], kw["envs"], kw["seconds"]) == (
        protocol.SELECT_DEMANDS, protocol.SELECTION_SEEDS, protocol.ENVS, protocol.SECONDS)
    assert deadlock.CLOUD_DEMANDS == protocol.SWEEP_DEMANDS and deadlock.CLOUD_SEED == protocol.REPORT_SEEDS[0]
    for mod in (evalset, select, deadlock):
        assert not hasattr(mod, "SEEDS") and not hasattr(mod, "DEMANDS")
