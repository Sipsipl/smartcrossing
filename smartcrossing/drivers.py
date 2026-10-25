"""The one driver factory (cli bench / render / gate / evalset / select / deadlock / cloud, site bench):

    light | reservation | none     -> rule-based baselines.LaneDriver
    policy:<file.npz> | <file.npz> -> policy.PolicyDriver (several files joined by '+': one per group)

``noise`` (policies only, cli deadlock --noise): a stochastic policy, action = tanh(mu + sigma * eps), sigma = the
checkpoint's trained std ("policy") or a fixed std (a number) on every action dim; eps from
default_rng([seed, 99]).
"""

import json

import numpy as np

NO_NOISE = (None, "", "none", 0, 0.0, "0")


def policy_logstd(path) -> np.ndarray:
    """The trained Gaussian log-std of a PPO checkpoint (``extra.logstd``)."""
    with np.load(str(path).removeprefix("policy:")) as z:
        return np.asarray(json.loads(str(z["extra"]))["logstd"], dtype=float)


def make_driver(spec: str, sim, noise=None, seed: int = 0):
    """Driver for ``spec`` (module docstring) on ``sim``; ``noise``: action noise of a policy driver."""
    from .policy import PolicyDriver

    if spec in ("light", "reservation", "none"):
        from .baselines import LaneDriver

        drv = LaneDriver(sim, spec)
    else:
        drv = PolicyDriver.load(spec.removeprefix("policy:"), sim)
    if noise in NO_NOISE:
        return drv
    if not isinstance(drv, PolicyDriver):
        raise ValueError("action noise needs a policy driver")
    sigma = np.exp(policy_logstd(spec)) if noise == "policy" else np.full(drv.sizes[-1], float(noise))
    drv.set_noise(sigma, np.random.default_rng([seed, 99]))
    return drv
