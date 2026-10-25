"""Reward accounting of ONE car stuck in the box (``cli incentives``, part of the deadlock study PLAN (G)):

    poetry run python -m smartcrossing.cli incentives [--driver policy:results/round7_best.npz] [--cfg JSON]
"""

from .sim import SimConfig


def incentives(cfg_d: dict, gamma: float = 0.99, ages=(20.0, 40.0), go_s: float = 4.0, crash_v=(2.0, 5.0, 10.0)) -> dict:
    """Reward accounting for ONE car stuck in the box (training reward of cfg_d, per-step discount gamma as in
    ppo.py): (a) keep waiting until the max_age removal (from age 20 / 40 s), (b) go now and get out after go_s
    seconds, (c) go now and crash at an impact speed of 2 / 5 / 10 m/s. Undiscounted and discounted values
    (what the critic / advantage sees). Progress shaping (r_progress per metre) is left out: the same few
    hundredths for every option up to the crash / exit point. break_even_p_crash: the crash probability below
    which going now beats waiting to the timeout (discounted, crash at 5 m/s)."""
    c = SimConfig(**{k: v for k, v in cfg_d.items() if k in SimConfig.__dataclass_fields__})
    g = lambda s: gamma ** (s / c.dt)  # noqa: E731  discount of a reward s seconds ahead

    def time_cost(s, disc):
        n = int(round(s / c.dt))
        return -c.r_time * c.dt * (sum(gamma ** k for k in range(n)) if disc else n)

    opt = {}
    for disc in (False, True):
        sfx = "_disc" if disc else ""
        for age in ages:
            left = c.max_age - age
            opt[f"wait_to_timeout_age{age:g}{sfx}"] = round(time_cost(left, disc) + c.r_timeout * (g(left) if disc else 1), 3)
        opt[f"go_exit_{go_s:g}s{sfx}"] = round(time_cost(go_s, disc) + c.r_exit * (g(go_s) if disc else 1), 3)
        for v in crash_v:
            opt[f"go_crash_{v:g}mps{sfx}"] = round(c.r_crash - c.r_crash_speed * v, 3)
    w, e, k = opt[f"wait_to_timeout_age{ages[0]:g}_disc"], opt[f"go_exit_{go_s:g}s_disc"], opt["go_crash_5mps_disc"]
    return {"gamma": gamma, "horizon_s": round(c.dt / (1 - gamma), 1), "r_time_per_s": c.r_time,
            "r_timeout": c.r_timeout, "max_age_s": c.max_age, "r_exit": c.r_exit, "r_crash": c.r_crash,
            "r_crash_speed": c.r_crash_speed, "options": opt,
            "break_even_p_crash": round((e - w) / (e - k), 3)}
