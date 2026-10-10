"""Summary metrics computed from accumulated simulation statistics."""

import numpy as np

from .sim import Sim


def jain(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0 or (x**2).sum() == 0:
        return float("nan")
    return float(x.sum() ** 2 / (len(x) * (x**2).sum()))


def summarize(sim: Sim, duration: float) -> dict:
    """Aggregate over all environments of a Sim run for `duration` seconds."""
    st = sim.stats
    minutes = duration / 60 * sim.B
    travel = np.array([t for env in st.travel for t in env]).reshape(-1, 2)
    ratio = travel[:, 0] / travel[:, 1] if len(travel) else np.array([])
    delay = travel[:, 0] - travel[:, 1] if len(travel) else np.array([])
    finished = st.exits.sum() + st.wrong_exits.sum() + st.crashes.sum() + st.offroad.sum() + st.timeouts.sum()
    bad = st.crashes.sum() + st.offroad.sum()
    return {
        "throughput_per_min": float(st.exits.sum() / minutes),
        "demand_per_min": float(st.arrived.sum() / minutes),
        "exit_ok_rate": float(st.exits.sum() / max(finished, 1)),
        "crash_cars_per_1000": float(1000 * bad / max(finished, 1)),
        "crash_cars": int(st.crashes.sum()),
        "offroad_cars": int(st.offroad.sum()),
        "wrong_exits": int(st.wrong_exits.sum()),
        "timeouts": int(st.timeouts.sum()),
        "queued_end": int(sim.queue.sum()),
        "delay_mean_s": float(delay.mean()) if len(delay) else float("nan"),
        "delay_p95_s": float(np.percentile(delay, 95)) if len(delay) else float("nan"),
        "travel_max_s": float(travel[:, 0].max()) if len(travel) else float("nan"),
        "jain_speed": jain(1 / ratio) if len(ratio) else float("nan"),
    }
