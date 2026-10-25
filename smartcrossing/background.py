"""Background mode of a training run (owner games / watches videos meanwhile): BELOW_NORMAL process priority on
Windows (``ppo --low-priority``, inherited by the spawned workers / eval processes via LOW_PRIORITY_ENV) and the
duty-cycle pause after each iteration (``--gpu-duty``, runs/<name>/duty.txt)."""

import os
import sys
from pathlib import Path

LOW_PRIORITY_ENV = "SC_LOW_PRIORITY"  # set by --low-priority; spawned workers / eval lower themselves too
BELOW_NORMAL_PRIORITY_CLASS = 0x4000


def set_low_priority() -> bool:
    """BELOW_NORMAL priority class for this process on Windows (ctypes, no dependency); elsewhere a no-op.
    Returns whether it was applied."""
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE  # (pseudo handle -1: must not be truncated to a 32-bit int)
    k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k32.SetPriorityClass.restype = wintypes.BOOL
    return bool(k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS))


def maybe_low_priority():
    """In a spawned worker / eval process: lower the priority if the training run asked for it."""
    if os.environ.get(LOW_PRIORITY_ENV) == "1":
        set_low_priority()


def read_duty(path: Path, default: float) -> float:
    """GPU duty for the next pause: the float in `path` (duty.txt) if present and valid (clamped to
    [0.02, 1]), else default."""
    try:
        return min(max(float(path.read_text(encoding="utf-8").strip()), 0.02), 1.0)
    except (OSError, ValueError):
        return default


def duty_sleep(t_busy: float, duty: float) -> float:
    """Pause after an iteration of t_busy seconds so that busy / (busy + pause) = duty."""
    return t_busy * (1.0 / duty - 1.0) if 0 < duty < 1 else 0.0
