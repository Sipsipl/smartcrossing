"""Hand-written reference drivers (they *do* use lanes and rules).

They are not part of the learning problem; they exist to (a) prove the simulator
is drivable and (b) give lower/upper reference numbers for throughput:

- ``light``: exclusive green per arm in rotation (a conservative traffic light)
- ``reservation``: AIM-style central manager (Dresner & Stone) granting
  space-time reservations of the intersection box; an upper reference
- ``none``: lane-following and car-following only, no intersection control

All share pure-pursuit steering along right-hand lane paths and IDM car following.
"""

import numpy as np

from .sim import LEFT, RIGHT, STRAIGHT, Sim, rot

STEP = 0.5  # path resolution in metres


def canonical_path(man, L, W, Rc):
    """Lane centre-line for a car entering from the south, right-hand traffic.
    Turns follow arcs concentric with the curb fillets."""
    pts = []
    c = W + Rc
    y0 = -L - 2
    if man == STRAIGHT:
        ys = np.arange(y0, L + 8, STEP)
        return np.stack([np.full_like(ys, W / 2), ys], -1)
    ys = np.arange(y0, -c, STEP)
    pts.append(np.stack([np.full_like(ys, W / 2), ys], -1))
    if man == RIGHT:
        r, cx, cy, a0, a1 = c - W / 2, c, -c, np.pi, np.pi / 2
    else:
        r, cx, cy, a0, a1 = c + W / 2, -c, -c, 0.0, np.pi / 2
    n = max(int(abs(a1 - a0) * r / STEP), 2)
    a = np.linspace(a0, a1, n)
    pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], -1))
    xs = np.arange(c, L + 8, STEP)
    if man == RIGHT:
        pts.append(np.stack([xs, np.full_like(xs, -W / 2)], -1))
    else:
        pts.append(np.stack([-xs, np.full_like(xs, W / 2)], -1))
    return np.concatenate(pts)


class LaneDriver:
    def __init__(self, sim: Sim, control: str = "reservation", green: float = 6.0, clearance: float = 3.0):
        self.sim = sim
        self.cfg = sim.cfg
        self.control = control
        self.green, self.clear = green, clearance
        L, W = self.cfg.L, self.cfg.W
        self.paths = {}
        for k in range(4):
            c, s = rot(k)
            for m in (RIGHT, STRAIGHT, LEFT):
                p = canonical_path(m, L, W, self.cfg.Rc)
                self.paths[k, m] = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], -1)
        # s-coordinate of the stop line (box edge) on every path
        self.box = W + self.cfg.Rc
        self.stop_s = {}
        self.box_out_s = {}
        for key, p in self.paths.items():
            inbox = (np.abs(p[:, 0]) <= self.box) & (np.abs(p[:, 1]) <= self.box)
            idx = np.nonzero(inbox)[0]
            self.stop_s[key] = idx[0] * STEP
            self.box_out_s[key] = (idx[-1] + 1) * STEP
        self.reservations = [dict() for _ in range(sim.B)]  # cell -> list of (t0, t1, uid)
        self.granted = [set() for _ in range(sim.B)]
        self.idx = {}  # (b, uid) -> path index

    # -------------------------------------------------------------- helpers
    def _progress(self, b, n):
        key = (b, self.sim.uid[b, n])
        p = self.paths[self.sim.arm[b, n], self.sim.man[b, n]]
        i0 = self.idx.get(key, 0)
        win = p[i0 : i0 + 40]
        d = np.hypot(win[:, 0] - self.sim.x[b, n], win[:, 1] - self.sim.y[b, n])
        i = i0 + int(np.argmin(d))
        self.idx[key] = i
        return p, i

    def _green_arm(self):
        cyc = self.green + self.clear
        ph = self.sim.t % (4 * cyc)
        k = int(ph // cyc)
        return k if (ph - k * cyc) < self.green else -1

    def _cells(self, path, s_from, v0, t0):
        """Box cells a car occupies when driving its path from s_from accelerating to v_max."""
        cfg = self.cfg
        out = []
        s, v, t = s_from, v0, t0
        end = self.box_out_s_cur + 6.0
        while s < end:
            i = min(int(s / STEP), len(path) - 1)
            x, y = path[i]
            if abs(x) <= self.box + 1 and abs(y) <= self.box + 1:
                out.append((int(np.floor(x)), int(np.floor(y)), t))
            v = min(v + cfg.acc_max * 0.1, cfg.v_max)
            s += v * 0.1
            t += 0.1
        return out

    def _try_reserve(self, b, uid, path, s_now, v, key):
        table = self.reservations[b]
        t0 = self.sim.t
        self.box_out_s_cur = self.box_out_s[key]
        cells = self._cells(path, s_now, v, t0)
        margin = 0.6  # seconds of separation in each cell
        claims = {}
        for cx, cy, t in cells:
            for ox in (-2, -1, 0, 1, 2):
                for oy in (-2, -1, 0, 1, 2):
                    if ox * ox + oy * oy > 5:
                        continue
                    c = (cx + ox, cy + oy)
                    lo, hi = claims.get(c, (t, t))
                    claims[c] = (min(lo, t), max(hi, t))
        for c, (lo, hi) in claims.items():
            for r0, r1, other in table.get(c, ()):
                if other != uid and lo - margin < r1 and r0 < hi + margin:
                    return False
        for c, (lo, hi) in claims.items():
            table.setdefault(c, []).append((lo, hi, uid))
        self.granted[b].add(uid)
        return True

    def _gc(self, b):
        t = self.sim.t
        table = self.reservations[b]
        for c in list(table):
            table[c] = [r for r in table[c] if r[1] > t - 1]
            if not table[c]:
                del table[c]

    # -------------------------------------------------------------- act
    def act(self) -> np.ndarray:
        sim, cfg = self.sim, self.cfg
        act = np.zeros((sim.B, cfg.n_slots, 2 + cfg.n_signal))
        green = self._green_arm()
        for b in range(sim.B):
            if self.control == "reservation":
                self._gc(b)
            alive = np.nonzero(sim.alive[b])[0]
            # order by distance to the box so earlier cars reserve first
            alive = sorted(alive, key=lambda n: np.hypot(sim.x[b, n], sim.y[b, n]))
            for n in alive:
                key = (sim.arm[b, n], sim.man[b, n])
                uid = sim.uid[b, n]
                path, i = self._progress(b, n)
                s_now = i * STEP
                v = sim.v[b, n]
                # pure pursuit
                look = max(4.0, 0.6 * v)
                j = min(i + int(look / STEP), len(path) - 1)
                tx, ty = path[j]
                alpha = np.arctan2(ty - sim.y[b, n], tx - sim.x[b, n]) - sim.th[b, n]
                alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
                kappa = 2 * np.sin(alpha) / np.hypot(tx - sim.x[b, n], ty - sim.y[b, n])
                # obstacles: nearest car on the path ahead
                gap, v_lead = np.inf, 0.0
                ahead = path[i : i + 90]
                in_box_or_granted = uid in self.granted[b] or s_now >= self.stop_s[key]
                for m in alive:
                    if m == n:
                        continue
                    if in_box_or_granted:
                        dth = (sim.th[b, m] - sim.th[b, n] + np.pi) % (2 * np.pi) - np.pi
                        if abs(dth) > np.pi / 3:
                            continue
                    d = np.hypot(ahead[:, 0] - sim.x[b, m], ahead[:, 1] - sim.y[b, m])
                    k = int(np.argmin(d))
                    if d[k] < 2.2 and k > 0:
                        g = k * STEP - 4.6
                        if g < gap:
                            gap, v_lead = g, sim.v[b, m]
                # intersection control: stop line acts as a standing obstacle
                before = s_now < self.stop_s[key]
                if before and self.control != "none":
                    allowed = False
                    if self.control == "light":
                        allowed = green == sim.arm[b, n]
                    elif self.control == "reservation":
                        allowed = uid in self.granted[b]
                        if not allowed and self.stop_s[key] - s_now < 25:
                            allowed = self._try_reserve(b, uid, path, s_now, v, key)
                    if not allowed:
                        g = self.stop_s[key] - s_now - 2.5
                        if g < gap:
                            gap, v_lead = g, 0.0
                acc = self._idm(v, gap, v_lead)
                if uid in self.granted[b] and np.isinf(gap):
                    acc = cfg.acc_max
                act[b, n, 0] = acc / cfg.acc_max if acc > 0 else acc / cfg.brake_max
                act[b, n, 1] = np.clip(kappa / cfg.kappa_max, -1, 1)
        return act

    def _idm(self, v, gap, v_lead, s0=2.0, T=1.0, b=3.0):
        cfg = self.cfg
        a = cfg.acc_max * (1 - (v / cfg.v_max) ** 4)
        if np.isfinite(gap):
            s_star = s0 + v * T + v * (v - v_lead) / (2 * np.sqrt(cfg.acc_max * b))
            a -= cfg.acc_max * (max(s_star, 0) / max(gap, 0.1)) ** 2
        return float(np.clip(a, -cfg.brake_max, cfg.acc_max))
