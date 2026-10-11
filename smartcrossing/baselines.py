"""Hand-written reference drivers (they *do* use lanes and rules).

They are not part of the learning problem; they exist to (a) prove the simulator
is drivable and (b) give lower/upper reference numbers for throughput:

- ``light``: actuated traffic light, exclusive green per arm in rotation (a
  conservative lower reference); empty approaches are skipped, green gaps out
  when its approach empties; amber rule: a car only enters if it can clear the
  box before the clearance interval ends
- ``reservation``: AIM-style central manager (Dresner & Stone): a car reserves
  its whole space-time trajectory (3-disc footprint per 0.1 s) through the box
  and then executes that speed plan exactly; an upper reference
- ``none``: lane-following and car-following only, no intersection control

Arrivals: the baselines set ``cfg.spawn_right`` (cars arrive in the right half of the road,
as lane-disciplined traffic would; since 2026-10-11 this is the default for the learners too).
Wrecks (``cfg.wreck_time``) are standing obstacles for the car following (no rule knows about them).

Lanes (right-hand traffic): the road has n_lanes(cfg) lanes per direction (1 on the 7 m
crossing, 2 on the 14 m crossing). Right turners use the right (outer) lane, left turners
the left (inner) lane, straight cars the lane nearest to their spawn offset (reservation: every
car keeps the lane nearest to its spawn offset, the manager resolves the crossing conflicts);
turns end in the corresponding lane of the target arm. Cars of different lanes queue independently.

All share a path-following steering law (curvature feed-forward + lateral and
heading feedback, critically damped in the arc-length domain), a speed limit
from a lateral-acceleration bound on the turns, and IDM car following.
"""

import numpy as np

from .sim import DISC_OFFSETS, DISC_R, LEFT, RIGHT, STRAIGHT, Sim, lane_offsets, rot

STEP = 0.5  # path resolution in metres
MIN_GREEN = 3.0  # s, actuated light: earliest gap-out
A_LAT = 4.0  # m/s^2 lateral acceleration bound on the turns
A_DEC = 3.0  # m/s^2 comfortable deceleration when approaching a turn
K_E, K_H = 0.05, 2 * np.sqrt(0.05)  # lateral / heading feedback gains (per m)
# merging (lateral error > MERGE_E on the approach, e.g. a car spawned in the oncoming half of
# the wide road): heading-target law (desired heading offset -K_PM * e, capped at H_MAX, tracked
# with gain K_HHM per m; ~critically damped), speed capped at V_MERGE_MAX so the grip limit
# allows the sidestep; while an oncoming car is in the strip to cross: no sidestep, crawl at
# V_MERGE (stop if it is right ahead in the own corridor)
MERGE_E = 0.8
K_PM, K_HHM, H_MAX = 0.15, 0.5, 0.5
V_MERGE, V_MERGE_MAX = 3.0, 9.0


def canonical_path(man, L, W, Rc, off=None):
    """Lane centre-line for a car entering from the south, right-hand traffic, at lateral
    offset `off` (right of the road centre; default W / 2), resampled at uniform arc
    length STEP. Turns follow arcs concentric with the curb fillets."""
    c = W + Rc
    y0 = -L - 2
    o = W / 2 if off is None else off
    if man == STRAIGHT:
        pts = [np.array([[o, y0], [o, L + 8]])]
    else:
        pts = [np.array([[o, y0], [o, -c]])]
        if man == RIGHT:
            r, cx, cy, a0, a1 = c - o, c, -c, np.pi, np.pi / 2
        else:
            r, cx, cy, a0, a1 = c + o, -c, -c, 0.0, np.pi / 2
        a = np.linspace(a0, a1, 200)
        pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], -1))
        if man == RIGHT:
            pts.append(np.array([[c, -o], [L + 8, -o]]))
        else:
            pts.append(np.array([[-c, o], [-L - 8, o]]))
    p = np.concatenate(pts)
    seg = np.hypot(*np.diff(p, axis=0).T)
    keep = np.concatenate([[True], seg > 1e-9])
    p = p[keep]
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    s = np.arange(0.0, cum[-1], STEP)
    return np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], -1)


class Path:
    """A resampled lane path with heading, curvature, speed limit and box limits."""

    def __init__(self, pts, cfg, box):
        self.p = pts
        d = np.diff(pts, axis=0)
        h = np.arctan2(d[:, 1], d[:, 0])
        self.h = np.concatenate([h, h[-1:]])
        dh = np.diff(np.unwrap(self.h)) / STEP
        k = np.concatenate([dh, [0.0]])
        # curvature at a point = mean of the adjacent segments (removes the half-step lag)
        self.k = 0.5 * (k + np.concatenate([[0.0], k[:-1]]))
        vl = np.minimum(cfg.v_max, np.sqrt(A_LAT / np.maximum(np.abs(self.k), 1e-6)))
        for i in range(len(vl) - 2, -1, -1):
            vl[i] = min(vl[i], np.sqrt(vl[i + 1] ** 2 + 2 * A_DEC * STEP))
        self.vlim = vl
        inbox = (np.abs(pts[:, 0]) <= box) & (np.abs(pts[:, 1]) <= box)
        idx = np.nonzero(inbox)[0]
        self.stop_s = idx[0] * STEP
        self.box_out_s = (idx[-1] + 1) * STEP
        self.n = len(pts)

    def at(self, s):
        """Interpolated (x, y, heading) at arc length s (array)."""
        f = np.clip(np.asarray(s) / STEP, 0, self.n - 1)
        i0 = np.minimum(f.astype(int), self.n - 2)
        w = f - i0
        x = self.p[i0, 0] * (1 - w) + self.p[i0 + 1, 0] * w
        y = self.p[i0, 1] * (1 - w) + self.p[i0 + 1, 1] * w
        h = self.h[i0] + w * ((self.h[i0 + 1] - self.h[i0] + np.pi) % (2 * np.pi) - np.pi)
        return x, y, h

    def vlim_at(self, s):
        return self.vlim[min(max(int(s / STEP), 0), self.n - 1)]


class LaneDriver:
    def __init__(self, sim: Sim, control: str = "reservation", green: float = 10.0, clearance: float = 3.0):
        assert control in ("none", "light", "reservation"), control
        self.sim = sim
        self.cfg = sim.cfg
        # rule-based traffic arrives lane-disciplined (right half of the road); the arrival
        # times, manoeuvres and the order of the random draws are unchanged (CRN)
        sim.cfg.spawn_right = True
        self.control = control
        self.green, self.clear = green, clearance
        L, W = self.cfg.L, self.cfg.W
        self.box = W + self.cfg.Rc
        self.offs = lane_offsets(self.cfg)  # (right lane, left lane) offsets; equal for one lane
        self.paths = {}  # (arm, manoeuvre, lane 0 = right / 1 = left) -> Path
        for k in range(4):
            c, s = rot(k)
            for m in (RIGHT, STRAIGHT, LEFT):
                for lane, off in enumerate(self.offs):
                    p = canonical_path(m, L, W, self.cfg.Rc, off)
                    p = np.stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]], -1)
                    self.paths[k, m, lane] = Path(p, self.cfg, self.box)
        self.lane = {}  # (b, uid) -> lane chosen at first sight
        self.merging = set()  # (b, uid) of cars in merge mode
        # backwards compatible views (right lane)
        self.stop_s = {key[:2]: p.stop_s for key, p in self.paths.items() if key[2] == 0}
        self.box_out_s = {key[:2]: p.box_out_s for key, p in self.paths.items() if key[2] == 0}
        self.idx = {}  # (b, uid) -> path index of the last projection
        self.committed = [set() for _ in range(sim.B)]  # uids allowed to enter the box
        self.plans = [dict() for _ in range(sim.B)]  # reservation: uid -> (k0, v, X, Y)
        self.granted = self.committed  # old name
        # actuated light state per env
        self.ls_arm = np.zeros(sim.B, dtype=int)
        self.ls_green = np.zeros(sim.B, dtype=bool)
        self.ls_t0 = np.zeros(sim.B)
        self.clear_by = np.zeros(sim.B)  # time by which all committed cars have left the box
        self.pending = np.zeros(sim.B, dtype=bool)  # a committed car has not yet left the box

    # -------------------------------------------------------------- helpers
    def _step_index(self):
        return int(round(self.sim.t / self.cfg.dt))

    def _project(self, b, n, path):
        """Continuous arc length and signed lateral error (left positive)."""
        key = (b, self.sim.uid[b, n])
        x, y = self.sim.x[b, n], self.sim.y[b, n]
        i0 = self.idx.get(key, 0)
        win = path.p[i0 : i0 + 60]
        i = i0 + int(np.argmin((win[:, 0] - x) ** 2 + (win[:, 1] - y) ** 2))
        self.idx[key] = i
        ch, sh = np.cos(path.h[i]), np.sin(path.h[i])
        dx, dy = x - path.p[i, 0], y - path.p[i, 1]
        return i * STEP + ch * dx + sh * dy, -sh * dx + ch * dy, i

    def _free_acc(self, v, s):
        cfg = self.cfg
        a = cfg.acc_max * (1 - (v / cfg.v_max) ** 4)
        target = self._path_cur.vlim_at(s + v * cfg.dt)
        a = min(a, (target - v) / cfg.dt)
        return max(-cfg.brake_max, min(cfg.acc_max, a))

    def _predict(self, path, s, v, s_end, max_steps=400):
        """Speeds and arc lengths under the free-driving law until s_end
        (same law as _free_acc, inlined for speed)."""
        cfg = self.cfg
        dt, am, bm, vmax = cfg.dt, cfg.acc_max, cfg.brake_max, cfg.v_max
        vl, nmax = path.vlim, path.n - 1
        ss, vs = [s], [v]
        for _ in range(max_steps):
            if s >= s_end:
                break
            a = am * (1 - (v / vmax) ** 4)
            i = int((s + v * dt) / STEP)
            a = min(a, (vl[min(max(i, 0), nmax)] - v) / dt)
            a = max(-bm, min(am, a))
            v = min(max(v + a * dt, 0.0), vmax)
            s += v * dt
            ss.append(s)
            vs.append(v)
        return np.array(ss), np.array(vs)

    def _update_light(self, b):
        """Actuated signal for env b: exclusive green per arm, max green
        `self.green`, gap-out after MIN_GREEN when the green approach is empty and
        another one waits, empty approaches are skipped, and the all-red interval
        lasts at least `self.clear` and until every committed car has (by its own
        prediction) left the box. Returns (green arm or -1, remaining max green)."""
        sim, t = self.sim, self.sim.t
        al = sim.alive[b] & ~np.isin(sim.uid[b], list(self.committed[b]))
        c, s = rot(-sim.arm[b])
        cy = s * sim.x[b] + c * sim.y[b]  # along-arm coordinate in the canonical frame
        appr_all = sim.alive[b] & (cy < -self.box) & (cy > -self.box - 45.0)
        appr = appr_all & al
        waiting = np.array([(appr & (sim.arm[b] == k)).any() for k in range(4)])
        present = np.array([(appr_all & (sim.arm[b] == k)).any() for k in range(4)])
        k = self.ls_arm[b]
        el = t - self.ls_t0[b]
        if self.ls_green[b]:
            if el >= self.green - 1e-9 or (el >= MIN_GREEN and not present[k] and waiting.any()):
                self.ls_green[b], self.ls_t0[b] = False, t
        elif el >= self.clear - 1e-9 and t >= self.clear_by[b] and not self.pending[b]:
            order = [(k + d) % 4 for d in range(1, 5)]
            nxt = next((a for a in order if waiting[a]), None)
            if nxt is not None:
                self.ls_arm[b], self.ls_green[b], self.ls_t0[b] = nxt, True, t
        if self.ls_green[b]:
            return self.ls_arm[b], self.green - (t - self.ls_t0[b])
        return -1, 0.0

    def _try_reserve(self, b, uid, path, s_now, v):
        ss, vs = self._predict(path, s_now, v, path.box_out_s + 25.0)
        x, y, h = path.at(ss)
        X = x[:, None] + np.cos(h)[:, None] * DISC_OFFSETS
        Y = y[:, None] + np.sin(h)[:, None] * DISC_OFFSETS
        k0 = self._step_index()
        M = 6  # steps of temporal separation required in every point of space
        need = (2 * DISC_R + 0.5) ** 2
        bb = (X.min() - 3, X.max() + 3, Y.min() - 3, Y.max() + 3)
        T = len(ss)
        A = np.arange(T)[:, None] + np.arange(-M, M + 1)[None]  # (T, 2M+1)
        for k1, _, X1, Y1 in self.plans[b].values():
            if X1.max() < bb[0] or X1.min() > bb[1] or Y1.max() < bb[2] or Y1.min() > bb[3]:
                continue
            Bi = A + (k0 - k1)
            valid = (Bi >= 0) & (Bi < len(X1))
            if not valid.any():
                continue
            Bc = np.clip(Bi, 0, len(X1) - 1)
            dx = X[:, None, :, None] - X1[Bc][:, :, None, :]  # (T, 2M+1, 3, 3)
            dy = Y[:, None, :, None] - Y1[Bc][:, :, None, :]
            close = ((dx**2 + dy**2) < need).any((2, 3))
            if (close & valid).any():
                return False
        self.plans[b][uid] = (k0, vs, X, Y)
        return True

    def _gc(self, b):
        alive_uids = set(self.sim.uid[b, self.sim.alive[b]].tolist())
        k = self._step_index()
        for uid in list(self.plans[b]):
            k0, vs, _, _ = self.plans[b][uid]
            if uid not in alive_uids and k0 + len(vs) < k - 10:
                del self.plans[b][uid]
        self.committed[b] &= alive_uids | set(self.plans[b])
        for key in [key for key in self.idx if key[0] == b and key[1] not in alive_uids]:
            del self.idx[key]
        for key in [key for key in self.lane if key[0] == b and key[1] not in alive_uids]:
            del self.lane[key]
            self.merging.discard(key)

    def _lane_of(self, b, n):
        """Right lane for right turns, left lane for left turns, straight: the lane nearest
        to the car's lateral offset when first seen (= its spawn offset)."""
        key = (b, int(self.sim.uid[b, n]))
        if key not in self.lane:
            m = int(self.sim.man[b, n])
            if self.control == "reservation":  # the manager resolves every conflict: no lane changes
                m = STRAIGHT
            if m == RIGHT:
                lane = 0
            elif m == LEFT:
                lane = 1
            else:
                c, s = rot(-self.sim.arm[b, n])
                cx = c * self.sim.x[b, n] - s * self.sim.y[b, n]  # right of travel on the entry arm
                lane = 0 if abs(cx - self.offs[0]) <= abs(cx - self.offs[1]) else 1
            self.lane[key] = lane
        return self.lane[key]

    # -------------------------------------------------------------- act
    def act(self) -> np.ndarray:
        sim, cfg = self.sim, self.cfg
        act = np.zeros((sim.B, cfg.n_slots, 2 + cfg.n_signal))
        kk = self._step_index()
        for b in range(sim.B):
            self._gc(b)
            if self.control == "light":
                green, rem = self._update_light(b)
            alive = np.nonzero(sim.alive[b])[0]
            if len(alive) == 0:
                self.pending[b] = False
                continue
            # order by distance to the centre so cars closer to the box decide first
            alive = alive[np.argsort(np.hypot(sim.x[b, alive], sim.y[b, alive]))]
            # obstacles: the driving cars (same order, index j) followed by the wrecks (standing cars)
            obst = np.concatenate([alive, np.nonzero(sim.wreck[b])[0]])
            ax, ay, ath, av = sim.x[b, obst], sim.y[b, obst], sim.th[b, obst], sim.v[b, obst]
            arm_blocked = {}  # (arm, lane) -> an uncommitted car of this lane is ahead
            uids = sim.uid[b, alive]
            planned = alive[[int(u) in self.plans[b] and kk - self.plans[b][int(u)][0] < len(self.plans[b][int(u)][1]) - 1 for u in uids]]
            for j, n in enumerate(alive):
                arm = int(sim.arm[b, n])
                lane = self._lane_of(b, n)
                path = self.paths[arm, int(sim.man[b, n]), lane]
                self._path_cur = path
                uid = int(sim.uid[b, n])
                s_now, e, i = self._project(b, n, path)
                v = sim.v[b, n]
                th = sim.th[b, n]
                # steering: curvature feed-forward + critically damped feedback
                i_ff = min(i + int(round(0.5 * v * cfg.dt / STEP)), path.n - 1)
                eh = (th - path.h[i] + np.pi) % (2 * np.pi) - np.pi
                # merge mode with hysteresis (switching back to the soft gains with a large
                # heading error would overshoot towards the curb)
                mkey = (b, uid)
                if abs(e) > MERGE_E and s_now < path.stop_s:
                    self.merging.add(mkey)
                elif mkey in self.merging and ((abs(e) < 0.15 and abs(np.sin(eh)) < 0.03) or s_now >= path.stop_s):
                    self.merging.discard(mkey)
                merging = mkey in self.merging
                hold, back, stop = False, False, False
                if merging and len(obst) > 1 and uid not in self.committed[b]:
                    # oncoming cars in the strip between me and my lane (incl. my own corridor):
                    # stop and wait (no sidestep) until they have passed or stopped for me;
                    # too close laterally: step back away from them
                    others = np.arange(len(obst)) != j
                    dx, dy = ax[others] - sim.x[b, n], ay[others] - sim.y[b, n]
                    f, lat = np.cos(th) * dx + np.sin(th) * dy, -np.sin(th) * dx + np.cos(th) * dy
                    onc = np.abs((ath[others] - th + np.pi) % (2 * np.pi) - np.pi) > 2 * np.pi / 3
                    toward = -np.sign(e) * lat  # > 0: on the side I have to move to
                    in_strip = (toward > -2.6) & (np.abs(lat) < abs(e) + 2.6)
                    moving = av[others] > 0.3
                    rel = onc & in_strip & moving & (f > -5.0) & (f < 12.0 + 3.0 * av[others])
                    hold = bool(rel.any())
                    adj = onc & (np.abs(lat) < 2.3) & (f > -5.0) & (f < 9.0) & (toward > 0)
                    stop = bool((adj & moving).any())  # it has not stopped for me (yet): stop
                    back = bool((adj & ~moving).any()) and not stop  # it waits for me: give way sideways
                if merging:
                    h_des = np.sign(e) * 0.3 if back else 0.0 if hold or stop else -np.clip(K_PM * e, -H_MAX, H_MAX)
                    kappa = path.k[i_ff] + K_HHM * (h_des - eh)
                else:
                    kappa = path.k[i_ff] - K_E * np.clip(e, -3.0, 3.0) - K_H * np.sin(eh)
                committed = uid in self.committed[b]
                if j == 0:
                    self.pending[b] = False
                if committed and s_now < path.box_out_s + 4.6:
                    self.pending[b] = True
                before = s_now < path.stop_s
                # car following: nearest other car near my path ahead
                gap, v_lead = np.inf, 0.0
                tight = False
                hi = min(i + 90, path.n)
                ahead = path.p[i + 1 : hi]
                if len(ahead):
                    others = np.arange(len(obst)) != j
                    d2 = (ahead[None, :, 0] - ax[others, None]) ** 2 + (ahead[None, :, 1] - ay[others, None]) ** 2
                    kmin = d2.argmin(1)
                    dmin = np.sqrt(d2[np.arange(len(kmin)), kmin])
                    dth = (ath[others] - th + np.pi) % (2 * np.pi) - np.pi
                    s_hit = (i + 1 + kmin) * STEP
                    in_box = (s_hit >= path.stop_s - 2.0) & (s_hit <= path.box_out_s + 2.0)
                    cand = dmin < 2.2
                    # oncoming cars (e.g. a fresh spawn merging out of my lane) only matter
                    # when close; they are treated as standing obstacles
                    oncoming = np.abs(dth) > 2 * np.pi / 3
                    # (not while I am still crossing the box: I must clear it first)
                    cand &= ~oncoming | ((kmin * STEP < 40.0) & (s_now > path.box_out_s))
                    if self.control == "none":
                        # nobody manages crossing traffic in the box
                        cand &= ~(in_box & (np.abs(dth) > np.pi / 3))
                    elif self.control == "reservation" and uid in self.plans[b]:
                        # separation from other planned cars is guaranteed by the manager
                        # (inside the box); outside it planned cars follow each other tightly
                        tight = True
                        cand &= ~(np.isin(obst[others], planned) & in_box)
                    if cand.any():
                        # oncoming cars (merging spawns): stop well before them so they can cross
                        g = (kmin + 1) * STEP - 4.6 - np.where(oncoming, 8.0, 0.0)
                        g = np.where(cand, g, np.inf)
                        m = int(np.argmin(g))
                        gap, v_lead = float(g[m]), float(max(av[others][m] * np.cos(dth[m]), 0.0))
                # intersection control: an uncommitted car treats the stop line as a wall
                if before and self.control != "none" and not committed:
                    allowed = False
                    if not arm_blocked.get((arm, lane)) and abs(e) < 1.0 and not hold:
                        if self.control == "light":
                            if green == arm:
                                ss, _ = self._predict(path, s_now, v, path.box_out_s + 4.6)
                                t_need = (len(ss) - 1) * cfg.dt
                                allowed = t_need <= rem + self.clear - 0.5
                                if allowed:
                                    self.clear_by[b] = max(self.clear_by[b], sim.t + t_need + 0.5)
                        elif (kk + uid) % 2 == 0:  # requests every 0.2 s per car (cost)
                            allowed = self._try_reserve(b, uid, path, s_now, v)
                    if allowed:
                        self.committed[b].add(uid)
                        committed = True
                    else:
                        arm_blocked[arm, lane] = True
                        g = path.stop_s - s_now - (6.0 if merging else 2.5)
                        if g < gap:
                            gap, v_lead = g, 0.0
                self._path_cur = path
                acc = min(self._free_acc(v, s_now), self._idm(v, gap, v_lead))
                plan = self.plans[b].get(uid)
                if plan is not None and kk + 1 - plan[0] < len(plan[1]):
                    # execute the reserved speed plan (IDM can only make it more careful)
                    a_plan = (plan[1][kk + 1 - plan[0]] - v) / cfg.dt
                    a_idm = self._idm(v, gap, v_lead, s0=1.0, T=0.3) if tight else self._idm(v, gap, v_lead)
                    acc = min(a_plan, a_idm, (path.vlim_at(s_now + v * cfg.dt) - v) / cfg.dt)
                if merging:
                    acc = min(acc, ((0.0 if stop else V_MERGE if hold or back else V_MERGE_MAX) - v) / cfg.dt)
                acc = float(np.clip(acc, -cfg.brake_max, cfg.acc_max))
                if v <= 0.0:  # the reference drivers never use the reverse gear
                    acc = max(acc, 0.0)
                act[b, n, 0] = acc / cfg.acc_max if acc > 0 else acc / cfg.brake_max
                act[b, n, 1] = np.clip(kappa / cfg.kappa_max, -1, 1)
        return act

    def _idm(self, v, gap, v_lead, s0=2.0, T=1.0, b=3.0):
        cfg = self.cfg
        if not np.isfinite(gap):
            return cfg.acc_max
        s_star = s0 + v * T + v * (v - v_lead) / (2 * np.sqrt(cfg.acc_max * b))
        a = cfg.acc_max * (1 - (v / cfg.v_max) ** 4) - cfg.acc_max * (max(s_star, 0) / max(gap, 0.1)) ** 2
        return float(np.clip(a, -cfg.brake_max, cfg.acc_max))
