"""Batched 2D simulation of a signal-free four-way intersection.

Geometry: two perpendicular roads of width 2*W crossing at the origin, each arm
reaching out to distance L. There are no lanes, markings or rules: the drivable
area is simply the cross-shaped union of both strips. Cars enter at the end of
an arm (at a random lateral offset) and must leave through their target arm.

Arms are indexed counter-clockwise: 0 = south, 1 = east, 2 = north, 3 = west.
A car entering from arm k with manoeuvre m in {1: right, 2: straight, 3: left}
must exit through arm (k + m) % 4.

All state lives in arrays of shape (B, N): B independent environments with N
car slots each. The policy is evaluated outside of this module.
"""

from dataclasses import dataclass, field

import numpy as np

RIGHT, STRAIGHT, LEFT = 1, 2, 3

# Car shape: three discs along the heading axis approximate a 4.6 x 1.8 m box.
DISC_OFFSETS = np.array([-1.4, 0.0, 1.4])
DISC_R = 0.9

SPAWN_LAT_MARGIN = 0.8  # lateral corridor (beyond car width) checked for closing traffic at spawn

N_RAYS = 5
RAY_ANGLES = np.radians([-60.0, -30.0, 0.0, 30.0, 60.0])
RAY_RANGE = 20.0  # metres; rays report the exact distance to the road edge, capped here
K_NEIGHBOURS = 4
NEIGHBOUR_RANGE = 30.0

# why a slot was cleared in the last step (Sim.done_kind)
DONE_NONE, DONE_EXIT, DONE_WRONG, DONE_CRASH, DONE_OFFROAD, DONE_TIMEOUT = 0, 1, 2, 3, 4, 5
DONE_NAMES = ("none", "exit", "wrong_exit", "crash", "offroad", "timeout")

ARM_U = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])  # outward unit vector of arm k


@dataclass
class SimConfig:
    L: float = 50.0  # arm length (spawn/exit distance from centre)
    W: float = 3.5  # half road width
    Rc: float = 6.0  # curb radius of the four inner corners
    dt: float = 0.1
    v_max: float = 14.0  # m/s (~50 km/h)
    acc_max: float = 3.0
    brake_max: float = 7.0
    kappa_max: float = 0.2  # 1 / minimum turning radius
    # steering rate limit: the curvature action is a target; the actual curvature moves
    # towards it by at most kappa_rate per second (0.4 = lock to lock in 1 s)
    kappa_rate: float = 0.4
    spawn_speed: float = 8.0
    spawn_clearance: float = 7.0
    # extra spawn clearance (s) per m/s of closing speed between the new car and an
    # existing one in its lateral corridor (no spawns head-on into exiting cars or
    # into the back of a queue)
    spawn_headway: float = 1.5
    max_age: float = 60.0  # cars older than this are removed (counted as timeout)
    n_slots: int = 32
    n_signal: int = 2  # dimensions of the free "light" signal
    demand: float = 0.15  # arrivals per second per arm
    # if set (lo, hi): every traffic stream draws its own demand uniformly at reset
    # (one batch mixes densities); cfg.demand is then ignored
    demand_range: tuple | None = None
    turn_probs: tuple = (1 / 3, 1 / 3, 1 / 3)  # right, straight, left
    signal_noise: float = 0.0  # std of gaussian noise on received signals
    # perception range (m) of the neighbour block (positions stay scaled by NEIGHBOUR_RANGE)
    neighbour_vis: float = 30.0
    signal_enabled: bool = True
    # reward weights. Per car (Sim.car_reward); the team reward returned by step() is
    # their sum over the cars of an environment minus the queue cost.
    r_exit: float = 1.0
    r_wrong_exit: float = -1.0
    r_crash: float = -2.0  # to every car involved in a collision
    r_offroad: float = -2.0
    r_timeout: float = -1.0
    r_progress: float = 0.01  # per metre of route-aware distance to the target gate (route_dist)
    r_time: float = 0.0  # cost per car per second on the road
    r_queue: float = 0.0  # team-level cost per queued (not yet entered) car per second


# own state: canonical pos (2), heading (2), speed, manoeuvre one-hot (3), goal vector (2),
# rays (N_RAYS), curvature kappa / kappa_max; the neighbour block is appended below
OBS_DIM = 4 + 1 + 3 + 2 + N_RAYS + 1


def obs_dim(cfg: SimConfig) -> int:
    return OBS_DIM + K_NEIGHBOURS * (6 + cfg.n_signal)


def act_dim(cfg: SimConfig) -> int:
    return 2 + cfg.n_signal


def rot(k):
    """Cos/sin of rotating by k * 90 degrees (k may be an array)."""
    a = np.asarray(k) * (np.pi / 2)
    return np.rint(np.cos(a)), np.rint(np.sin(a))


def corner_dist(x, y, W, Rc):
    """Distance-like measure to the off-road corner blocks: a point is off-road iff
    the result is < Rc (the blocks are quadrants with rounded inner corners)."""
    return np.hypot(np.maximum(W + Rc - np.abs(x), 0.0), np.maximum(W + Rc - np.abs(y), 0.0))


def offroad_point(x, y, W, Rc):
    return corner_dist(x, y, W, Rc) < Rc


def _disc_entry(px, py, du, dv, R):
    """First t >= 0 at which the ray (unit direction) enters the disc of radius R around
    the origin, given the ray start (px, py) relative to the disc centre."""
    bq = px * du + py * dv
    disc = bq * bq - (px * px + py * py - R * R)
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0, t1 = -bq - sq, -bq + sq
    return np.where((disc >= 0) & (t1 >= 0), np.maximum(t0, 0.0), np.inf)


def ray_edge_dist(x, y, dx, dy, W, Rc, tmax=RAY_RANGE):
    """Exact distance along the ray (x, y) + t (dx, dy) (unit direction) to the first
    off-road point, capped at tmax. The off-road set is the union of four corner
    blocks; the block of quadrant (sx, sy) is every point within Rc of the quarter
    plane {sx x >= W + Rc, sy y >= W + Rc}, i.e. two quarter planes plus a disc."""
    c = W + Rc
    x, y, dx, dy = np.broadcast_arrays(*(np.asarray(z, dtype=float) for z in (x, y, dx, dy)))
    # the four quadrants along a new leading axis: (sx, sy) = (+,+), (+,-), (-,+), (-,-)
    sx = np.array([1.0, 1.0, -1.0, -1.0]).reshape((4,) + (1,) * x.ndim)
    sy = np.array([1.0, -1.0, 1.0, -1.0]).reshape((4,) + (1,) * x.ndim)
    u, v, du, dv = sx * x, sy * y, sx * dx, sy * dy
    # two quarter planes {u >= W, v >= c} and {u >= c, v >= W} ; a zero direction
    # component is treated as +1e-12, which keeps the formula branch-free
    du_ = np.where(np.abs(du) < 1e-12, 1e-12, du)
    dv_ = np.where(np.abs(dv) < 1e-12, 1e-12, dv)
    iu, iv = 1.0 / du_, 1.0 / dv_
    tWu, tcu, tWv, tcv = (W - u) * iu, (c - u) * iu, (W - v) * iv, (c - v) * iv
    pu, pv = du_ > 0, dv_ > 0
    ninf, inf = -np.inf, np.inf
    lo1 = np.maximum(np.maximum(np.where(pu, tWu, ninf), np.where(pv, tcv, ninf)), 0.0)
    hi1 = np.minimum(np.where(pu, inf, tWu), np.where(pv, inf, tcv))
    lo2 = np.maximum(np.maximum(np.where(pu, tcu, ninf), np.where(pv, tWv, ninf)), 0.0)
    hi2 = np.minimum(np.where(pu, inf, tcu), np.where(pv, inf, tWv))
    t = np.minimum(np.where(lo1 <= hi1, lo1, inf), np.where(lo2 <= hi2, lo2, inf))
    t = np.minimum(t, _disc_entry(u - c, v - c, du, dv, Rc))
    return np.minimum(t.min(0), tmax)


def route_dist(cfg, arm, man, x, y):
    """Route-aware distance to the target gate (no lanes): in the target arm the
    remaining arm length; in the box the distance to the target mouth centre;
    in any other arm via that arm's mouth centre. Unlike the euclidean gate
    distance it does not reward cutting through the curb corners."""
    c0 = cfg.W + cfg.Rc
    co, si = rot(-arm)
    cx = co * x - si * y
    cy = si * x + co * y
    ut = ARM_U[man]  # canonical target arm index == manoeuvre
    in_box = (np.abs(cx) <= c0) & (np.abs(cy) <= c0)
    a = np.argmax(np.stack([-cy, cx, cy, -cx], -1), -1)
    ua = ARM_U[a]
    d_target_arm = cfg.L - (cx * ut[..., 0] + cy * ut[..., 1])
    d_box = np.hypot(cx - c0 * ut[..., 0], cy - c0 * ut[..., 1]) + cfg.L - c0
    d_other = (
        np.hypot(cx - c0 * ua[..., 0], cy - c0 * ua[..., 1])
        + c0 * np.hypot(ua[..., 0] - ut[..., 0], ua[..., 1] - ut[..., 1])
        + cfg.L
        - c0
    )
    return np.where(in_box, d_box, np.where(a == man, d_target_arm, d_other))


def free_flow_time(cfg: SimConfig, manoeuvre):
    """Approximate travel time of an unobstructed car at v_max (for delay metrics)."""
    straight_len = 2 * cfg.L - 2.5
    c = cfg.W + cfg.Rc
    arm_len = 2 * (cfg.L - c) - 2.5
    length = np.where(
        manoeuvre == STRAIGHT,
        straight_len,
        np.where(manoeuvre == RIGHT, arm_len + np.pi / 2 * (c - cfg.W / 2), arm_len + np.pi / 2 * (c + cfg.W / 2)),
    )
    return length / cfg.v_max


@dataclass
class Stats:
    """Accumulated per-environment counters plus per-vehicle travel records."""

    B: int
    arrived: np.ndarray = None
    spawned: np.ndarray = None
    exits: np.ndarray = None
    wrong_exits: np.ndarray = None
    crashes: np.ndarray = None  # number of cars involved in car-car collisions
    offroad: np.ndarray = None
    timeouts: np.ndarray = None
    travel: list = field(default_factory=list)  # per env: list of (time, free_flow_time)

    def __post_init__(self):
        for name in ("arrived", "spawned", "exits", "wrong_exits", "crashes", "offroad", "timeouts"):
            setattr(self, name, np.zeros(self.B, dtype=np.int64))
        self.travel = [[] for _ in range(self.B)]


class Sim:
    def __init__(self, cfg: SimConfig, n_envs: int, seed: int = 0, n_scenarios: int | None = None):
        """n_scenarios: number of distinct random traffic streams. Environment b uses
        stream b % n_scenarios, so several policies can be compared on identical
        traffic (common random numbers). Defaults to one stream per environment."""
        self.cfg = cfg
        self.B = n_envs
        self.S = n_scenarios or n_envs
        self.scen = np.arange(self.B) % self.S
        self.reset(seed)

    def reset(self, seed: int = 0):
        """Empty roads and queues, fresh traffic streams (and per-stream demand)."""
        cfg = self.cfg
        self.rng = np.random.default_rng(seed)  # traffic stream (arrivals only)
        self.noise_rng = np.random.default_rng([seed, 1])  # signal noise (keeps traffic CRN-aligned)
        B, N = self.B, cfg.n_slots
        # demand per environment (from its traffic stream); None = use cfg.demand live
        self._stream_demand = self.demand = None
        if cfg.demand_range is not None:
            lo, hi = cfg.demand_range
            self._stream_demand = self.rng.uniform(lo, hi, self.S)
            self.demand = self._stream_demand[self.scen]
        self.alive = np.zeros((B, N), dtype=bool)
        self.x = np.zeros((B, N))
        self.y = np.zeros((B, N))
        self.th = np.zeros((B, N))
        self.v = np.zeros((B, N))
        self.kappa = np.zeros((B, N))  # actual curvature (rate limited)
        self.sig = np.zeros((B, N, cfg.n_signal))
        self.arm = np.zeros((B, N), dtype=np.int64)
        self.man = np.zeros((B, N), dtype=np.int64)  # manoeuvre 1..3
        self.age = np.zeros((B, N))
        self.uid = np.full((B, N), -1, dtype=np.int64)  # stable id per car (trajectory id for RL)
        self.group = np.zeros((B, N), dtype=np.int64)  # policy group (for mixed populations)
        self.queue = np.zeros((B, 4), dtype=np.int64)  # cars waiting to enter per arm
        self.queue_lat = np.zeros((B, 4))  # lateral offset of the head of each queue
        self.queue_man = np.ones((B, 4), dtype=np.int64)
        self.queue_grp = np.zeros((B, 4), dtype=np.int64)
        # FIFO of queued cars' attributes (drawn at arrival time, so CRN holds even
        # when two policies release cars from the queue at different times)
        self._qcap = 64
        self._qhead = np.zeros((B, 4), dtype=np.int64)
        self._qbuf = np.zeros((B, 4, self._qcap, 3))  # lat, man, grp
        self.next_uid = 0
        self.t = 0.0
        self.stats = Stats(B)
        self.events = []  # (b, kind, x, y) of the last step, for rendering
        self.group_probs = getattr(self, "group_probs", np.array([1.0]))
        self._prev_dist = np.zeros((B, N))
        # per-step outputs for RL (valid after step()):
        self.car_reward = np.zeros((B, N))  # reward of the car that occupied the slot during the step
        self.done_kind = np.zeros((B, N), dtype=np.int8)  # DONE_* why the slot was cleared (0 = not)
        self.done_uid = np.full((B, N), -1, dtype=np.int64)  # uid of the car that was removed
        self.queue_cost = np.zeros(B)  # team-level queue term (already subtracted from step()'s reward)

    # ------------------------------------------------------------------ geometry
    @property
    def tgt(self):
        return (self.arm + self.man) % 4

    def gate_xy(self):
        """Centre of the target arm's exit gate for every slot."""
        c, s = rot(self.tgt)
        # arm 0 gate is (0, -L); rotate by tgt * 90deg
        return s * self.cfg.L, -c * self.cfg.L

    def dist_to_gate(self):
        """Route-aware remaining distance of every slot (see route_dist)."""
        return route_dist(self.cfg, self.arm, self.man, self.x, self.y)

    # ------------------------------------------------------------------ observe
    def observe(self) -> np.ndarray:
        """(B, N, obs_dim) observations; rows of empty slots are zero. Features are only
        computed for occupied slots (typically a small fraction of N)."""
        cfg = self.cfg
        B, N = self.B, cfg.n_slots
        L, W = cfg.L, cfg.W
        obs = np.zeros((B, N, obs_dim(cfg)))
        bi, ni = np.nonzero(self.alive)
        if len(bi) == 0:
            return obs
        x, y, th, arm, man = self.x[bi, ni], self.y[bi, ni], self.th[bi, ni], self.arm[bi, ni], self.man[bi, ni]
        c, s = rot(-arm)
        # canonical frame: rotate world so that the entry arm is south
        cth = th - arm * (np.pi / 2)
        gc, gs = rot((arm + man) % 4)
        dx, dy = gs * L - x, -gc * L - y  # to the centre of the target gate
        ct, st = np.cos(th), np.sin(th)
        o = obs[bi, ni]
        o[:, 0] = (c * x - s * y) / L
        o[:, 1] = (s * x + c * y) / L
        o[:, 2] = np.cos(cth)
        o[:, 3] = np.sin(cth)
        o[:, 4] = self.v[bi, ni] / cfg.v_max
        o[:, 5] = man == RIGHT
        o[:, 6] = man == STRAIGHT
        o[:, 7] = man == LEFT
        o[:, 8] = (ct * dx + st * dy) / L
        o[:, 9] = (-st * dx + ct * dy) / L
        # rays to the road edge in the ego frame (exact distance, capped)
        ang = th[:, None] + RAY_ANGLES  # (M,R)
        o[:, 10 : 10 + N_RAYS] = ray_edge_dist(x[:, None], y[:, None], np.cos(ang), np.sin(ang), W, cfg.Rc) / RAY_RANGE
        o[:, 10 + N_RAYS] = self.kappa[bi, ni] / cfg.kappa_max
        # neighbours: K nearest other cars within range, sorted by distance
        ddx = self.x[bi] - x[:, None]  # (M, N) = x_j - x_i
        ddy = self.y[bi] - y[:, None]
        d = np.hypot(ddx, ddy)
        valid = self.alive[bi] & (d < cfg.neighbour_vis)
        valid[np.arange(len(bi)), ni] = False
        d = np.where(valid, d, np.inf)
        K = min(K_NEIGHBOURS, N)
        if N > K:
            idx = np.argpartition(d, K - 1, axis=-1)[:, :K]
            idx = np.take_along_axis(idx, np.argsort(np.take_along_axis(d, idx, -1), axis=-1), -1)
        else:
            idx = np.argsort(d, axis=-1)
        present = np.take_along_axis(valid, idx, -1)
        nx = np.take_along_axis(ddx, idx, -1)
        ny = np.take_along_axis(ddy, idx, -1)
        b2 = bi[:, None]
        nth = self.th[b2, idx] - th[:, None]
        blk = 6 + cfg.n_signal
        f = np.zeros((len(bi), K, blk))
        ct2, st2 = ct[:, None], st[:, None]
        f[..., 0] = (ct2 * nx + st2 * ny) / NEIGHBOUR_RANGE
        f[..., 1] = (-st2 * nx + ct2 * ny) / NEIGHBOUR_RANGE
        f[..., 2] = np.cos(nth)
        f[..., 3] = np.sin(nth)
        f[..., 4] = self.v[b2, idx] / cfg.v_max
        f[..., 5] = 1.0
        if cfg.signal_enabled:
            f[..., 6:] = self.sig[b2, idx]
            if cfg.signal_noise > 0:
                f[..., 6:] += self.noise_rng.normal(0, cfg.signal_noise, f[..., 6:].shape)
        f *= present[..., None]
        o[:, OBS_DIM : OBS_DIM + K * blk] = f.reshape(len(bi), -1)
        obs[bi, ni] = o
        return obs

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray) -> np.ndarray:
        """Advance one timestep. action: (B, N, act_dim) in [-1, 1]. Returns the team
        reward (B,) = car_reward.sum(1) - queue_cost. Also sets, per slot, car_reward
        (reward of the car that drove in the slot this step, incl. its terminal reward),
        done_kind (DONE_*: why the slot was cleared) and done_uid (uid of the removed
        car). A slot can be cleared and refilled by a new car (new uid) in one step."""
        cfg = self.cfg
        a = np.clip(action, -1.0, 1.0)
        alive = self.alive
        acc = np.where(a[..., 0] > 0, a[..., 0] * cfg.acc_max, a[..., 0] * cfg.brake_max)
        dk = cfg.kappa_rate * cfg.dt
        self.kappa = np.where(alive, self.kappa + np.clip(a[..., 1] * cfg.kappa_max - self.kappa, -dk, dk), 0.0)
        self.sig = np.where(alive[..., None], a[..., 2:], 0.0)
        v = np.clip(self.v + acc * cfg.dt, 0.0, cfg.v_max)
        th = self.th + v * self.kappa * cfg.dt
        self.x = np.where(alive, self.x + v * np.cos(th) * cfg.dt, self.x)
        self.y = np.where(alive, self.y + v * np.sin(th) * cfg.dt, self.y)
        self.th = np.where(alive, (th + np.pi) % (2 * np.pi) - np.pi, self.th)
        self.v = np.where(alive, v, 0.0)
        self.age += cfg.dt * alive
        self.t += cfg.dt
        self.events = []
        self.done_kind = np.zeros(alive.shape, dtype=np.int8)
        self.done_uid = np.where(alive, self.uid, -1)

        cr = np.where(alive, -cfg.r_time * cfg.dt, 0.0)
        dist = self.dist_to_gate()
        cr += np.where(alive, cfg.r_progress * (self._prev_dist - dist), 0.0)
        self._prev_dist = dist

        # exits: beyond L in some arm
        out = alive & ((np.abs(self.x) > cfg.L) | (np.abs(self.y) > cfg.L))
        if out.any():
            exit_arm = np.argmax(np.stack([-self.y, self.x, self.y, -self.x], -1), -1)
            ok = out & (exit_arm == self.tgt)
            wrong = out & ~ok
            cr += cfg.r_exit * ok + cfg.r_wrong_exit * wrong
            self.done_kind[ok] = DONE_EXIT
            self.done_kind[wrong] = DONE_WRONG
            self.stats.exits += ok.sum(1)
            self.stats.wrong_exits += wrong.sum(1)
            ffs = free_flow_time(cfg, self.man)
            for b, n in zip(*np.nonzero(ok)):
                self.stats.travel[b].append((self.age[b, n], ffs[b, n]))
            self._remove(out)
            alive = self.alive

        # off-road: any disc overlapping the complement of the road
        ox, oy = self._discs()
        off = (corner_dist(ox, oy, cfg.W, cfg.Rc) < cfg.Rc + DISC_R).any(-1) & alive
        # collisions between cars (both cars of a pair are marked)
        col = self._collisions(ox, oy, alive)
        crash = col & ~off
        for kind, m in (("offroad", off), ("crash", crash)):
            for b, n in zip(*np.nonzero(m)):
                self.events.append((b, kind, self.x[b, n], self.y[b, n]))
        cr += cfg.r_offroad * off + cfg.r_crash * crash
        self.done_kind[off] = DONE_OFFROAD
        self.done_kind[crash] = DONE_CRASH
        self.stats.offroad += off.sum(1)
        self.stats.crashes += crash.sum(1)
        self._remove(off | crash)

        timeout = self.alive & (self.age > cfg.max_age)
        cr += cfg.r_timeout * timeout
        self.done_kind[timeout] = DONE_TIMEOUT
        self.stats.timeouts += timeout.sum(1)
        self._remove(timeout)

        self._arrivals()
        self.car_reward = cr
        self.queue_cost = cfg.r_queue * cfg.dt * self.queue.sum(1)
        return cr.sum(1) - self.queue_cost

    def _discs(self):
        ct, st = np.cos(self.th)[..., None], np.sin(self.th)[..., None]
        return self.x[..., None] + ct * DISC_OFFSETS, self.y[..., None] + st * DISC_OFFSETS

    def _collisions(self, ox, oy, alive):
        N = self.cfg.n_slots
        dcx = self.x[:, :, None] - self.x[:, None, :]
        dcy = self.y[:, :, None] - self.y[:, None, :]
        near = (dcx**2 + dcy**2 < (2 * 1.4 + 2 * DISC_R) ** 2) & alive[:, :, None] & alive[:, None, :]
        near &= ~np.eye(N, dtype=bool)[None]
        if not near.any():
            return np.zeros_like(alive)
        b, i, j = np.nonzero(near)
        dx = ox[b, i][:, :, None] - ox[b, j][:, None, :]
        dy = oy[b, i][:, :, None] - oy[b, j][:, None, :]
        hit = ((dx**2 + dy**2) < (2 * DISC_R) ** 2).any((1, 2))
        col = np.zeros_like(alive)
        col[b[hit], i[hit]] = True
        return col

    def _remove(self, mask):
        self.alive &= ~mask
        self.v[mask] = 0.0
        self.sig[mask] = 0.0
        self.age[mask] = 0.0
        self.uid[mask] = -1
        # park dead cars far away so they never interact
        self.x[mask] = 1e4
        self.y[mask] = 1e4

    # ------------------------------------------------------------------ spawning
    def _arrivals(self):
        cfg = self.cfg
        S = self.S
        # draw every random number unconditionally so streams stay aligned (CRN)
        dem = cfg.demand if self._stream_demand is None else self._stream_demand[:, None]
        arrive = self.rng.random((S, 4)) < dem * cfg.dt
        lat = self.rng.uniform(-(cfg.W - 1.2), cfg.W - 1.2, (S, 4))
        man = self.rng.choice([RIGHT, STRAIGHT, LEFT], size=(S, 4), p=np.asarray(cfg.turn_probs) / sum(cfg.turn_probs))
        grp = self.rng.choice(len(self.group_probs), size=(S, 4), p=self.group_probs)
        arrive, lat, man, grp = arrive[self.scen], lat[self.scen], man[self.scen], grp[self.scen]
        if (self.queue + arrive).max() > self._qcap:  # grow the FIFO (keeps order)
            new = np.zeros((self.B, 4, 2 * self._qcap, 3))
            for j in range(self._qcap):
                new[:, :, j] = np.take_along_axis(self._qbuf, ((self._qhead + j) % self._qcap)[..., None, None], 2)[:, :, 0]
            self._qbuf, self._qhead, self._qcap = new, np.zeros_like(self._qhead), 2 * self._qcap
        bi, ki = np.nonzero(arrive)
        slot = (self._qhead[bi, ki] + self.queue[bi, ki]) % self._qcap
        self._qbuf[bi, ki, slot] = np.stack([lat[bi, ki], man[bi, ki], grp[bi, ki]], -1)
        self.queue += arrive
        self.stats.arrived += arrive.sum(1)
        head = np.take_along_axis(self._qbuf, self._qhead[..., None, None], 2)[:, :, 0]
        self.queue_lat, self.queue_man, self.queue_grp = head[..., 0], head[..., 1].astype(np.int64), head[..., 2].astype(np.int64)
        # velocities of existing cars (for the closing-speed spawn clearance)
        vx, vy = self.v * np.cos(self.th), self.v * np.sin(self.th)
        spawned = np.zeros(self.alive.shape, dtype=bool)
        for k in range(4):
            want = self.queue[:, k] > 0
            if not want.any():
                continue
            c, s = rot(k)
            u = self.queue_lat[:, k]
            # canonical spawn point (u, -L + 2.5) rotated by k*90deg
            sx = c * u - s * (-cfg.L + 2.5)
            sy = s * u + c * (-cfg.L + 2.5)
            h = np.pi / 2 + k * np.pi / 2  # canonical heading (north) rotated by k*90deg
            dx, dy = self.x - sx[:, None], self.y - sy[:, None]
            dist = np.hypot(dx, dy)
            # closing speed along the line of sight between the new car and car j
            rvx, rvy = cfg.spawn_speed * np.cos(h) - vx, cfg.spawn_speed * np.sin(h) - vy
            closing = np.maximum((rvx * dx + rvy * dy) / np.maximum(dist, 1e-6), 0.0)
            # outside the new car's lateral corridor only its own speed counts (the other
            # car is treated as standing: it may be queued in the lane the new car joins)
            own = np.maximum(cfg.spawn_speed * (np.cos(h) * dx + np.sin(h) * dy) / np.maximum(dist, 1e-6), 0.0)
            lat_off = np.abs(-np.sin(h) * dx + np.cos(h) * dy)
            closing = np.where(lat_off < 2 * DISC_R + SPAWN_LAT_MARGIN, closing, own)
            margin = cfg.spawn_headway * closing
            clear = (dist > cfg.spawn_clearance + margin) | ~self.alive
            ok = want & clear.all(1) & (~self.alive).any(1)
            for b in np.nonzero(ok)[0]:
                n = np.argmin(self.alive[b])
                self.alive[b, n] = True
                self.x[b, n], self.y[b, n] = sx[b], sy[b]
                self.th[b, n] = np.arctan2(np.sin(h), np.cos(h))
                self.v[b, n] = cfg.spawn_speed
                self.kappa[b, n] = 0.0
                self.arm[b, n] = k
                self.man[b, n] = self.queue_man[b, k]
                self.group[b, n] = self.queue_grp[b, k]
                self.age[b, n] = 0.0
                self.uid[b, n] = self.next_uid
                self.next_uid += 1
                self.queue[b, k] -= 1
                self._qhead[b, k] = (self._qhead[b, k] + 1) % self._qcap
                self.stats.spawned[b] += 1
                spawned[b, n] = True
                hd = self._qbuf[b, k, self._qhead[b, k]]
                self.queue_lat[b, k], self.queue_man[b, k], self.queue_grp[b, k] = hd[0], int(hd[1]), int(hd[2])
        if spawned.any():
            self._prev_dist = np.where(spawned, self.dist_to_gate(), self._prev_dist)
