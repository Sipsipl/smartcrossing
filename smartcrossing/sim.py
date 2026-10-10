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

N_RAYS = 5
RAY_ANGLES = np.radians([-60.0, -30.0, 0.0, 30.0, 60.0])
RAY_STEPS = np.arange(1.0, 21.0)  # metres
K_NEIGHBOURS = 4
NEIGHBOUR_RANGE = 30.0


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
    spawn_speed: float = 8.0
    spawn_clearance: float = 7.0
    max_age: float = 60.0  # cars older than this are removed (counted as timeout)
    n_slots: int = 32
    n_signal: int = 2  # dimensions of the free "light" signal
    demand: float = 0.15  # arrivals per second per arm
    turn_probs: tuple = (1 / 3, 1 / 3, 1 / 3)  # right, straight, left
    signal_noise: float = 0.0  # std of gaussian noise on received signals
    signal_enabled: bool = True
    # reward weights (team fitness)
    r_exit: float = 1.0
    r_wrong_exit: float = -1.0
    r_crash: float = -2.0
    r_offroad: float = -2.0
    r_timeout: float = -1.0
    r_progress: float = 0.01  # per metre towards the target gate


OBS_DIM = 4 + 1 + 3 + 2 + N_RAYS  # own state; neighbour block appended below


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
        self.rng = np.random.default_rng(seed)
        B, N = self.B, cfg.n_slots
        self.alive = np.zeros((B, N), dtype=bool)
        self.x = np.zeros((B, N))
        self.y = np.zeros((B, N))
        self.th = np.zeros((B, N))
        self.v = np.zeros((B, N))
        self.kappa = np.zeros((B, N))
        self.sig = np.zeros((B, N, cfg.n_signal))
        self.arm = np.zeros((B, N), dtype=np.int64)
        self.man = np.zeros((B, N), dtype=np.int64)  # manoeuvre 1..3
        self.age = np.zeros((B, N))
        self.uid = np.full((B, N), -1, dtype=np.int64)  # stable id for rendering
        self.group = np.zeros((B, N), dtype=np.int64)  # policy group (for mixed populations)
        self.queue = np.zeros((B, 4), dtype=np.int64)  # cars waiting to enter per arm
        self.queue_lat = np.zeros((B, 4))  # lateral offset of the head of each queue
        self.queue_man = np.ones((B, 4), dtype=np.int64)
        self.queue_grp = np.zeros((B, 4), dtype=np.int64)
        self.next_uid = 0
        self.t = 0.0
        self.stats = Stats(B)
        self.events = []  # (b, kind, x, y) of the last step, for rendering
        self.group_probs = np.array([1.0])
        self._prev_dist = np.zeros((B, N))

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
        gx, gy = self.gate_xy()
        return np.hypot(self.x - gx, self.y - gy)

    # ------------------------------------------------------------------ observe
    def observe(self) -> np.ndarray:
        cfg = self.cfg
        B, N = self.B, cfg.n_slots
        L, W = cfg.L, cfg.W
        c, s = rot(-self.arm)
        # canonical frame: rotate world so that the entry arm is south
        cx = c * self.x - s * self.y
        cy = s * self.x + c * self.y
        cth = self.th - self.arm * (np.pi / 2)
        man_onehot = np.stack([self.man == RIGHT, self.man == STRAIGHT, self.man == LEFT], -1).astype(float)
        gx, gy = self.gate_xy()
        dx, dy = gx - self.x, gy - self.y
        ct, st = np.cos(self.th), np.sin(self.th)
        gex = (ct * dx + st * dy) / L
        gey = (-st * dx + ct * dy) / L
        # rays to the road edge in the ego frame
        ang = self.th[..., None] + RAY_ANGLES  # (B,N,R)
        px = self.x[..., None, None] + np.cos(ang)[..., None] * RAY_STEPS
        py = self.y[..., None, None] + np.sin(ang)[..., None] * RAY_STEPS
        hit = offroad_point(px, py, W, cfg.Rc)
        first = np.where(hit.any(-1), hit.argmax(-1), len(RAY_STEPS))
        rays = first / len(RAY_STEPS)
        own = np.concatenate(
            [
                np.stack([cx / L, cy / L, np.cos(cth), np.sin(cth), self.v / cfg.v_max], -1),
                man_onehot,
                np.stack([gex, gey], -1),
                rays,
            ],
            -1,
        )
        # neighbours
        ddx = self.x[:, None, :] - self.x[:, :, None]  # (B, i, j) = x_j - x_i
        ddy = self.y[:, None, :] - self.y[:, :, None]
        d = np.hypot(ddx, ddy)
        valid = self.alive[:, None, :] & self.alive[:, :, None] & ~np.eye(N, dtype=bool)[None]
        valid &= d < NEIGHBOUR_RANGE
        d = np.where(valid, d, np.inf)
        idx = np.argsort(d, axis=-1)[..., :K_NEIGHBOURS]  # (B,N,K)
        present = np.take_along_axis(valid, idx, -1)
        nx = np.take_along_axis(ddx, idx, -1)
        ny = np.take_along_axis(ddy, idx, -1)
        bi = np.arange(B)[:, None, None]
        nth = self.th[bi, idx] - self.th[..., None]
        nv = self.v[bi, idx] / cfg.v_max
        nsig = self.sig[bi, idx]  # (B,N,K,S)
        if not cfg.signal_enabled:
            nsig = np.zeros_like(nsig)
        elif cfg.signal_noise > 0:
            nsig = nsig + self.rng.normal(0, cfg.signal_noise, nsig.shape)
        ct3, st3 = ct[..., None], st[..., None]
        rel_x = (ct3 * nx + st3 * ny) / NEIGHBOUR_RANGE
        rel_y = (-st3 * nx + ct3 * ny) / NEIGHBOUR_RANGE
        feats = np.concatenate(
            [np.stack([rel_x, rel_y, np.cos(nth), np.sin(nth), nv, present.astype(float)], -1), nsig], -1
        )
        feats *= present[..., None]
        obs = np.concatenate([own, feats.reshape(B, N, -1)], -1)
        return obs * self.alive[..., None]

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray) -> np.ndarray:
        """Advance one timestep. action: (B, N, act_dim) in [-1, 1]. Returns team reward (B,)."""
        cfg = self.cfg
        a = np.clip(action, -1.0, 1.0)
        alive = self.alive
        acc = np.where(a[..., 0] > 0, a[..., 0] * cfg.acc_max, a[..., 0] * cfg.brake_max)
        self.kappa = a[..., 1] * cfg.kappa_max
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

        reward = np.zeros(self.B)
        dist = self.dist_to_gate()
        reward += cfg.r_progress * ((self._prev_dist - dist) * alive).sum(1)
        self._prev_dist = dist

        # exits: beyond L in some arm
        out = alive & ((np.abs(self.x) > cfg.L) | (np.abs(self.y) > cfg.L))
        if out.any():
            exit_arm = np.argmax(np.stack([-self.y, self.x, self.y, -self.x], -1), -1)
            ok = out & (exit_arm == self.tgt)
            wrong = out & ~ok
            reward += cfg.r_exit * ok.sum(1) + cfg.r_wrong_exit * wrong.sum(1)
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
        # collisions between cars
        col = self._collisions(ox, oy, alive)
        crash = col & ~off
        for kind, m in (("offroad", off), ("crash", crash)):
            for b, n in zip(*np.nonzero(m)):
                self.events.append((b, kind, self.x[b, n], self.y[b, n]))
        reward += cfg.r_offroad * off.sum(1) + cfg.r_crash * crash.sum(1)
        self.stats.offroad += off.sum(1)
        self.stats.crashes += crash.sum(1)
        self._remove(off | crash)

        timeout = self.alive & (self.age > cfg.max_age)
        reward += cfg.r_timeout * timeout.sum(1)
        self.stats.timeouts += timeout.sum(1)
        self._remove(timeout)

        self._arrivals()
        return reward

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
        arrive = self.rng.random((S, 4)) < cfg.demand * cfg.dt
        lat = self.rng.uniform(-(cfg.W - 1.2), cfg.W - 1.2, (S, 4))
        man = self.rng.choice([RIGHT, STRAIGHT, LEFT], size=(S, 4), p=np.asarray(cfg.turn_probs) / sum(cfg.turn_probs))
        grp = self.rng.choice(len(self.group_probs), size=(S, 4), p=self.group_probs)
        arrive, lat, man, grp = arrive[self.scen], lat[self.scen], man[self.scen], grp[self.scen]
        empty = self.queue == 0
        self.queue_lat = np.where(empty & arrive, lat, self.queue_lat)
        self.queue_man = np.where(empty & arrive, man, self.queue_man)
        self.queue_grp = np.where(empty & arrive, grp, self.queue_grp)
        self.queue += arrive
        self.stats.arrived += arrive.sum(1)
        for k in range(4):
            want = self.queue[:, k] > 0
            if not want.any():
                continue
            c, s = rot(k)
            u = self.queue_lat[:, k]
            # canonical spawn point (u, -L + 2.5) rotated by k*90deg
            sx = c * u - s * (-cfg.L + 2.5)
            sy = s * u + c * (-cfg.L + 2.5)
            clear = (np.hypot(self.x - sx[:, None], self.y - sy[:, None]) > cfg.spawn_clearance) | ~self.alive
            ok = want & clear.all(1) & (~self.alive).any(1)
            for b in np.nonzero(ok)[0]:
                n = np.argmin(self.alive[b])
                self.alive[b, n] = True
                self.x[b, n], self.y[b, n] = sx[b], sy[b]
                h = np.pi / 2 + k * np.pi / 2  # canonical heading (north) rotated by k*90deg
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
                self.stats.spawned[b] += 1
                gx, gy = self.gate_xy()
                self._prev_dist[b, n] = np.hypot(sx[b] - gx[b, n], sy[b] - gy[b, n])
                # the next queued car gets fresh attributes drawn from the same stream
                if self.queue[b, k] > 0:
                    self.queue_lat[b, k] = lat[b, k]
                    self.queue_man[b, k] = man[b, k]
                    self.queue_grp[b, k] = grp[b, k]
