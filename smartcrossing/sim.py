"""Batched 2D simulation of a signal-free four-way intersection.

Geometry: two perpendicular roads of width 2*W crossing at the origin, each arm
reaching out to distance L. There are no lanes, markings or rules: the drivable
area is simply the cross-shaped union of both strips. Default: W = 7 m (14 m wide
roads, room for two cars side by side per direction), curb radius 9 m; the old
7 m crossing is ``SimConfig(**NARROW)``.

Vehicle physics: kinematic unicycle with a speed and curvature limit, a steering-rate
limit and a tyre-grip limit (lateral acceleration v^2 * kappa <= a_lat_max), so tight
turns are only possible at low speed. Cars enter at the end of
an arm (at a random lateral offset, by default in the right half of the road) and must leave
through their target arm. Cars of a car-car crash stay as standing wrecks for cfg.wreck_time
(hazard lights on; driving into one is a crash), off-road cars vanish.

Arms are indexed counter-clockwise: 0 = south, 1 = east, 2 = north, 3 = west.
A car entering from arm k with manoeuvre m in {1: right, 2: straight, 3: left}
must exit through arm (k + m) % 4.

All state lives in arrays of shape (B, N): B independent environments with N
car slots each (``alive``: driving cars; ``wreck``: wrecks, which occupy their slot until
removed). The policy is evaluated outside of this module.
"""

from dataclasses import dataclass, field

import numpy as np

RIGHT, STRAIGHT, LEFT = 1, 2, 3

# Car shape: three discs along the heading axis approximate a 4.6 x 1.8 m box.
DISC_OFFSETS = np.array([-1.4, 0.0, 1.4])
DISC_R = 0.9

SPAWN_LAT_MARGIN = 0.8  # lateral corridor (beyond car width) checked for closing traffic at spawn
SPAWN_BACK = 2.5  # cars spawn this far inside the end of their entry arm
SPAWN_CURB_GAP = 1.2  # lateral spawn offsets keep this distance to the road edge
SPAWN_RIGHT_MIN = 1.0  # spawn_right: smallest offset right of the centre line
SPAWN_LAT_STEP = 0.2  # spacing (m) of the candidate lateral spawn offsets
SPAWN_MAX_PER_STEP = 3  # at most this many cars enter per arm and step (side by side)
SPEED_TAU = 0.5  # s, time constant of the legacy speed_control interface

N_RAYS = 5
RAY_ANGLES = np.radians([-60.0, -30.0, 0.0, 30.0, 60.0])
RAY_RANGE = 20.0  # metres; rays report the exact distance to the road edge, capped here
K_NEIGHBOURS = 4
NEIGHBOUR_RANGE = 30.0  # neighbour positions are divided by this (fixed, independent of the perception range)
POS_SCALE = 10.0  # own canonical position is given in units of 10 m (fixed: independent of L, W)
GOAL_SCALE = 25.0  # goal vectors (exit gate, exit mouth) in units of 25 m
# per neighbour before its lamps: rel. pos (2), rel. heading (2), speed, present, curvature, then
# brake light + hazard lights (cfg.brake_light, the default) or the legacy acceleration (8 features)
NB_FEATURES = 9
NB_FEATURES_LEGACY = 8
BRAKE_ACC = -0.5  # m/s^2: brake lights are on while the actual acceleration is below this

# why a slot was cleared in the last step (Sim.done_kind)
DONE_NONE, DONE_EXIT, DONE_WRONG, DONE_CRASH, DONE_OFFROAD, DONE_TIMEOUT = 0, 1, 2, 3, 4, 5
DONE_NAMES = ("none", "exit", "wrong_exit", "crash", "offroad", "timeout")

ARM_U = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])  # outward unit vector of arm k


LANE_WIDTH = 3.5  # nominal lane width; only used by the rule-based baselines and delay metrics
# the original crossing (one car per direction, no grip limit, no reverse): SimConfig(**NARROW)
NARROW = {"W": 3.5, "Rc": 6.0, "a_lat_max": 0.0, "v_reverse": 0.0}


@dataclass
class SimConfig:
    L: float = 50.0  # arm length (spawn/exit distance from centre)
    W: float = 7.0  # half road width (7 = two cars side by side per direction)
    Rc: float = 9.0  # curb radius of the four inner corners
    dt: float = 0.1
    v_max: float = 14.0  # m/s (~50 km/h)
    acc_max: float = 3.0
    brake_max: float = 7.0
    # slow reverse gear: speed range [-v_reverse, v_max] (0 = no reverse, the old physics).
    # Heading/position use the signed speed (steering left while reversing turns the car
    # clockwise, like a real car); grip uses v^2. Leaving through a wrong arm backwards is a
    # wrong exit like any other.
    v_reverse: float = 2.0
    kappa_max: float = 0.2  # 1 / minimum turning radius
    # steering rate limit: the curvature action is a target; the actual curvature moves
    # towards it by at most kappa_rate per second (0.4 = lock to lock in 1 s)
    kappa_rate: float = 0.4
    # tyre grip: the actual curvature is limited to min(kappa_max, a_lat_max / v^2) (after
    # the steering-rate limit). 4 m/s^2 ~ 0.4 g (normal driving); the tightest baseline
    # turn (right turn, r ~ 10.75 m) then allows ~6.6 m/s; at 14 m/s r_min = 49 m.
    # 0 disables the limit (the original physics: r_min = 5 m at any speed).
    a_lat_max: float = 4.0
    # lateral spawn offset: True = only the right half of the road in driving direction (default
    # for everyone since the owner's decision of 2026-10-11); False = uniform over the whole road
    # width (own spot, or uniform among the clear positions if it is blocked; both halves, no side
    # preferred; how the old checkpoints were trained). Same random draws either way (Sim._arrivals).
    spawn_right: bool = True
    spawn_speed: float = 8.0
    spawn_clearance: float = 7.0
    # extra spawn clearance (s) per m/s of closing speed between the new car and an
    # existing one in its lateral corridor (no spawns head-on into exiting cars or
    # into the back of a queue)
    spawn_headway: float = 1.5
    max_age: float = 60.0  # cars older than this are removed (counted as timeout)
    # wrecks: cars of a car-car crash stay this long (s) as standing obstacles (speed 0, no control,
    # no reward, hazard lights on; their episode ends at the crash). A moving car that hits a wreck
    # crashes too (impact speed = its own speed) and becomes a wreck. 0 = crashed cars vanish at
    # once (the old behaviour). Off-road cars always vanish.
    wreck_time: float = 5.0
    n_slots: int = 32
    n_signal: int = 1  # dimensions of the free "light" signal (old checkpoints: 2)
    demand: float = 0.15  # arrivals per second per arm
    # if set (lo, hi): every traffic stream draws its own demand uniformly at reset
    # (one batch mixes densities); cfg.demand is then ignored
    demand_range: tuple | None = None
    turn_probs: tuple = (1 / 3, 1 / 3, 1 / 3)  # right, straight, left
    signal_noise: float = 0.0  # std of gaussian noise on received signals
    # Per-checkpoint interface settings, kept only so old checkpoints still run as trained
    # (applied by PolicyDriver.load; experiments.md: neither gave a benefit).
    # perception range (m) of the neighbour block (positions stay scaled by NEIGHBOUR_RANGE);
    # 60 in ppo11 / ppo14 / ppo15
    neighbour_vis: float = 30.0
    # ppo16 only: action[0] is a target speed in [0, v_max] reached with
    # acc = clip((v_target - v) / SPEED_TAU, -brake_max, acc_max) (no reverse)
    speed_control: bool = False
    # neighbour block: binary brake light + hazard lights (True) or the legacy continuous
    # acceleration feature (False, checkpoints trained before 2026-10-11; no hazard feature)
    brake_light: bool = True
    signal_enabled: bool = True
    # reward weights. Per car (Sim.car_reward); the team reward returned by step() is
    # their sum over the cars of an environment minus the queue cost.
    r_exit: float = 1.0
    r_wrong_exit: float = -1.0
    # crash / off-road: base + per m/s of impact speed (car-car: |relative velocity| of the
    # pair at contact, max over partners; off-road: own speed). Pure outcome, no blame: every
    # car involved pays its own term. Order kept: exit (+1) > waiting > timeout (-1) > crash
    # (-1.5 at walking pace ... -2.9 at 14 m/s ... -4.3 head-on at 2 x 14 m/s).
    r_crash: float = -1.5  # to every car involved in a collision
    r_crash_speed: float = 0.1  # extra cost per m/s of relative impact speed
    r_offroad: float = -1.5
    r_offroad_speed: float = 0.1  # extra cost per m/s of own speed when leaving the road
    r_timeout: float = -1.0
    r_progress: float = 0.01  # per metre of route-aware distance to the target gate (route_dist)
    r_time: float = 0.0  # cost per car per second on the road
    r_queue: float = 0.0  # team-level cost per queued (not yet entered) car per second


# own state: canonical pos (2, /POS_SCALE), heading (2), speed, manoeuvre one-hot (3),
# ego-frame vectors to the exit gate (2) and to the exit mouth (2) (/GOAL_SCALE), rays (N_RAYS),
# curvature kappa / kappa_max, grip-limited max curvature at the current speed / kappa_max;
# the neighbour blocks are appended below
OBS_DIM = 4 + 1 + 3 + 2 + 2 + N_RAYS + 1 + 1
OBS_RAYS = 12  # index of the first ray


def js_constants() -> dict:
    """Everything the browser port (site/src/js/sim.js) takes from this module: SimConfig
    defaults and the geometry / observation constants. site/build.py writes it to
    js/sim-config.js, so no number is duplicated in JavaScript."""
    defaults = {k: (list(v) if isinstance(v, tuple) else v) for k, v in SimConfig().__dict__.items()}
    return {
        "defaults": defaults,
        "RIGHT": RIGHT, "STRAIGHT": STRAIGHT, "LEFT": LEFT,
        "DISC_OFFSETS": DISC_OFFSETS.tolist(), "DISC_R": DISC_R, "SPAWN_LAT_MARGIN": SPAWN_LAT_MARGIN,
        "SPAWN_BACK": SPAWN_BACK, "SPAWN_CURB_GAP": SPAWN_CURB_GAP, "SPAWN_RIGHT_MIN": SPAWN_RIGHT_MIN,
        "SPAWN_LAT_STEP": SPAWN_LAT_STEP, "SPAWN_MAX_PER_STEP": SPAWN_MAX_PER_STEP, "SPEED_TAU": SPEED_TAU, "N_RAYS": N_RAYS, "RAY_ANGLES": RAY_ANGLES.tolist(), "RAY_RANGE": RAY_RANGE,
        "K_NEIGHBOURS": K_NEIGHBOURS, "NEIGHBOUR_RANGE": NEIGHBOUR_RANGE, "POS_SCALE": POS_SCALE,
        "GOAL_SCALE": GOAL_SCALE, "NB_FEATURES": NB_FEATURES, "NB_FEATURES_LEGACY": NB_FEATURES_LEGACY,
        "BRAKE_ACC": BRAKE_ACC, "LANE_WIDTH": LANE_WIDTH,
        "OBS_DIM": OBS_DIM, "OBS_RAYS": OBS_RAYS, "DONE_NAMES": list(DONE_NAMES), "ARM_U": ARM_U.tolist(),
    }


def nb_features(cfg: SimConfig) -> int:
    """Features per neighbour before its lamps (depends on the brake_light interface)."""
    return NB_FEATURES if cfg.brake_light else NB_FEATURES_LEGACY


def obs_dim(cfg: SimConfig) -> int:
    return OBS_DIM + K_NEIGHBOURS * (nb_features(cfg) + cfg.n_signal)


def wreck_steps(cfg: SimConfig) -> int:
    """How many steps a wreck stays (it is part of the state after this many steps)."""
    return int(round(cfg.wreck_time / cfg.dt))


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


def n_lanes(cfg: SimConfig) -> int:
    """Lanes per direction a W-wide half road has room for (baselines / metrics only)."""
    return max(1, int(round(cfg.W / LANE_WIDTH)))


def lane_offsets(cfg: SimConfig):
    """Lateral offset (right of travel) of the outermost (right) and innermost (left) lane centre."""
    lw = cfg.W / n_lanes(cfg)
    return cfg.W - lw / 2, lw / 2


def free_flow_time(cfg: SimConfig, manoeuvre):
    """Approximate travel time of an unobstructed car (for delay metrics): v_max on the
    arms, the turn arc (right: from the right lane, left: from the left lane) at the
    grip-limited speed min(v_max, sqrt(a_lat_max * r))."""
    straight_len = 2 * cfg.L - SPAWN_BACK
    c = cfg.W + cfg.Rc
    arm_len = 2 * (cfg.L - c) - SPAWN_BACK
    off_r, off_l = lane_offsets(cfg)
    r = np.where(manoeuvre == RIGHT, c - off_r, c + off_l)
    v_arc = np.minimum(cfg.v_max, np.sqrt(cfg.a_lat_max * r)) if cfg.a_lat_max > 0 else cfg.v_max
    turn = arm_len / cfg.v_max + np.pi / 2 * r / v_arc
    return np.where(manoeuvre == STRAIGHT, straight_len / cfg.v_max, turn)


def grip_kappa(cfg: SimConfig, v):
    """Largest curvature the tyres allow at speed v."""
    if cfg.a_lat_max <= 0:
        return np.full(np.shape(v), cfg.kappa_max)
    return np.minimum(cfg.kappa_max, cfg.a_lat_max / np.maximum(np.asarray(v, dtype=float) ** 2, 1e-9))


@dataclass
class Stats:
    """Accumulated per-environment counters plus per-vehicle travel records."""

    B: int
    arrived: np.ndarray = None
    spawned: np.ndarray = None
    exits: np.ndarray = None
    wrong_exits: np.ndarray = None
    crashes: np.ndarray = None  # number of (moving) cars involved in car-car collisions; wrecks are not recounted
    wreck_hits: np.ndarray = None  # of these: cars that hit (at least) one wreck
    offroad: np.ndarray = None
    timeouts: np.ndarray = None
    travel: list = field(default_factory=list)  # per env: list of (time, free_flow_time)

    def __post_init__(self):
        for name in ("arrived", "spawned", "exits", "wrong_exits", "crashes", "wreck_hits", "offroad", "timeouts"):
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
        self.alive = np.zeros((B, N), dtype=bool)  # driving cars (controlled, observed, rewarded)
        self.wreck = np.zeros((B, N), dtype=bool)  # crashed cars standing as obstacles (see cfg.wreck_time)
        self.wreck_left = np.zeros((B, N), dtype=np.int64)  # steps until the wreck is removed
        self.x = np.zeros((B, N))
        self.y = np.zeros((B, N))
        self.th = np.zeros((B, N))
        self.v = np.zeros((B, N))
        self.kappa = np.zeros((B, N))  # actual curvature (rate and grip limited)
        self.acc = np.zeros((B, N))  # actual acceleration of the last step ((v_new - v_old) / dt)
        self.sig = np.zeros((B, N, cfg.n_signal))
        self.arm = np.zeros((B, N), dtype=np.int64)
        self.man = np.zeros((B, N), dtype=np.int64)  # manoeuvre 1..3
        self.age = np.zeros((B, N))
        self.uid = np.full((B, N), -1, dtype=np.int64)  # stable id per car (trajectory id for RL)
        self.group = np.zeros((B, N), dtype=np.int64)  # policy group (for mixed populations)
        self.queue = np.zeros((B, 4), dtype=np.int64)  # cars waiting to enter per arm
        self.queue_u = np.zeros((B, 4))  # lateral draw u of the head of each queue (see _arrivals)
        self.queue_man = np.ones((B, 4), dtype=np.int64)
        self.queue_grp = np.zeros((B, 4), dtype=np.int64)
        # FIFO of queued cars' attributes (drawn at arrival time, so CRN holds even
        # when two policies release cars from the queue at different times)
        self._qcap = 64
        self._qhead = np.zeros((B, 4), dtype=np.int64)
        self._qbuf = np.zeros((B, 4, self._qcap, 3))  # u (lateral draw), man, grp
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
        ut = ARM_U[(arm + man) % 4]  # outward unit vector of the target arm (world)
        c0 = W + cfg.Rc
        dx, dy = L * ut[:, 0] - x, L * ut[:, 1] - y  # to the centre of the target (exit) gate
        mx, my = c0 * ut[:, 0] - x, c0 * ut[:, 1] - y  # to the centre of the target arm's mouth
        ct, st = np.cos(th), np.sin(th)
        o = obs[bi, ni]
        o[:, 0] = (c * x - s * y) / POS_SCALE
        o[:, 1] = (s * x + c * y) / POS_SCALE
        o[:, 2] = np.cos(cth)
        o[:, 3] = np.sin(cth)
        o[:, 4] = self.v[bi, ni] / cfg.v_max
        o[:, 5] = man == RIGHT
        o[:, 6] = man == STRAIGHT
        o[:, 7] = man == LEFT
        o[:, 8] = (ct * dx + st * dy) / GOAL_SCALE
        o[:, 9] = (-st * dx + ct * dy) / GOAL_SCALE
        o[:, 10] = (ct * mx + st * my) / GOAL_SCALE
        o[:, 11] = (-st * mx + ct * my) / GOAL_SCALE
        # rays to the road edge in the ego frame (exact distance, capped)
        ang = th[:, None] + RAY_ANGLES  # (M,R)
        o[:, OBS_RAYS : OBS_RAYS + N_RAYS] = ray_edge_dist(x[:, None], y[:, None], np.cos(ang), np.sin(ang), W, cfg.Rc) / RAY_RANGE
        o[:, OBS_RAYS + N_RAYS] = self.kappa[bi, ni] / cfg.kappa_max
        o[:, OBS_RAYS + N_RAYS + 1] = grip_kappa(cfg, self.v[bi, ni]) / cfg.kappa_max
        # neighbours: K nearest other cars (driving or wrecked) within range, sorted by distance
        ddx = self.x[bi] - x[:, None]  # (M, N) = x_j - x_i
        ddy = self.y[bi] - y[:, None]
        d = np.hypot(ddx, ddy)
        valid = (self.alive | self.wreck)[bi] & (d < cfg.neighbour_vis)
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
        nbf = nb_features(cfg)
        blk = nbf + cfg.n_signal
        f = np.zeros((len(bi), K, blk))
        ct2, st2 = ct[:, None], st[:, None]
        f[..., 0] = (ct2 * nx + st2 * ny) / NEIGHBOUR_RANGE
        f[..., 1] = (-st2 * nx + ct2 * ny) / NEIGHBOUR_RANGE
        f[..., 2] = np.cos(nth)
        f[..., 3] = np.sin(nth)
        f[..., 4] = self.v[b2, idx] / cfg.v_max
        f[..., 5] = 1.0
        f[..., 6] = self.kappa[b2, idx] / cfg.kappa_max  # visible: how hard it is turning
        if cfg.brake_light:
            f[..., 7] = self.acc[b2, idx] < BRAKE_ACC  # visible: brake lights
            f[..., 8] = self.wreck[b2, idx]  # visible: hazard lights (a wreck)
        else:
            f[..., 7] = self.acc[b2, idx] / cfg.brake_max  # legacy interface: acceleration
        if cfg.signal_enabled:
            f[..., nbf:] = self.sig[b2, idx]
            if cfg.signal_noise > 0:
                f[..., nbf:] += self.noise_rng.normal(0, cfg.signal_noise, f[..., nbf:].shape)
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
        if cfg.speed_control:
            v_t = (a[..., 0] + 1.0) * 0.5 * cfg.v_max
            acc = np.clip((v_t - self.v) / SPEED_TAU, -cfg.brake_max, cfg.acc_max)
            v = np.clip(self.v + acc * cfg.dt, 0.0, cfg.v_max)  # (no reverse in this interface)
        else:
            # pedal: + accelerates forward (acc_max) / brakes a reversing car (brake_max);
            # - brakes a forward-moving car (brake_max) / accelerates backwards (acc_max) once
            # stopped. Within one step the speed never crosses 0 (stop, then change direction).
            u0 = a[..., 0]
            fwd = self.v > 0
            bwd = self.v < 0
            acc = np.where(u0 > 0, u0 * np.where(bwd, cfg.brake_max, cfg.acc_max), u0 * np.where(fwd, cfg.brake_max, cfg.acc_max))
            v = np.clip(self.v + acc * cfg.dt, -cfg.v_reverse, cfg.v_max)
            v = np.where((fwd & (v < 0)) | (bwd & (v > 0)), 0.0, v)
        dk = cfg.kappa_rate * cfg.dt
        kap = self.kappa + np.clip(a[..., 1] * cfg.kappa_max - self.kappa, -dk, dk)
        kg = grip_kappa(cfg, v)  # tyre grip at the new speed (after the steering-rate limit)
        self.kappa = np.where(alive, np.clip(kap, -kg, kg), 0.0)
        self.sig = np.where(alive[..., None], a[..., 2 : 2 + cfg.n_signal], 0.0)
        th = self.th + v * self.kappa * cfg.dt
        self.acc = np.where(alive, (v - self.v) / cfg.dt, 0.0)
        self.x = np.where(alive, self.x + v * np.cos(th) * cfg.dt, self.x)
        self.y = np.where(alive, self.y + v * np.sin(th) * cfg.dt, self.y)
        self.th = np.where(alive, (th + np.pi) % (2 * np.pi) - np.pi, self.th)
        self.v = np.where(alive, v, 0.0)
        self.age += cfg.dt * alive
        self.t += cfg.dt
        self.events = []
        self.done_kind = np.zeros(alive.shape, dtype=np.int8)
        self.done_uid = np.where(alive, self.uid, -1)
        if self.wreck.any():  # wrecks whose time is up disappear (before the collision check)
            self.wreck_left -= self.wreck
            self._remove(self.wreck & (self.wreck_left <= 0))

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
        # collisions between cars (both cars of a pair are marked; vrel = impact speed)
        col, vrel, hit_wreck = self._collisions(ox, oy, alive)
        crash = col & ~off
        for kind, m in (("offroad", off), ("crash", crash)):
            for b, n in zip(*np.nonzero(m)):
                self.events.append((b, kind, self.x[b, n], self.y[b, n]))
        cr += off * (cfg.r_offroad - cfg.r_offroad_speed * self.v) + crash * (cfg.r_crash - cfg.r_crash_speed * vrel)
        self.done_kind[off] = DONE_OFFROAD
        self.done_kind[crash] = DONE_CRASH
        self.stats.offroad += off.sum(1)
        self.stats.crashes += crash.sum(1)
        self.stats.wreck_hits += (crash & hit_wreck).sum(1)
        if wreck_steps(cfg) > 0:  # crashed cars stay as wrecks, off-road cars vanish
            self._remove(off)
            self._make_wreck(crash)
        else:
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
        """Collisions of driving cars with driving cars or wrecks: (col, vrel, hit_wreck) per slot
        (only driving cars are marked; wrecks are not hit again, wreck-wreck pairs are ignored)."""
        N = self.cfg.n_slots
        dcx = self.x[:, :, None] - self.x[:, None, :]
        dcy = self.y[:, :, None] - self.y[:, None, :]
        near = (dcx**2 + dcy**2 < (2 * DISC_OFFSETS[-1] + 2 * DISC_R) ** 2) & alive[:, :, None] & (alive | self.wreck)[:, None, :]
        near &= ~np.eye(N, dtype=bool)[None]
        vrel = np.zeros(alive.shape)
        hit_wreck = np.zeros_like(alive)
        if not near.any():
            return np.zeros_like(alive), vrel, hit_wreck
        b, i, j = np.nonzero(near)
        dx = ox[b, i][:, :, None] - ox[b, j][:, None, :]
        dy = oy[b, i][:, :, None] - oy[b, j][:, None, :]
        hit = ((dx**2 + dy**2) < (2 * DISC_R) ** 2).any((1, 2))
        col = np.zeros_like(alive)
        col[b[hit], i[hit]] = True
        b, i, j = b[hit], i[hit], j[hit]
        vx, vy = self.v * np.cos(self.th), self.v * np.sin(self.th)
        np.maximum.at(vrel, (b, i), np.hypot(vx[b, i] - vx[b, j], vy[b, i] - vy[b, j]))
        np.logical_or.at(hit_wreck, (b, i), self.wreck[b, j])  # (several partners per car)
        return col, vrel, hit_wreck

    def _make_wreck(self, mask):
        """Crashed cars become standing wrecks at their crash position (no control, speed 0,
        lamps off, hazard lights on) for wreck_steps(cfg) steps; they keep their uid."""
        self.alive &= ~mask
        self.wreck |= mask
        self.wreck_left[mask] = wreck_steps(self.cfg)
        self.v[mask] = 0.0
        self.acc[mask] = 0.0
        self.kappa[mask] = 0.0
        self.sig[mask] = 0.0

    def _remove(self, mask):
        self.alive &= ~mask
        self.wreck &= ~mask
        self.wreck_left[mask] = 0
        self.v[mask] = 0.0
        self.acc[mask] = 0.0
        self.sig[mask] = 0.0
        self.age[mask] = 0.0
        self.uid[mask] = -1
        # park dead cars far away so they never interact
        self.x[mask] = 1e4
        self.y[mask] = 1e4

    # ------------------------------------------------------------------ spawning
    def spawn_lats(self):
        """Candidate lateral spawn offsets (right of travel): an even grid over the spawn
        range (whole width, or the right half with spawn_right)."""
        cfg = self.cfg
        lo = SPAWN_RIGHT_MIN if cfg.spawn_right else -(cfg.W - SPAWN_CURB_GAP)
        hi = cfg.W - SPAWN_CURB_GAP
        m = max(int(round((hi - lo) / SPAWN_LAT_STEP)), 1)
        return lo + (hi - lo) * np.arange(m + 1) / m

    def _spawn_clear(self, sx, sy, h, x, y, vx, vy, alive):
        """(b, M) whether candidate spawn points (sx, sy) (M,) with heading h keep the spawn
        clearance (+ closing-speed headway) to every car (x, y, vx, vy, alive: (b, N))."""
        cfg = self.cfg
        dx, dy = x[:, None, :] - sx[:, None], y[:, None, :] - sy[:, None]
        dist = np.hypot(dx, dy)
        dd = np.maximum(dist, 1e-6)
        ch, sh = np.cos(h), np.sin(h)
        # closing speed along the line of sight between the new car and car j
        rvx, rvy = cfg.spawn_speed * ch - vx[:, None, :], cfg.spawn_speed * sh - vy[:, None, :]
        closing = np.maximum((rvx * dx + rvy * dy) / dd, 0.0)
        # outside the new car's lateral corridor only its own speed counts (the other
        # car is treated as standing: it may be queued in the lane the new car joins)
        own = np.maximum(cfg.spawn_speed * (ch * dx + sh * dy) / dd, 0.0)
        lat_off = np.abs(-sh * dx + ch * dy)
        closing = np.where(lat_off < 2 * DISC_R + SPAWN_LAT_MARGIN, closing, own)
        return ((dist > cfg.spawn_clearance + cfg.spawn_headway * closing) | ~alive[:, None, :]).all(-1)

    def _arrivals(self):
        """Queue new arrivals, then release queued cars. Each arrival draws (u, manoeuvre,
        group) when it arrives (CRN: the same draws for every policy). The head of a queue
        enters as soon as some lateral position of its arm is clear (spawn_clearance +
        closing-speed headway to every car) and a slot is free: at its own spot (fraction u
        across the spawn range, spawn_lats grid) if that is clear, else at the floor(u * n_free)-th
        clear position (uniform among the free ones; no side preferred). Several cars of one
        arm may enter in the same step (side by side) while room is left."""
        cfg = self.cfg
        S = self.S
        # draw every random number unconditionally so streams stay aligned (CRN)
        dem = cfg.demand if self._stream_demand is None else self._stream_demand[:, None]
        arrive = self.rng.random((S, 4)) < dem * cfg.dt
        u = self.rng.random((S, 4))
        man = self.rng.choice([RIGHT, STRAIGHT, LEFT], size=(S, 4), p=np.asarray(cfg.turn_probs) / sum(cfg.turn_probs))
        grp = self.rng.choice(len(self.group_probs), size=(S, 4), p=self.group_probs)
        arrive, u, man, grp = arrive[self.scen], u[self.scen], man[self.scen], grp[self.scen]
        if (self.queue + arrive).max() > self._qcap:  # grow the FIFO (keeps order)
            new = np.zeros((self.B, 4, 2 * self._qcap, 3))
            for j in range(self._qcap):
                new[:, :, j] = np.take_along_axis(self._qbuf, ((self._qhead + j) % self._qcap)[..., None, None], 2)[:, :, 0]
            self._qbuf, self._qhead, self._qcap = new, np.zeros_like(self._qhead), 2 * self._qcap
        bi, ki = np.nonzero(arrive)
        slot = (self._qhead[bi, ki] + self.queue[bi, ki]) % self._qcap
        self._qbuf[bi, ki, slot] = np.stack([u[bi, ki], man[bi, ki], grp[bi, ki]], -1)
        self.queue += arrive
        self.stats.arrived += arrive.sum(1)
        # velocities of existing cars (for the closing-speed spawn clearance)
        vx, vy = self.v * np.cos(self.th), self.v * np.sin(self.th)
        spawned = np.zeros(self.alive.shape, dtype=bool)
        occ = self.alive | self.wreck  # wrecks occupy their slot and block like a standing car
        lats = self.spawn_lats()
        back = -cfg.L + SPAWN_BACK
        for k in range(4):
            c, s = rot(k)
            h = np.pi / 2 + k * np.pi / 2  # canonical heading (north) rotated by k*90deg
            hn = np.arctan2(np.sin(h), np.cos(h))
            # candidate spawn points (lat, -L + SPAWN_BACK) rotated by k*90deg
            px, py = c * lats - s * back, s * lats + c * back
            mx, my, half = px.mean(), py.mean(), 0.5 * (lats[-1] - lats[0])
            for _ in range(SPAWN_MAX_PER_STEP):
                b = np.nonzero((self.queue[:, k] > 0) & (~occ).any(1))[0]
                if len(b) == 0:
                    break
                # only cars that can be within reach matter: clearance + headway * the largest
                # possible closing speed around any candidate point (keeps the (b, M, K) block small)
                reach = cfg.spawn_clearance + cfg.spawn_headway * (cfg.spawn_speed + np.hypot(vx[b], vy[b]))
                near = occ[b] & (np.hypot(self.x[b] - mx, self.y[b] - my) < reach + half)
                K = int(near.sum(1).max())
                if K == 0:
                    clear = np.ones((len(b), len(lats)), dtype=bool)
                else:
                    idx = np.argsort(~near, axis=1, kind="stable")[:, :K]
                    g = lambda a: np.take_along_axis(a, idx, 1)  # noqa: E731
                    clear = self._spawn_clear(px, py, h, g(self.x[b]), g(self.y[b]), g(vx[b]), g(vy[b]), g(near))
                cnt = clear.sum(1)
                ok = cnt > 0
                if not ok.any():
                    break
                b, clear, cnt = b[ok], clear[ok], cnt[ok]
                head = self._qhead[b, k]
                qu, qman, qgrp = self._qbuf[b, k, head].T
                # own spot (u across the range) if clear, else the clear spot of the same rank u
                j0 = np.floor(qu * (len(lats) - 1) + 0.5).astype(np.int64)
                pick = np.minimum(np.floor(qu * cnt).astype(np.int64), cnt - 1)
                j = np.where(clear[np.arange(len(b)), j0], j0, np.argmax(np.cumsum(clear, 1) > pick[:, None], 1))
                n = np.argmin(occ[b], 1)
                self.alive[b, n] = True
                occ[b, n] = True
                self.x[b, n], self.y[b, n] = px[j], py[j]
                self.th[b, n] = hn
                self.v[b, n] = cfg.spawn_speed
                vx[b, n], vy[b, n] = cfg.spawn_speed * np.cos(hn), cfg.spawn_speed * np.sin(hn)
                self.kappa[b, n] = 0.0
                self.acc[b, n] = 0.0
                self.arm[b, n] = k
                self.man[b, n] = qman.astype(np.int64)
                self.group[b, n] = qgrp.astype(np.int64)
                self.age[b, n] = 0.0
                self.uid[b, n] = self.next_uid + np.arange(len(b))
                self.next_uid += len(b)
                self.queue[b, k] -= 1
                self._qhead[b, k] = (head + 1) % self._qcap
                self.stats.spawned[b] += 1
                spawned[b, n] = True
        head = np.take_along_axis(self._qbuf, self._qhead[..., None, None], 2)[:, :, 0]
        self.queue_u, self.queue_man, self.queue_grp = head[..., 0], head[..., 1].astype(np.int64), head[..., 2].astype(np.int64)
        if spawned.any():
            self._prev_dist = np.where(spawned, self.dist_to_gate(), self._prev_dist)
