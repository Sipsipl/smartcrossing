"""PyTorch port of :class:`smartcrossing.sim.Sim`: the simulator of the GPU rollouts
(rollout_torch.py, ``ppo --rollout-device cuda``).

Same physics, rewards, removal, arrivals and observations as sim.py, vectorised over
(envs, slots) without data-dependent Python control flow, so it runs on cpu or cuda
with (almost) no host syncs. Constants and SimConfig come from sim.py (nothing is
duplicated). Structure and names mirror sim.py.

Differences to sim.py (by design):
- Everything is computed densely for all N slots (dead slots are masked), instead of
  gathering the occupied ones; the spawn loop always runs its 4 x SPAWN_MAX_PER_STEP
  passes (masked) instead of breaking early, and the spawn-clearance check looks at every
  alive car instead of the pre-filtered "near" ones (a pure optimisation in sim.py).
- Randomness: ``rng="numpy"`` draws every random number with numpy generators in exactly
  the order sim.py does (bit-identical traffic streams, used by the parity test);
  ``rng="torch"`` draws on the device (fast; same distributions, different streams).
- Not ported: ``Stats.travel`` per-vehicle lists (replaced by per-env sums
  ``travel_time`` / ``travel_free`` / exits count) and ``events`` (rendering only).
  The FIFO growth check costs one host sync per step (disabled in CUDA-graph mode, which
  uses a fixed capacity of at least 256 per arm and only flags an overflow in ``fifo_overflow``).
- ``cuda_graph(fn)`` captures observe + policy + step into one CUDA graph: eager torch on the
  GPU is launch-bound (~1000 small kernels per step, ~55 ms on Windows regardless of B).
"""

import math

import numpy as np
import torch

from .sim import (
    ARM_U,
    BRAKE_ACC,
    DISC_OFFSETS,
    DISC_R,
    DONE_CRASH,
    DONE_EXIT,
    DONE_OFFROAD,
    DONE_TIMEOUT,
    DONE_WRONG,
    GOAL_SCALE,
    K_NEIGHBOURS,
    LEFT,
    NEIGHBOUR_RANGE,
    POS_SCALE,
    RAY_ANGLES,
    RAY_RANGE,
    RIGHT,
    SPAWN_LAT_MARGIN,
    SPAWN_MAX_PER_STEP,
    SPEED_TAU,
    STAT_NAMES,
    STRAIGHT,
    SimConfig,
    free_flow_time,
    nb_features,
    obstacle_clear,
    obstacle_pose,
    obstacle_steps,
    spawn_lats,
    spawn_points,
    wreck_steps,
)

GRAPH_FIFO_CAP = 256  # arrival FIFO capacity per arm in CUDA-graph mode (no growth possible)

_COS4 = (1.0, 0.0, -1.0, 0.0)  # rint(cos(k * 90deg)), k = 0..3
_SIN4 = (0.0, 1.0, 0.0, -1.0)


class _Consts:
    """Per (device, dtype) lookup tables."""

    def __init__(self, device, dtype):
        kw = dict(device=device, dtype=dtype)
        self.cos4 = torch.tensor(_COS4, **kw)
        self.sin4 = torch.tensor(_SIN4, **kw)
        self.arm_u = torch.tensor(ARM_U, **kw)
        self.ray_angles = torch.tensor(RAY_ANGLES, **kw)
        self.quad_sx = torch.tensor([1.0, 1.0, -1.0, -1.0], **kw)
        self.quad_sy = torch.tensor([1.0, -1.0, 1.0, -1.0], **kw)
        self.disc_offsets = torch.tensor(DISC_OFFSETS, **kw)


def rot_t(k, C: _Consts):
    """Cos/sin of rotating by k * 90 degrees (k: integer tensor)."""
    k = torch.remainder(k, 4)
    return C.cos4[k], C.sin4[k]


def corner_dist(x, y, W, Rc):
    return torch.hypot(torch.clamp(W + Rc - x.abs(), min=0.0), torch.clamp(W + Rc - y.abs(), min=0.0))


def _disc_entry(px, py, du, dv, R):
    bq = px * du + py * dv
    disc = bq * bq - (px * px + py * py - R * R)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t0, t1 = -bq - sq, -bq + sq
    return torch.where((disc >= 0) & (t1 >= 0), torch.clamp(t0, min=0.0), math.inf)


def ray_edge_dist(x, y, dx, dy, W, Rc, C: _Consts, tmax=RAY_RANGE):
    """Torch version of sim.ray_edge_dist (all inputs broadcastable tensors)."""
    c = W + Rc
    x, y, dx, dy = torch.broadcast_tensors(x, y, dx, dy)
    shp = (4,) + (1,) * x.dim()
    sx, sy = C.quad_sx.view(shp), C.quad_sy.view(shp)
    u, v, du, dv = sx * x, sy * y, sx * dx, sy * dy
    du_ = torch.where(du.abs() < 1e-12, 1e-12, du)
    dv_ = torch.where(dv.abs() < 1e-12, 1e-12, dv)
    iu, iv = 1.0 / du_, 1.0 / dv_
    tWu, tcu, tWv, tcv = (W - u) * iu, (c - u) * iu, (W - v) * iv, (c - v) * iv
    pu, pv = du_ > 0, dv_ > 0
    ninf, inf = -math.inf, math.inf
    lo1 = torch.clamp(torch.maximum(torch.where(pu, tWu, ninf), torch.where(pv, tcv, ninf)), min=0.0)
    hi1 = torch.minimum(torch.where(pu, inf, tWu), torch.where(pv, inf, tcv))
    lo2 = torch.clamp(torch.maximum(torch.where(pu, tcu, ninf), torch.where(pv, tWv, ninf)), min=0.0)
    hi2 = torch.minimum(torch.where(pu, inf, tcu), torch.where(pv, inf, tWv))
    t = torch.minimum(torch.where(lo1 <= hi1, lo1, inf), torch.where(lo2 <= hi2, lo2, inf))
    t = torch.minimum(t, _disc_entry(u - c, v - c, du, dv, Rc))
    return torch.clamp(t.amin(0), max=tmax)


def route_dist(cfg: SimConfig, arm, man, x, y, C: _Consts):
    """Torch version of sim.route_dist."""
    c0 = cfg.W + cfg.Rc
    co, si = rot_t(-arm, C)
    cx = co * x - si * y
    cy = si * x + co * y
    ut = C.arm_u[man]
    in_box = (cx.abs() <= c0) & (cy.abs() <= c0)
    a = torch.argmax(torch.stack([-cy, cx, cy, -cx], -1), -1)
    ua = C.arm_u[a]
    d_target_arm = cfg.L - (cx * ut[..., 0] + cy * ut[..., 1])
    d_box = torch.hypot(cx - c0 * ut[..., 0], cy - c0 * ut[..., 1]) + cfg.L - c0
    d_other = (
        torch.hypot(cx - c0 * ua[..., 0], cy - c0 * ua[..., 1])
        + c0 * torch.hypot(ua[..., 0] - ut[..., 0], ua[..., 1] - ut[..., 1])
        + cfg.L
        - c0
    )
    return torch.where(in_box, d_box, torch.where(a == man, d_target_arm, d_other))


def grip_kappa(cfg: SimConfig, v):
    if cfg.a_lat_max <= 0:
        return torch.full_like(v, cfg.kappa_max)
    return torch.clamp(cfg.a_lat_max / torch.clamp(v * v, min=1e-9), max=cfg.kappa_max)


class TorchStats:
    """Per-environment counters (as sim.Stats, but tensors; travel as per-env sums)."""

    NAMES = STAT_NAMES

    def __init__(self, B, device, dtype):
        for name in self.NAMES:
            setattr(self, name, torch.zeros(B, dtype=torch.int64, device=device))
        self.travel_time = torch.zeros(B, dtype=dtype, device=device)  # sum of exit ages
        self.travel_free = torch.zeros(B, dtype=dtype, device=device)  # sum of their free-flow times


class TorchSim:
    def __init__(self, cfg: SimConfig, n_envs: int, seed: int = 0, n_scenarios: int | None = None,
                 device="cpu", dtype=torch.float64, rng: str = "torch", group_probs=None, fifo_cap: int = 64):
        """rng: "numpy" = sim.py's random streams (bit-identical draws, for parity tests),
        "torch" = on-device generator (fast). group_probs: as sim.Sim. fifo_cap: initial capacity
        of the per-arm arrival FIFO (grows on demand, except in CUDA-graph mode)."""
        assert rng in ("numpy", "torch")
        self.cfg = cfg
        self.B = n_envs
        self.S = n_scenarios or n_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.rng_mode = rng
        self.C = _Consts(self.device, dtype)
        self.scen = torch.arange(self.B, device=self.device) % self.S
        self._scen_np = np.arange(self.B) % self.S
        self.group_probs = np.array([1.0]) if group_probs is None else np.asarray(group_probs, dtype=float)
        self.fifo_cap = fifo_cap
        self.fifo_sync = True  # FIFO growth check (one host sync per step); off in CUDA-graph mode
        self.reset(seed)

    # ------------------------------------------------------------------ helpers
    def _f(self, *shape, fill=0.0):
        return torch.full(shape, fill, dtype=self.dtype, device=self.device)

    def _i(self, *shape, fill=0, dtype=torch.int64):
        return torch.full(shape, fill, dtype=dtype, device=self.device)

    def _t(self, a, dtype=None):
        return torch.as_tensor(np.asarray(a), dtype=dtype or self.dtype, device=self.device)

    def reset(self, seed: int = 0):
        cfg = self.cfg
        B, N = self.B, cfg.n_slots
        if self.rng_mode == "numpy":
            self.rng = np.random.default_rng(seed)
            self.noise_rng = np.random.default_rng([seed, 1])
        else:
            self.gen = torch.Generator(device=self.device)
            self.gen.manual_seed(int(seed))
        self._stream_demand = self.demand = None
        if cfg.demand_range is not None:
            lo, hi = cfg.demand_range
            if self.rng_mode == "numpy":
                self._stream_demand_np = self.rng.uniform(lo, hi, self.S)
                sd = self._t(self._stream_demand_np)
            else:
                sd = lo + (hi - lo) * torch.rand(self.S, generator=self.gen, device=self.device, dtype=self.dtype)
            self._stream_demand = sd
            self.demand = sd[self.scen]
        self.alive = self._i(B, N, dtype=torch.bool)
        self.wreck = self._i(B, N, dtype=torch.bool)  # crashed cars standing as obstacles (sim.py)
        self.wreck_left = self._i(B, N)  # steps until the wreck is removed
        self.obstacle = self._i(B, N, dtype=torch.bool)  # obstacles (cfg.obstacle_*; sim.py)
        self.parked = self._i(B, N, dtype=torch.bool)
        self.obst_env = torch.arange(B, device=self.device) < int(round(cfg.obstacle_envs * B))
        if self.rng_mode == "numpy":
            self.obst_rng = np.random.default_rng([seed, 2])
        self.x, self.y, self.th, self.v = self._f(B, N), self._f(B, N), self._f(B, N), self._f(B, N)
        self.kappa, self.acc = self._f(B, N), self._f(B, N)
        self.sig = self._f(B, N, cfg.n_signal)
        self.arm, self.man = self._i(B, N), self._i(B, N)
        self.age = self._f(B, N)
        self.uid = self._i(B, N, fill=-1)
        self.group = self._i(B, N)
        self.queue = self._i(B, 4)
        self._qcap = self.fifo_cap
        self.fifo_overflow = torch.zeros((), dtype=torch.bool, device=self.device)
        tp = np.asarray(self.cfg.turn_probs, dtype=float)
        self._turn_cdf = self._t(np.cumsum(tp / tp.sum())[:-1])
        gp = self.group_probs
        self._group_cdf = self._t(np.cumsum(gp / gp.sum())[:-1])
        self._qhead = self._i(B, 4)
        self._qbuf = self._f(B, 4, self._qcap, 3)
        self.next_uid = self._i()
        self.t = 0.0
        self.stats = TorchStats(B, self.device, self.dtype)
        self._prev_dist = self._f(B, N)
        self.car_reward = self._f(B, N)
        self.done_kind = self._i(B, N, dtype=torch.int8)
        self.done_uid = self._i(B, N, fill=-1)
        self.queue_cost = self._f(B)
        self._arange_B = torch.arange(B, device=self.device)
        self._arange_N = torch.arange(N, device=self.device)
        self._eye = torch.eye(N, dtype=torch.bool, device=self.device)
        self._lats = self._t(spawn_lats(cfg))
        # per arm: (px, py, h, hn) exactly as sim.py computes them (numpy)
        self._spawn_pts = [(self._t(px), self._t(py), float(h), float(hn)) for px, py, h, hn in spawn_points(cfg)]

    # ------------------------------------------------------------------ geometry
    @property
    def tgt(self):
        return (self.arm + self.man) % 4

    def dist_to_gate(self):
        return route_dist(self.cfg, self.arm, self.man, self.x, self.y, self.C)

    # ------------------------------------------------------------------ observe
    def observe(self) -> torch.Tensor:
        """(B, N, obs_dim) observations; rows of empty slots are zero (computed densely)."""
        cfg, C = self.cfg, self.C
        B, N = self.B, cfg.n_slots
        L, W = cfg.L, cfg.W
        x, y, th, arm, man = self.x, self.y, self.th, self.arm, self.man
        c, s = rot_t(-arm, C)
        cth = th - arm.to(self.dtype) * (math.pi / 2)
        ut = C.arm_u[(arm + man) % 4]
        c0 = W + cfg.Rc
        dx, dy = L * ut[..., 0] - x, L * ut[..., 1] - y
        mx, my = c0 * ut[..., 0] - x, c0 * ut[..., 1] - y
        ct, st = torch.cos(th), torch.sin(th)
        ang = th[..., None] + C.ray_angles
        rays = ray_edge_dist(x[..., None], y[..., None], torch.cos(ang), torch.sin(ang), W, cfg.Rc, C) / RAY_RANGE
        own = torch.cat([
            torch.stack([
                (c * x - s * y) / POS_SCALE,
                (s * x + c * y) / POS_SCALE,
                torch.cos(cth),
                torch.sin(cth),
                self.v / cfg.v_max,
                (man == RIGHT).to(self.dtype),
                (man == STRAIGHT).to(self.dtype),
                (man == LEFT).to(self.dtype),
                (ct * dx + st * dy) / GOAL_SCALE,
                (-st * dx + ct * dy) / GOAL_SCALE,
                (ct * mx + st * my) / GOAL_SCALE,
                (-st * mx + ct * my) / GOAL_SCALE,
            ], -1),
            rays,
            torch.stack([self.kappa / cfg.kappa_max, grip_kappa(cfg, self.v) / cfg.kappa_max], -1),
        ], -1)
        # neighbours: K nearest other cars within range, sorted by distance
        ddx = x[:, None, :] - x[:, :, None]  # (B, N_i, N_j) = x_j - x_i
        ddy = y[:, None, :] - y[:, :, None]
        d = torch.hypot(ddx, ddy)
        valid = (self.alive | self.wreck)[:, None, :] & (d < cfg.neighbour_vis) & ~self._eye
        d = torch.where(valid, d, math.inf)
        K = min(K_NEIGHBOURS, N)
        idx = torch.topk(d, K, dim=-1, largest=False, sorted=True).indices  # (B, N, K)

        def g(a):  # gather neighbour attribute a (B, N) -> (B, N, K)
            return torch.gather(a[:, None, :].expand(B, N, N), 2, idx)

        present = torch.gather(valid, 2, idx)
        nx, ny = torch.gather(ddx, 2, idx), torch.gather(ddy, 2, idx)
        nth = g(th) - th[..., None]
        ct2, st2 = ct[..., None], st[..., None]
        feats = [
            (ct2 * nx + st2 * ny) / NEIGHBOUR_RANGE,
            (-st2 * nx + ct2 * ny) / NEIGHBOUR_RANGE,
            torch.cos(nth),
            torch.sin(nth),
            g(self.v) / cfg.v_max,
            torch.ones_like(nx),
            g(self.kappa) / cfg.kappa_max,
        ]
        if cfg.brake_light:  # brake light + hazard lights
            feats += [(g(self.acc) < BRAKE_ACC).to(self.dtype), g((self.wreck & ~self.parked).to(self.dtype))]
        else:  # legacy interface: acceleration
            feats += [g(self.acc) / cfg.brake_max]
        assert len(feats) == nb_features(cfg)
        f = torch.stack(feats, -1)
        ns = cfg.n_signal
        if ns:
            if cfg.signal_enabled:
                sg = torch.gather(self.sig[:, None, :, :].expand(B, N, N, ns), 2, idx[..., None].expand(B, N, K, ns))
                if cfg.signal_noise > 0:
                    sg = sg + self._signal_noise(K, ns)
            else:
                sg = torch.zeros(B, N, K, ns, dtype=self.dtype, device=self.device)
            f = torch.cat([f, sg], -1)
        f = f * present[..., None]
        obs = torch.cat([own, f.reshape(B, N, -1)], -1)
        return torch.where(self.alive[..., None], obs, 0.0)

    def _signal_noise(self, K, ns):
        B, N = self.alive.shape
        if self.rng_mode == "numpy":  # same draws as sim.py: (M, K, ns) for the alive slots in row-major order
            m = int(self.alive.sum())
            out = self._f(B, N, K, ns)
            out[self.alive] = self._t(self.noise_rng.normal(0, self.cfg.signal_noise, (m, K, ns)))
            return out
        return self.cfg.signal_noise * torch.randn(B, N, K, ns, generator=self.gen, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------ step
    def step(self, action: torch.Tensor) -> torch.Tensor:
        """As sim.Sim.step. action: (B, N, act_dim) tensor in [-1, 1]; returns the team reward (B,)."""
        cfg = self.cfg
        a = torch.clamp(action.to(dtype=self.dtype, device=self.device), -1.0, 1.0)
        alive = self.alive
        if cfg.speed_control:
            v_t = (a[..., 0] + 1.0) * 0.5 * cfg.v_max
            acc = torch.clamp((v_t - self.v) / SPEED_TAU, -cfg.brake_max, cfg.acc_max)
            v = torch.clamp(self.v + acc * cfg.dt, 0.0, cfg.v_max)
        else:
            u0 = a[..., 0]
            fwd = self.v > 0
            bwd = self.v < 0
            acc = torch.where(u0 > 0, u0 * torch.where(bwd, cfg.brake_max, cfg.acc_max),
                              u0 * torch.where(fwd, cfg.brake_max, cfg.acc_max))
            v = torch.clamp(self.v + acc * cfg.dt, -cfg.v_reverse, cfg.v_max)
            v = torch.where((fwd & (v < 0)) | (bwd & (v > 0)), 0.0, v)
        dk = cfg.kappa_rate * cfg.dt
        kap = self.kappa + torch.clamp(a[..., 1] * cfg.kappa_max - self.kappa, -dk, dk)
        kg = grip_kappa(cfg, v)
        self.kappa = torch.where(alive, torch.minimum(torch.maximum(kap, -kg), kg), 0.0)
        self.sig = torch.where(alive[..., None], a[..., 2 : 2 + cfg.n_signal], 0.0)
        th = self.th + v * self.kappa * cfg.dt
        self.acc = torch.where(alive, (v - self.v) / cfg.dt, 0.0)
        self.x = torch.where(alive, self.x + v * torch.cos(th) * cfg.dt, self.x)
        self.y = torch.where(alive, self.y + v * torch.sin(th) * cfg.dt, self.y)
        self.th = torch.where(alive, torch.remainder(th + math.pi, 2 * math.pi) - math.pi, self.th)
        self.v = torch.where(alive, v, 0.0)
        fa = alive.to(self.dtype)
        self.age = self.age + cfg.dt * fa
        self.t += cfg.dt
        done_kind = torch.zeros(alive.shape, dtype=torch.int8, device=self.device)
        self.done_uid = torch.where(alive, self.uid, -1)
        ws = wreck_steps(cfg)
        if ws > 0 or cfg.obstacle_envs > 0:  # wrecks whose time is up disappear (before the collision check)
            self.wreck_left = self.wreck_left - self.wreck.to(torch.int64)
            self._remove(self.wreck & (self.wreck_left <= 0))

        cr = (-cfg.r_time * cfg.dt) * fa
        dist = self.dist_to_gate()
        cr = cr + torch.where(alive, cfg.r_progress * (self._prev_dist - dist), 0.0)
        self._prev_dist = dist

        # exits: beyond L in some arm
        out = alive & ((self.x.abs() > cfg.L) | (self.y.abs() > cfg.L))
        exit_arm = torch.argmax(torch.stack([-self.y, self.x, self.y, -self.x], -1), -1)
        ok = out & (exit_arm == self.tgt)
        wrong = out & ~ok
        cr = cr + cfg.r_exit * ok.to(self.dtype) + cfg.r_wrong_exit * wrong.to(self.dtype)
        done_kind = torch.where(ok, DONE_EXIT, done_kind).to(torch.int8)
        done_kind = torch.where(wrong, DONE_WRONG, done_kind).to(torch.int8)
        self.stats.exits += ok.sum(1)
        self.stats.wrong_exits += wrong.sum(1)
        self.stats.travel_time += torch.where(ok, self.age, 0.0).sum(1)
        self.stats.travel_free += torch.where(ok, self._free_flow(), 0.0).sum(1)
        self._remove(out)
        alive = self.alive

        # off-road: any disc overlapping the complement of the road
        ox, oy = self._discs()
        off = (corner_dist(ox, oy, cfg.W, cfg.Rc) < cfg.Rc + DISC_R).any(-1) & alive
        col, vrel, hit_wreck, hit_obst = self._collisions(ox, oy, alive)
        crash = col & ~off
        if cfg.crash_cost_speed == "own":  # sim.py: own speed at impact
            vrel = self.v.abs()
        else:
            assert cfg.crash_cost_speed == "relative", cfg.crash_cost_speed
        cr = cr + off.to(self.dtype) * (cfg.r_offroad - cfg.r_offroad_speed * self.v) + crash.to(self.dtype) * (cfg.r_crash - cfg.r_crash_speed * vrel)
        done_kind = torch.where(off, DONE_OFFROAD, done_kind).to(torch.int8)
        done_kind = torch.where(crash, DONE_CRASH, done_kind).to(torch.int8)
        self.stats.offroad += off.sum(1)
        self.stats.crashes += crash.sum(1)
        self.stats.wreck_hits += (crash & hit_wreck).sum(1)
        self.stats.obstacle_hits += (crash & hit_obst).sum(1)
        if ws > 0:  # crashed cars stay as wrecks, off-road cars vanish
            self._remove(off)
            self._make_wreck(crash, ws)
        else:
            self._remove(off | crash)

        timeout = self.alive & (self.age > cfg.max_age)
        cr = cr + cfg.r_timeout * timeout.to(self.dtype)
        done_kind = torch.where(timeout, DONE_TIMEOUT, done_kind).to(torch.int8)
        self.stats.timeouts += timeout.sum(1)
        self._remove(timeout)
        self.done_kind = done_kind

        if cfg.obstacle_envs > 0:
            self._obstacles()
        self._arrivals()
        self.car_reward = cr
        self.queue_cost = cfg.r_queue * cfg.dt * self.queue.sum(1).to(self.dtype)
        return cr.sum(1) - self.queue_cost

    def _free_flow(self):
        # free_flow_time is cheap and only depends on the manoeuvre: tabulate it per manoeuvre
        if not hasattr(self, "_ff_tab"):
            self._ff_tab = self._t(free_flow_time(self.cfg, np.arange(4)))
        return self._ff_tab[self.man]

    def _discs(self):
        ct, st = torch.cos(self.th)[..., None], torch.sin(self.th)[..., None]
        off = self.C.disc_offsets
        return self.x[..., None] + ct * off, self.y[..., None] + st * off

    def _collisions(self, ox, oy, alive):
        """Dense (B, N, N) pair check; same result as sim.Sim._collisions."""
        dcx = self.x[:, :, None] - self.x[:, None, :]
        dcy = self.y[:, :, None] - self.y[:, None, :]
        near = (dcx**2 + dcy**2 < (2 * DISC_OFFSETS[-1] + 2 * DISC_R) ** 2) & alive[:, :, None] & (alive | self.wreck)[:, None, :]
        near = near & ~self._eye
        r2 = (2 * DISC_R) ** 2
        hit = torch.zeros_like(near)
        nd = len(DISC_OFFSETS)
        for p in range(nd):
            for q in range(nd):
                dx = ox[:, :, None, p] - ox[:, None, :, q]
                dy = oy[:, :, None, p] - oy[:, None, :, q]
                hit |= dx**2 + dy**2 < r2
        hit &= near
        col = hit.any(-1)
        vx, vy = self.v * torch.cos(self.th), self.v * torch.sin(self.th)
        dv = torch.hypot(vx[:, :, None] - vx[:, None, :], vy[:, :, None] - vy[:, None, :])
        vrel = torch.where(hit, dv, 0.0).amax(-1)
        hit_wreck = (hit & (self.wreck & ~self.obstacle)[:, None, :]).any(-1)
        hit_obst = (hit & self.obstacle[:, None, :]).any(-1)
        return col, vrel, hit_wreck, hit_obst

    def _make_wreck(self, mask, ws):
        """As sim.Sim._make_wreck."""
        self.alive = self.alive & ~mask
        self.wreck = self.wreck | mask
        self.wreck_left = torch.where(mask, ws, self.wreck_left)
        self.v = torch.where(mask, 0.0, self.v)
        self.acc = torch.where(mask, 0.0, self.acc)
        self.kappa = torch.where(mask, 0.0, self.kappa)
        self.sig = torch.where(mask[..., None], 0.0, self.sig)

    def _remove(self, mask):
        self.alive = self.alive & ~mask
        self.wreck = self.wreck & ~mask
        self.wreck_left = torch.where(mask, 0, self.wreck_left)
        self.obstacle = self.obstacle & ~mask
        self.parked = self.parked & ~mask
        self.v = torch.where(mask, 0.0, self.v)
        self.acc = torch.where(mask, 0.0, self.acc)
        self.sig = torch.where(mask[..., None], 0.0, self.sig)
        self.age = torch.where(mask, 0.0, self.age)
        self.uid = torch.where(mask, -1, self.uid)
        self.x = torch.where(mask, 1e4, self.x)
        self.y = torch.where(mask, 1e4, self.y)

    # ------------------------------------------------------------------ obstacles
    def _obstacles(self):
        """As sim.Sim._obstacles (masked, no branches)."""
        cfg = self.cfg
        if self.rng_mode == "numpy":
            u = self._t(self.obst_rng.random((self.B, 7)))
        else:
            u = torch.rand(self.B, 7, generator=self.gen, device=self.device, dtype=self.dtype)
        occ = self.alive | self.wreck
        x, y, th = obstacle_pose(cfg, u)
        vx, vy = self.v * torch.cos(self.th), self.v * torch.sin(self.th)
        clear = (obstacle_clear(cfg, x[:, None] - self.x, y[:, None] - self.y, vx, vy) | ~occ).all(1)
        ok = (self.obst_env & (u[:, 0] < cfg.obstacle_rate * cfg.dt) & (self.obstacle.sum(1) < cfg.obstacle_max)
              & (~occ).any(1) & clear)
        n = torch.argmax((~occ).to(torch.int8), 1)  # first free slot
        sel = ok[:, None] & (self._arange_N[None, :] == n[:, None])
        self.wreck = self.wreck | sel
        self.obstacle = self.obstacle | sel
        self.parked = torch.where(sel, (u[:, 1] < cfg.obstacle_parked)[:, None], self.parked)
        self.wreck_left = torch.where(sel, obstacle_steps(cfg), self.wreck_left)
        self.x = torch.where(sel, x[:, None], self.x)
        self.y = torch.where(sel, y[:, None], self.y)
        self.th = torch.where(sel, th[:, None], self.th)
        for name in ("v", "acc", "kappa", "age"):
            setattr(self, name, torch.where(sel, 0.0, getattr(self, name)))
        self.sig = torch.where(sel[..., None], 0.0, self.sig)
        self.arm = torch.where(sel, 0, self.arm)
        self.man = torch.where(sel, 0, self.man)
        self.uid = torch.where(sel, -1, self.uid)
        self.group = torch.where(sel, 0, self.group)
        self.stats.obstacles += ok

    # ------------------------------------------------------------------ spawning
    def _spawn_clear(self, sx, sy, h, x, y, vx, vy, alive):
        """(B, M) whether candidate spawn points (sx, sy) (M,) keep the clearance to every alive car."""
        cfg = self.cfg
        dx, dy = x[:, None, :] - sx[:, None], y[:, None, :] - sy[:, None]
        dist = torch.hypot(dx, dy)
        dd = torch.clamp(dist, min=1e-6)
        ch, sh = float(np.cos(h)), float(np.sin(h))  # numpy's values (bit-identical to sim.py)
        rvx, rvy = cfg.spawn_speed * ch - vx[:, None, :], cfg.spawn_speed * sh - vy[:, None, :]
        closing = torch.clamp((rvx * dx + rvy * dy) / dd, min=0.0)
        own = torch.clamp(cfg.spawn_speed * (ch * dx + sh * dy) / dd, min=0.0)
        lat_off = (-sh * dx + ch * dy).abs()
        closing = torch.where(lat_off < 2 * DISC_R + SPAWN_LAT_MARGIN, closing, own)
        return ((dist > cfg.spawn_clearance + cfg.spawn_headway * closing) | ~alive[:, None, :]).all(-1)

    def _draws(self):
        """(B, 4) arrive (bool), u, man, grp: one draw per stream and arm, indexed by scenario."""
        cfg, S = self.cfg, self.S
        if self.rng_mode == "numpy":  # exactly sim.Sim._arrivals' calls
            dem = cfg.demand if self._stream_demand is None else self._stream_demand_np[:, None]
            arrive = self.rng.random((S, 4)) < dem * cfg.dt
            u = self.rng.random((S, 4))
            man = self.rng.choice([RIGHT, STRAIGHT, LEFT], size=(S, 4), p=np.asarray(cfg.turn_probs) / sum(cfg.turn_probs))
            grp = self.rng.choice(len(self.group_probs), size=(S, 4), p=self.group_probs)
            sc = self._scen_np
            return (self._t(arrive[sc], torch.bool), self._t(u[sc]), self._t(man[sc], torch.int64),
                    self._t(grp[sc], torch.int64))
        kw = dict(generator=self.gen, device=self.device)
        dem = cfg.demand if self._stream_demand is None else self._stream_demand[:, None]
        arrive = torch.rand(S, 4, dtype=self.dtype, **kw) < dem * cfg.dt
        u = torch.rand(S, 4, dtype=self.dtype, **kw)
        # categorical draws by inverse cdf (sync-free, CUDA-graph safe)
        man = RIGHT + (torch.rand(S, 4, 1, dtype=self.dtype, **kw) > self._turn_cdf).sum(-1)
        if len(self.group_probs) > 1:
            grp = (torch.rand(S, 4, 1, dtype=self.dtype, **kw) > self._group_cdf).sum(-1)
        else:
            grp = torch.zeros(S, 4, dtype=torch.int64, device=self.device)
        sc = self.scen
        return arrive[sc], u[sc], man[sc], grp[sc]

    def _arrivals(self):
        """As sim.Sim._arrivals, without data-dependent branches (masked passes)."""
        cfg = self.cfg
        B, N = self.B, cfg.n_slots
        arrive, u, man, grp = self._draws()
        need = (self.queue + arrive).max()
        if not self.fifo_sync:  # CUDA-graph mode: fixed capacity, overflow only flagged (check fifo_overflow)
            self.fifo_overflow |= need > self._qcap
        elif int(need) > self._qcap:  # grow the FIFO (keeps order); one host sync
            self._widen_fifo(2 * self._qcap)
        slot = ((self._qhead + self.queue) % self._qcap)[..., None, None].expand(B, 4, 1, 3)
        vals = torch.stack([u, man.to(self.dtype), grp.to(self.dtype)], -1)[:, :, None, :]
        cur = torch.gather(self._qbuf, 2, slot)
        self._qbuf.scatter_(2, slot, torch.where(arrive[..., None, None], vals, cur))
        self.queue = self.queue + arrive
        self.stats.arrived += arrive.sum(1)
        vx, vy = self.v * torch.cos(self.th), self.v * torch.sin(self.th)
        spawned = torch.zeros_like(self.alive)
        occ = self.alive | self.wreck  # wrecks occupy their slot and block like a standing car
        M = len(self._lats)
        arB, arN = self._arange_B, self._arange_N
        for k in range(4):
            px, py, h, hn = self._spawn_pts[k]
            chn, shn = float(np.cos(hn)), float(np.sin(hn))
            # cars do not move during the arrivals: check the clearance to all cars once per arm,
            # then only against the car spawned in the previous pass (same result as rechecking all)
            clear = self._spawn_clear(px, py, h, self.x, self.y, vx, vy, occ)
            for p in range(SPAWN_MAX_PER_STEP):
                cand = (self.queue[:, k] > 0) & (~occ).any(1)
                cnt = clear.sum(1)
                ok = cand & (cnt > 0)
                head = self._qhead[:, k]
                q = self._qbuf[arB, k, head]  # (B, 3)
                qu, qman, qgrp = q[:, 0], q[:, 1].to(torch.int64), q[:, 2].to(torch.int64)
                j0 = torch.floor(qu * (M - 1) + 0.5).to(torch.int64)
                pick = torch.minimum(torch.floor(qu * cnt).to(torch.int64), cnt - 1)
                jalt = torch.argmax((torch.cumsum(clear, 1) > pick[:, None]).to(torch.int8), 1)
                j = torch.where(clear[arB, j0], j0, jalt)
                n = torch.argmax((~occ).to(torch.int8), 1)  # first free slot
                sel = ok[:, None] & (arN[None, :] == n[:, None])
                self.alive = self.alive | sel
                occ = occ | sel
                self.x = torch.where(sel, px[j][:, None], self.x)
                self.y = torch.where(sel, py[j][:, None], self.y)
                self.th = torch.where(sel, hn, self.th)
                self.v = torch.where(sel, cfg.spawn_speed, self.v)
                vx = torch.where(sel, cfg.spawn_speed * chn, vx)
                vy = torch.where(sel, cfg.spawn_speed * shn, vy)
                self.kappa = torch.where(sel, 0.0, self.kappa)
                self.acc = torch.where(sel, 0.0, self.acc)
                self.arm = torch.where(sel, k, self.arm)
                self.man = torch.where(sel, qman[:, None], self.man)
                self.group = torch.where(sel, qgrp[:, None], self.group)
                self.age = torch.where(sel, 0.0, self.age)
                rank = torch.cumsum(ok, 0) - 1
                self.uid = torch.where(sel, (self.next_uid + rank)[:, None], self.uid)
                self.next_uid = self.next_uid + ok.sum()
                self.queue[:, k] -= ok.to(torch.int64)
                self._qhead[:, k] = torch.where(ok, (head + 1) % self._qcap, head)
                self.stats.spawned += ok
                spawned |= sel
                if p == SPAWN_MAX_PER_STEP - 1:
                    break
                sxn, syn = px[j][:, None], py[j][:, None]
                svx = torch.full_like(sxn, cfg.spawn_speed * chn)
                svy = torch.full_like(syn, cfg.spawn_speed * shn)
                clear = clear & self._spawn_clear(px, py, h, sxn, syn, svx, svy, ok[:, None])
        self._prev_dist = torch.where(spawned, self.dist_to_gate(), self._prev_dist)

    def _widen_fifo(self, cap):
        """Grow the arrival FIFO to capacity cap (keeps the queue order; the head moves to 0)."""
        if self._qcap >= cap:
            return
        B = self.B
        j = (self._qhead[..., None] + torch.arange(self._qcap, device=self.device)) % self._qcap
        old = torch.gather(self._qbuf, 2, j[..., None].expand(B, 4, self._qcap, 3))
        self._qbuf = torch.cat([old, torch.zeros(B, 4, cap - self._qcap, 3, dtype=self.dtype, device=self.device)], 2)
        self._qhead = torch.zeros_like(self._qhead)
        self._qcap = cap

    # ------------------------------------------------------------------ CUDA graphs
    def reset_inplace(self, seed: int, cfg: SimConfig | None = None):
        """New episode without reallocating: the state of a fresh ``TorchSim(cfg, seed=seed)`` (rng
        "torch"), incl. its generator state, is copied into the existing tensors (and generator),
        so a captured CUDA graph stays valid and the episode is reproducible from the seed (same
        trajectory as a fresh sim). cfg may only differ in ``demand_range`` (must stay set / unset
        as before)."""
        assert self.rng_mode == "torch"
        cfg = cfg or self.cfg
        assert (cfg.demand_range is None) == (self.cfg.demand_range is None)
        fresh = TorchSim(cfg, self.B, seed, n_scenarios=self.S, device=self.device, dtype=self.dtype, rng="torch",
                         group_probs=self.group_probs, fifo_cap=self._qcap)
        for o, f in ((self, fresh), (self.stats, fresh.stats)):
            fv = vars(f)
            for n, v in vars(o).items():
                if isinstance(v, torch.Tensor) and n in fv and fv[n].shape == v.shape:
                    v.copy_(fv[n])
        self.gen.set_state(fresh.gen.get_state())
        self.cfg = cfg
        self.t = 0.0

    def _run_static(self, fn):
        """Run fn() and copy every reassigned tensor attribute back into its old storage, so the
        state lives at fixed addresses (needed for CUDA-graph replay)."""
        objs = [self, self.stats]
        before = [(o, n, v) for o in objs for n, v in vars(o).items() if isinstance(v, torch.Tensor)]
        out = fn()
        for o, n, v in before:
            cur = getattr(o, n)
            if cur is not v:
                v.copy_(cur)
                setattr(o, n, v)
        return out

    def cuda_graph(self, fn, generators=(), warmup=3):
        """Capture fn() (observe / policy / step on this sim; all tensors it uses must be
        long-lived) into a CUDA graph and return replay(). Requires rng="torch"; the FIFO gets a
        fixed capacity (at least GRAPH_FIFO_CAP; fifo_overflow flags an overflow)."""
        assert self.device.type == "cuda" and self.rng_mode == "torch"
        if self.fifo_sync:
            self.fifo_sync = False
            self.fifo_cap = max(self.fifo_cap, GRAPH_FIFO_CAP)  # also after reset()
            self._widen_fifo(self.fifo_cap)  # once, before the addresses are fixed
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(warmup):
                self._run_static(fn)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        for gen in (self.gen, *generators):
            g.register_generator_state(gen)
        t0 = self.t
        with torch.cuda.graph(g):
            self._run_static(fn)
        dt_t = self.t - t0  # capture only records the kernels: the state did not advance
        self.t = t0

        def replay():
            g.replay()
            self.t += dt_t

        return replay
