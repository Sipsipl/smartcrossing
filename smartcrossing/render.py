"""Headless rendering with Pillow: single frames, contact sheets and GIFs."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sim import BRAKE_ACC, CAR_LENGTH, CAR_WIDTH, LEFT, RIGHT, STRAIGHT, Sim, hazard_on

MAN_COLOURS = {RIGHT: (66, 135, 245), STRAIGHT: (240, 240, 240), LEFT: (245, 166, 35)}
GROUP_OUTLINE = [(20, 20, 20), (200, 30, 200), (30, 200, 120), (230, 60, 60)]
OBSTACLE_BODY = (150, 150, 156)  # obstacles (cfg.obstacle_*): no route, drawn grey
BG = (58, 110, 62)
ROAD = (70, 70, 74)


def _signal_colour(sig):
    """Map a signal vector in [-1, 1]^k to a colour: one lamp -> dark (-1) ... bright violet-white
    (+1) (as the website); two lamps (old checkpoints) -> red / blue mix."""
    if len(sig) == 1:
        t = (float(np.clip(sig[0], -1, 1)) + 1) / 2
        return (int(40 + 185 * t), int(40 + 130 * t), int(50 + 205 * t))
    s = np.zeros(2)
    s[: min(2, len(sig))] = sig[:2]
    r = int(127 + 127 * s[0])
    bl = int(127 + 127 * s[1]) if len(sig) > 1 else 127
    return (r, 40, bl)


def default_view(cfg) -> float:
    """Half-width of the default view: the box plus ~30 m of each arm."""
    return cfg.W + cfg.Rc + 30.0


def _legend(d, size, cfg, font):
    """Small legend in the bottom-left corner: body colour = manoeuvre, roof lamp = signal."""
    x0, y0 = 6, size - (46 if cfg.n_signal > 0 else 34)
    d.rectangle([x0 - 3, y0 - 3, x0 + 118, size - 4], fill=(30, 30, 34))
    for i, (m, name) in enumerate(((RIGHT, "right"), (STRAIGHT, "straight"), (LEFT, "left"))):
        xx = x0 + i * 40
        d.rectangle([xx, y0 + 2, xx + 9, y0 + 8], fill=MAN_COLOURS[m])
        d.text((xx + 12, y0), name[:5] if name != "straight" else "str", fill=(230, 230, 230), font=font)
    d.text((x0, y0 + 13), "colour = manoeuvre", fill=(200, 200, 200), font=font)
    if cfg.n_signal > 0:
        lo, hi = (np.array([1.0, -1.0]), np.array([-1.0, 1.0])) if cfg.n_signal > 1 else (np.array([-1.0]), np.array([1.0]))
        d.ellipse([x0, y0 + 27, x0 + 7, y0 + 34], fill=_signal_colour(lo))
        d.ellipse([x0 + 9, y0 + 27, x0 + 16, y0 + 34], fill=_signal_colour(hi))
        d.text((x0 + 20, y0 + 25), "roof lamp = signal", fill=(200, 200, 200), font=font)


def draw_frame(sim: Sim, b: int = 0, size: int = 512, view: float | None = None, label: str = "", trails=None,
               legend: bool = True) -> Image.Image:
    cfg = sim.cfg
    view = default_view(cfg) if view is None else view
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    sc = size / (2 * view)

    def P(x, y):
        return (size / 2 + x * sc, size / 2 - y * sc)

    W, Rc = cfg.W, cfg.Rc
    for sx in (-1, 1):
        for sy in (-1, 1):
            d.rectangle([P(min(sx * W, sx * (W + Rc)), max(sy * W, sy * (W + Rc))),
                         P(max(sx * W, sx * (W + Rc)), min(sy * W, sy * (W + Rc)))], fill=ROAD)
            cx, cy = P(sx * (W + Rc), sy * (W + Rc))
            d.ellipse([cx - Rc * sc, cy - Rc * sc, cx + Rc * sc, cy + Rc * sc], fill=BG)
    d.rectangle([P(-view, W), P(view, -W)], fill=ROAD)
    d.rectangle([P(-W, view), P(W, -view)], fill=ROAD)
    if trails is not None:
        for pts in trails:
            if len(pts) > 1:
                d.line([P(*p) for p in pts], fill=(120, 120, 128), width=1)
    blink = hazard_on(sim.t)
    for n in np.nonzero(sim.alive[b] | sim.wreck[b])[0]:
        x, y, th = sim.x[b, n], sim.y[b, n], sim.th[b, n]
        c, s = np.cos(th), np.sin(th)
        hl, hw = CAR_LENGTH / 2, CAR_WIDTH / 2
        corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
        poly = [P(x + c * a - s * bb, y + s * a + c * bb) for a, bb in corners]
        wreck = bool(sim.wreck[b, n])
        parked = bool(sim.parked[b, n])  # an obstacle drawn as a parked car: undimmed body, hazard lights too
        body = OBSTACLE_BODY if sim.obstacle[b, n] else MAN_COLOURS[int(sim.man[b, n])]
        if wreck and not parked:  # a wreck: dimmed body
            body = tuple(int(0.45 * v + 30) for v in body)
        d.polygon(poly, fill=body, outline=GROUP_OUTLINE[int(sim.group[b, n]) % 4])
        r = max(1.5, 0.35 * sc)
        if parked:  # nose marker (heading of the parked car)
            nx, ny = P(x + c * 2.0, y + s * 2.0)
            rr = max(1.5, 0.3 * sc)
            d.ellipse([nx - rr, ny - rr, nx + rr, ny + rr], fill=(0, 0, 0))
        if wreck:  # every non-participant: blinking hazard lights at the four corners
            if blink:
                for a, bb in corners:
                    hx, hy = P(x + c * a * 0.9 - s * bb * 0.9, y + s * a * 0.9 + c * bb * 0.9)
                    d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(255, 170, 0))
            continue
        if sim.acc[b, n] < BRAKE_ACC:  # brake lights
            for bb in (-0.6, 0.6):
                hx, hy = P(x - c * 2.1 - s * bb, y - s * 2.1 + c * bb)
                d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=(255, 30, 30))
        # signal lamp on the roof
        if cfg.n_signal > 0:
            lx, ly = P(x - c * 0.6, y - s * 0.6)
            r = 0.7 * sc
            d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=_signal_colour(sim.sig[b, n]))
        # nose marker
        nx, ny = P(x + c * 2.0, y + s * 2.0)
        r = max(1.5, 0.3 * sc)
        d.ellipse([nx - r, ny - r, nx + r, ny + r], fill=(0, 0, 0))
    for eb, kind, ex, ey in sim.events:
        if eb == b:
            cx, cy = P(ex, ey)
            r = 6
            col = (255, 40, 40) if kind == "crash" else (255, 220, 0)
            d.line([cx - r, cy - r, cx + r, cy + r], fill=col, width=3)
            d.line([cx - r, cy + r, cx + r, cy - r], fill=col, width=3)
    st = sim.stats
    font = ImageFont.load_default()
    txt = (f"{label} t={sim.t:5.1f}s  exits={st.exits[b]} crash={st.crashes[b]} off={st.offroad[b]} "
           f"wrecks={(sim.wreck[b] & ~sim.obstacle[b]).sum()} q={sim.queue[b].sum()}"
           + (f" obst={sim.obstacle[b].sum()}" if sim.obstacle[b].any() or st.obstacles[b] else ""))
    d.rectangle([0, 0, size, 18], fill=(30, 30, 34))
    d.text((6, 4), txt, fill=(255, 255, 255), font=font)
    if legend:
        _legend(d, size, cfg, font)
    return img


def contact_sheet(frames, cols=3, gap=4) -> Image.Image:
    w, h = frames[0].size
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % cols) * (w + gap), (i // cols) * (h + gap)))
    return sheet


def save_gif(frames, path, fps=10):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


class SpaceTime:
    """Space-time diagnostic for one env: per entry arm, every car's distance to
    the centre over time: -r while approaching, r - 2*r_min after the closest
    point (continuous; = signed distance for straight cars). Flat segments are
    stops, slopes are speeds; colour = manoeuvre."""

    def __init__(self, sim: Sim, b: int = 0):
        self.sim, self.b = sim, b
        self.tracks = {}  # uid -> (arm, man, [(t, d)])
        self._min = {}

    def record(self):
        sim, b = self.sim, self.b
        for n in np.nonzero(sim.alive[b])[0]:
            uid = int(sim.uid[b, n])
            r = float(np.hypot(sim.x[b, n], sim.y[b, n]))
            m = min(self._min.get(uid, np.inf), r)
            self._min[uid] = m
            dist = r - 2 * m if r > m + 0.5 else -r  # continuous through the closest point
            tr = self.tracks.setdefault(uid, (int(sim.arm[b, n]), int(sim.man[b, n]), []))
            tr[2].append((sim.t, dist))

    def image(self, w: int = 360, h: int = 240) -> Image.Image:
        cfg = self.sim.cfg
        font = ImageFont.load_default()
        img = Image.new("RGB", (2 * w, 2 * h), (24, 24, 28))
        d = ImageDraw.Draw(img)
        t1 = max(self.sim.t, 1e-6)
        names = ["south", "east", "north", "west"]
        box = cfg.W + cfg.Rc
        for k in range(4):
            ox, oy = (k % 2) * w, (k // 2) * h
            pad = 22

            def P(t, dd):
                return (ox + pad + (t / t1) * (w - pad - 6), oy + h / 2 - dd / cfg.L * (h / 2 - 14))

            d.rectangle([ox + pad, oy + 14, ox + w - 6, oy + h - 14], fill=(36, 36, 40))
            for yy in (-box, box):
                d.line([P(0, yy), P(t1, yy)], fill=(90, 90, 96))
            d.line([P(0, 0), P(t1, 0)], fill=(140, 140, 150))
            d.text((ox + pad + 2, oy + 1), f"from {names[k]}  (d to centre vs t, 0..{t1:.0f}s)", fill=(220, 220, 220), font=font)
            d.text((ox + 2, oy + 14), f"+{cfg.L:.0f}", fill=(160, 160, 160), font=font)
            d.text((ox + 2, oy + h - 26), f"-{cfg.L:.0f}", fill=(160, 160, 160), font=font)
            for arm, man, pts in self.tracks.values():
                if arm != k or len(pts) < 2:
                    continue
                d.line([P(t, dd) for t, dd in pts], fill=MAN_COLOURS[man], width=1)
        return img
