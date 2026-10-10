"""Headless rendering with Pillow: single frames, contact sheets and GIFs."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sim import LEFT, RIGHT, STRAIGHT, Sim

MAN_COLOURS = {RIGHT: (66, 135, 245), STRAIGHT: (240, 240, 240), LEFT: (245, 166, 35)}
GROUP_OUTLINE = [(20, 20, 20), (200, 30, 200), (30, 200, 120), (230, 60, 60)]
BG = (58, 110, 62)
ROAD = (70, 70, 74)


def _signal_colour(sig):
    """Map a signal vector in [-1, 1]^k to a colour (first two dims -> red/blue)."""
    s = np.zeros(2)
    s[: min(2, len(sig))] = sig[:2]
    r = int(127 + 127 * s[0])
    bl = int(127 + 127 * s[1]) if len(sig) > 1 else 127
    return (r, 40, bl)


def draw_frame(sim: Sim, b: int = 0, size: int = 512, view: float = 52.0, label: str = "", trails=None) -> Image.Image:
    cfg = sim.cfg
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
    for n in np.nonzero(sim.alive[b])[0]:
        x, y, th = sim.x[b, n], sim.y[b, n], sim.th[b, n]
        c, s = np.cos(th), np.sin(th)
        hl, hw = 2.3, 0.9
        corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
        poly = [P(x + c * a - s * bb, y + s * a + c * bb) for a, bb in corners]
        d.polygon(poly, fill=MAN_COLOURS[int(sim.man[b, n])], outline=GROUP_OUTLINE[int(sim.group[b, n]) % 4])
        # signal lamp on the roof
        if cfg.n_signal > 0:
            lx, ly = P(x - c * 0.6, y - s * 0.6)
            r = 0.7 * sc
            d.ellipse([lx - r, ly - r, lx + r, ly + r], fill=_signal_colour(sim.sig[b, n]))
        # nose marker
        nx, ny = P(x + c * 2.0, y + s * 2.0)
        d.ellipse([nx - 1.5, ny - 1.5, nx + 1.5, ny + 1.5], fill=(0, 0, 0))
    for eb, kind, ex, ey in sim.events:
        if eb == b:
            cx, cy = P(ex, ey)
            r = 6
            col = (255, 40, 40) if kind == "crash" else (255, 220, 0)
            d.line([cx - r, cy - r, cx + r, cy + r], fill=col, width=3)
            d.line([cx - r, cy + r, cx + r, cy - r], fill=col, width=3)
    st = sim.stats
    txt = f"{label} t={sim.t:5.1f}s  exits={st.exits[b]} crash={st.crashes[b]} off={st.offroad[b]} q={sim.queue[b].sum()}"
    d.text((6, 6), txt, fill=(255, 255, 255), font=ImageFont.load_default())
    return img


def contact_sheet(frames, cols=3) -> Image.Image:
    w, h = frames[0].size
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, ((i % cols) * w, (i // cols) * h))
    return sheet


def save_gif(frames, path, fps=10):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
