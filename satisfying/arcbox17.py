"""Arc Squeeze a27 – Halbkreis-Miau-Video (wie a13/a25) mit neuen Farben und NEUEN FORMEN:
jede Generation hat eine eigene Form + Farbe: Herz, Stern, Diamant, Würfel, Dreieck, Sechseck, Tropfen, Kreis.
Physik unverändert (Quetschung an der Wand verdoppelt), Deep Meow bei jeder Quetschung, nahtloser Loop.
Farbwelt: tiefes Indigo, goldener Rahmen, magenta Neon-Bögen; Halbkreise starten versetzt (alle im Uhrzeigersinn)."""
import math
from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox6 as F
from . import arcbox7 as G

THEME = dict(bg=(10, 5, 28), bg_glow=(30, 16, 70), frame=(255, 214, 120), frame_glow=(255, 170, 40),
             pivot=(40, 26, 6), text=(255, 240, 210))
ARCS = dict(NEON=(255, 40, 200), NEON_HI=(255, 200, 245), NEON_D=(160, 0, 120), NEON_DD=(70, 0, 55))
BALLS = [(255, 50, 80), (255, 215, 40), (40, 220, 255), (120, 240, 70), (255, 140, 30), (170, 100, 255),
         (60, 150, 255), (250, 250, 255)]
SHAPES = ["heart", "star", "diamond", "square", "triangle", "hexagon", "drop", "circle"]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS


def _poly(shape, R, cx, cy):
    if shape == "star":
        return [(cx + math.cos(-math.pi / 2 + i * math.pi / 5) * (R * 1.18 if i % 2 == 0 else R * 0.5),
                 cy + math.sin(-math.pi / 2 + i * math.pi / 5) * (R * 1.18 if i % 2 == 0 else R * 0.5))
                for i in range(10)]
    if shape == "diamond":
        return [(cx, cy - R * 1.2), (cx + R * 0.95, cy), (cx, cy + R * 1.2), (cx - R * 0.95, cy)]
    if shape == "triangle":
        return [(cx + math.cos(-math.pi / 2 + i * 2 * math.pi / 3) * R * 1.25,
                 cy + R * 0.15 + math.sin(-math.pi / 2 + i * 2 * math.pi / 3) * R * 1.25) for i in range(3)]
    if shape == "hexagon":
        return [(cx + math.cos(i * math.pi / 3) * R * 1.05, cy + math.sin(i * math.pi / 3) * R * 1.05) for i in range(6)]
    if shape == "heart":
        pts = []
        for i in range(60):
            a = 2 * math.pi * i / 60
            x = 16 * math.sin(a) ** 3
            y = -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a))
            pts.append((cx + x * R / 15.0, cy + (y + 2.5) * R / 15.0))
        return pts
    if shape == "drop":
        pts = []
        for i in range(60):
            a = 2 * math.pi * i / 60
            x = math.sin(a) * (1 - math.cos(a)) * 0.62
            y = -math.cos(a)
            pts.append((cx + x * R * 1.05, cy - y * R * 1.15 + R * 0.05))
        return pts
    return None


def shape_sprite(r, col, glow_amt, shape):
    """glänzende Form (Größe 5 r wie die Ball-Sprites), Zentrum = Ballmittelpunkt."""
    n = int(r * 5)
    up = 4
    m = Image.new("L", (n * up, n * up), 0)
    d = ImageDraw.Draw(m)
    c0 = n * up / 2
    R = r * up
    if shape == "circle":
        d.ellipse([c0 - R, c0 - R, c0 + R, c0 + R], fill=255)
    elif shape == "square":
        d.rounded_rectangle([c0 - R * 0.92, c0 - R * 0.92, c0 + R * 0.92, c0 + R * 0.92], radius=R * 0.28, fill=255)
    else:
        d.polygon(_poly(shape, R, c0, c0), fill=255)
    mask = np.asarray(m.resize((n, n), Image.LANCZOS), np.float32) / 255
    er = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5)), np.float32) / 255
    rim = np.clip(mask - er, 0, 1)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2 + 0.5
    c = np.array(col, np.float32)
    light = np.clip(1.0 - ((xx + r * 0.35) ** 2 + (yy + r * 0.45) ** 2) / (2.4 * r * r), 0, 1)
    rgb = c * (0.68 + 0.45 * light[..., None])
    rgb = rgb * (1 - 0.45 * rim[..., None])
    hard = np.exp(-(((xx + r * 0.33) / (r * 0.22)) ** 2 + ((yy + r * 0.38) / (r * 0.15)) ** 2)) * mask
    rgb = rgb + (hard * 0.9)[..., None] * (255 - rgb)
    gl = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r * 0.7)),
                    np.float32) / 255 * glow_amt * 0.9
    a = np.maximum(mask, gl)
    rgb = rgb * mask[..., None] + c * (1 - mask[..., None])
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA")


class ShapeRenderer(C.ShortRenderer2):
    def __init__(self, sim):
        super().__init__(sim)
        k = A.SS
        for g in self.GLOW_LEVELS:
            sps = [shape_sprite(sim.R * k * 1.05, c, g, SHAPES[i % len(SHAPES)]) for i, c in enumerate(BALLS)]
            self.sprites[g] = sps
            ghs = []
            for sp in sps:
                a = np.asarray(sp).copy()
                a[..., 3] = (a[..., 3] * 0.26).astype(np.uint8)
                ghs.append(Image.fromarray(a, "RGBA"))
            self.ghost[g] = ghs


if __name__ == "__main__":
    import sys
    G.apply_theme = apply_theme
    G.ArcLoopWait.PH = (0.3, 0.3 + 1.6, 0.3 + 3.9, 0.3 + 4.8)      # versetzt, alle im Uhrzeigersinn
    sim, first = G.build(seed=53)
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"),
          "erste", [round(e[0], 2) for e in sim.events if e[1] == "split"][:6])
    if "--preview" in sys.argv:
        R = ShapeRenderer(sim)
        ts = [2.0, 15.0, 30.0]
        out = Image.new("RGB", (1080, 640))
        for i, x in enumerate(ts):
            out.paste(R.frame(int(x * A.FPS)).resize((360, 640)), (i * 360, 0))
        out.save("output/preview/a27_check.png")
        R.frame(int(30 * A.FPS)).crop((250, 600, 750, 1100)).save("output/preview/a27_zoom.png")
    else:
        A.ArcRenderer, A.build_audio = ShapeRenderer, partial(F.audio, sample="assets/meow.wav", vol=0.55)
        A.render(sim, "output/a27_meow_formen_vertical.mp4")
        print("output/a27_meow_formen_vertical.mp4")
