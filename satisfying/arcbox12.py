"""Arc Squeeze a21 – komplett neues Design "Holzspielzeug" (helles Design statt Neon):
cremefarbenes Papier, Rahmen aus dunklem Nussholz, Arme aus hellem Holz mit Maserung und Messing-Nieten,
Bälle wie lackierte Holzperlen (Tomate, Sonnengelb, Petrol, Kobalt, Pflaume). Kein Leuchten.
System wie a14: Split in 3, schwergängige Arme, Hook -> Spannung -> Explosion, nahtloser Loop."""
import math

import numpy as np
from PIL import Image

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox9 as H
from .arcbox import ARC_R, ARC_W, SPAN, FPS

THEME = dict(bg=(240, 231, 214), bg_glow=(10, 9, 6), frame=(92, 60, 38), frame_glow=(0, 0, 0),
             pivot=(214, 178, 92), text=(78, 52, 34))
ARCS = dict(NEON=(170, 120, 70), NEON_HI=(232, 196, 148), NEON_D=(110, 72, 40), NEON_DD=(80, 50, 28))
BALLS = [(226, 72, 58), (240, 176, 30), (18, 150, 140), (44, 92, 196), (146, 62, 140)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS
    B.BLOOM = 0.0


class WoodRenderer(H.MetalRenderer):
    GLOW_LEVELS = (0.0, 0.0, 0.0, 0.0)
    TRAIL = 2

    def arc_layer(self, d, angs, alpha, pulse):
        k = A.SS
        for idx, (px, py, a) in enumerate(angs):
            p = pulse[idx] * 0.25

            def bb(dr):
                return [(px - ARC_R - dr) * k, (py - ARC_R - dr) * k, (px + ARC_R + dr) * k, (py + ARC_R + dr) * k]

            def mix(c):
                return tuple(int(v + (255 - v) * p) for v in c) + (255,)
            a0, a1 = math.degrees(a), math.degrees(a + SPAN)
            d.arc(bb(0), a0, a1, fill=(96, 62, 34, 255), width=int((ARC_W + 5) * k))      # dunkle Kante
            d.arc(bb(0), a0, a1, fill=mix((200, 146, 88)), width=int(ARC_W * k))           # Holz
            for off, col, w in ((-5.0, (226, 184, 128), 2.2), (-1.5, (176, 122, 70), 1.2),
                                (2.0, (184, 130, 76), 1.4), (5.0, (160, 108, 60), 1.6)):  # Maserung
                d.arc(bb(off), a0 + 0.4, a1 - 0.4, fill=mix(col), width=max(1, int(w * k)))
            for e in (a, a + SPAN):                                                       # Messing-Nieten
                ex, ey = px + math.cos(e) * ARC_R, py + math.sin(e) * ARC_R
                rr = ARC_W / 2 + 2
                d.ellipse([(ex - rr) * k, (ey - rr) * k, (ex + rr) * k, (ey + rr) * k], fill=(200, 146, 88, 255),
                          outline=(96, 62, 34, 255), width=2 * k)
                d.ellipse([(ex - 4.5) * k, (ey - 4.5) * k, (ex + 4.5) * k, (ey + 4.5) * k], fill=(222, 186, 96, 255),
                          outline=(140, 104, 40, 255), width=k)


def build(seed=9):
    apply_theme()
    first, SP, SV = H.find_starts(seed)
    sim = H.ArcTriple(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


if __name__ == "__main__":
    import sys
    sim, first = build()
    print("erster Crack", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Bälle", [int(np.sum(sim.frames[min(sim.n_frames - 1, q * FPS)][5] > 0.5)) for q in range(0, 60, 4)])
    if "--preview" in sys.argv:
        R = WoodRenderer(sim)
        ts = [0.5, 8.0, 22.0, 38.0, sim.full_t + 1, sim.loop_t - 0.3]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, x in enumerate(ts):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a21_check.png")
    else:
        A.ArcRenderer, A.build_audio = WoodRenderer, H.audio
        A.render(sim, "output/a21_holzspielzeug_vertical.mp4")
        print("output/a21_holzspielzeug_vertical.mp4")
