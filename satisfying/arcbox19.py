"""Arc Squeeze a29 – Halbkreis-Video mit NEUER Halbkreis-Form und MEHR Bällen:
  * Halbkreise als glänzende Perlenketten: eine weiche Schiene, darauf leuchtende Perlen mit Farbverlauf
    (Gold -> Pink), die ruhig entlang des Bogens wandern – auffällig, aber smooth
  * kleinere Jelly-Bälle (R 12 statt 15) -> das Quadrat füllt sich mit ~1400 Bällen statt 900
  * neue Farben "Candy Pop": dunkle Beere, rosé-weißer Rahmen, Jellys in Pastell-Bonbonfarben
  * Sound: "FAHH" bei jeder Quetschung (überlappend); Halbkreise versetzt, alle im Uhrzeigersinn, nahtloser Loop
"""
import math
import sys
from functools import partial

import numpy as np
from PIL import Image

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox6 as F
from . import arcbox7 as G
from .arcbox import ARC_R, ARC_W, SPAN, SS, FPS
from .jelly import jelly_sprite

THEME = dict(bg=(22, 7, 18), bg_glow=(62, 18, 48), frame=(255, 215, 232), frame_glow=(255, 120, 190),
             pivot=(50, 14, 36), text=(255, 236, 246))
ARCS = dict(NEON=(255, 120, 150), NEON_HI=(255, 230, 240), NEON_D=(140, 30, 80), NEON_DD=(60, 10, 35))
BALLS = [(120, 240, 200), (110, 190, 255), (195, 160, 255), (255, 240, 120), (255, 180, 140), (250, 250, 255),
         (90, 230, 240), (255, 130, 150)]
C0, C1 = np.array((255, 205, 60), float), np.array((255, 60, 160), float)


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS
    B.BLOOM = 0.25


class PearlRenderer(C.ShortRenderer2):
    TRAIL = 2
    NP = 30                       # Perlen pro Halbkreis (dicht an dicht)

    def __init__(self, sim):
        super().__init__(sim)
        sps = [jelly_sprite(sim.R * SS * 1.05, c) for c in BALLS]
        ghs = []
        for sp in sps:
            a = np.asarray(sp).copy()
            a[..., 3] = (a[..., 3] * 0.22).astype(np.uint8)
            ghs.append(Image.fromarray(a, "RGBA"))
        for g in self.GLOW_LEVELS:
            self.sprites[g] = sps
            self.ghost[g] = ghs
        self._fi = 0

    def frame(self, fi):
        self._fi = fi
        return super().frame(fi)

    def arc_layer(self, d, angs, alpha, pulse):
        k = SS
        t = (self._fi + 1) / FPS
        flow = (t * 0.6) % 1.0                         # Perlen wandern ruhig den Bogen entlang (Loop: 51 s * 0.6 = ganzzahlig? egal: Phase wird unten periodisch gemacht)
        flow = (self._fi / self.sim.n_frames * 30) % 1.0   # exakt 30 Perlen-Schritte pro Loop -> nahtlos
        for idx, (px, py, a) in enumerate(angs):
            p = pulse[idx] * 0.4

            def bb(dr):
                return [(px - ARC_R - dr) * k, (py - ARC_R - dr) * k, (px + ARC_R + dr) * k, (py + ARC_R + dr) * k]

            def col(u, light=0.0, dark=0.0, al=255):
                c = C0 + (C1 - C0) * u
                c = c + (255 - c) * min(1.0, light + p)
                c = c * (1 - dark)
                return tuple(int(v) for v in np.clip(c, 0, 255)) + (al,)
            a0, a1 = math.degrees(a), math.degrees(a + SPAN)
            d.arc(bb(0), a0, a1, fill=(40, 12, 30, 255), width=int((ARC_W + 2) * k))        # Schiene
            d.arc(bb(0), a0, a1, fill=(120, 60, 95, 255), width=int(ARC_W * 0.45 * k))
            rp = ARC_W * 0.72
            for j in range(self.NP + 1):
                u = (j + flow) / self.NP
                if u > 1.0:
                    continue
                e = a + SPAN * u
                x, y = px + math.cos(e) * ARC_R, py + math.sin(e) * ARC_R
                for rr, c in ((rp + 2.2, col(u, dark=0.6)), (rp, col(u)), (rp * 0.55, col(u, light=0.45))):
                    d.ellipse([(x - rr) * k, (y - rr) * k, (x + rr) * k, (y + rr) * k], fill=c)
                d.ellipse([(x - rp * 0.55) * k, (y - rp * 0.62) * k, (x - rp * 0.1) * k, (y - rp * 0.22) * k],
                          fill=(255, 255, 255, 230))
            for e, u in ((a, 0.0), (a + SPAN, 1.0)):     # große Endperlen
                x, y = px + math.cos(e) * ARC_R, py + math.sin(e) * ARC_R
                R_ = ARC_W * 0.85
                for rr, c in ((R_ + 2.5, col(u, dark=0.6)), (R_, col(u)), (R_ * 0.5, col(u, light=0.5))):
                    d.ellipse([(x - rr) * k, (y - rr) * k, (x + rr) * k, (y + rr) * k], fill=c)
                d.ellipse([(x - R_ * 0.55) * k, (y - R_ * 0.62) * k, (x - R_ * 0.1) * k, (y - R_ * 0.2) * k],
                          fill=(255, 255, 255, 235))


def build():
    G.apply_theme = apply_theme
    W = G.ArcLoopWait
    W.PH = (0.3, 0.3 + 1.2, 0.3 + 3.0, 0.3 + 4.4)         # versetzt, alle im Uhrzeigersinn
    W.R = 12.0
    W.N_FULL = 1400
    W.CAP = 2000
    W.GAP = 0.2
    return G.build(seed=61)


if __name__ == "__main__":
    sim, first = build()
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"),
          "Bälle", [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(0, 60, 5)])
    if "--preview" in sys.argv:
        R = PearlRenderer(sim)
        out = Image.new("RGB", (1080, 640))
        for i, x in enumerate((2.0, 20.0, sim.full_t + 0.5)):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), (i * 360, 0))
        out.save("output/preview/a29_check.png")
        R.frame(int(20 * FPS)).crop((100, 250, 950, 1150)).save("output/preview/a29_zoom.png")
    else:
        A.ArcRenderer, A.build_audio = PearlRenderer, partial(F.audio, sample="assets/fahh.wav", vol=0.6)
        A.render(sim, "output/a29_perlen_fahh_vertical.mp4")
        print("output/a29_perlen_fahh_vertical.mp4")
