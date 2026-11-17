"""Arc Squeeze – neue Farbwelt (Synthwave: Magenta-Bögen, eisblauer Rahmen, Cyan->Violett-Bälle),
langsamerer Start (2 Bälle) und längeres Video (~30 s), sonst wie die viral-optimierte Loop-Version."""
import math

import numpy as np
from PIL import Image

from . import arcbox2 as B
from . import arcbox3 as C
from .arcbox import FPS

THEME = dict(bg=(8, 5, 20), bg_glow=(20, 8, 34), frame=(205, 235, 255), frame_glow=(90, 150, 255),
             pivot=(35, 10, 55), text=(230, 240, 255))
ARCS = dict(NEON=(255, 45, 200), NEON_HI=(255, 205, 240), NEON_D=(150, 10, 110), NEON_DD=(70, 5, 50))
BALLS = [(0, 235, 255), (40, 255, 190), (190, 255, 40), (255, 215, 0), (255, 140, 30), (255, 80, 90),
         (160, 90, 255), (90, 150, 255)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS


class ArcLoopLong(C.ArcLoopShort):
    R = 16.0
    N_FULL = 860
    CAP = 1100
    T_FULL = 22.0
    GAP = 0.12
    SPEED = 540.0
    OM = 2 * math.pi / (150 / 60)      # eine Umdrehung = 150 Bilder (2,5 s)


def find_starts(seed, n_balls=2, tries=300, lo=0.8, hi=1.6):
    """Start so wählen, dass die erste Quetschung etwas später kommt (ruhigerer Einstieg, ~1 s)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        SP = np.array([(rng.uniform(C.X0 + 50, C.X1 - 50), rng.uniform(C.Y0 + 50, C.Y1 - 50)) for _k in range(n_balls)])
        a = rng.uniform(0, 2 * math.pi, n_balls)
        SV = np.stack([np.cos(a), np.sin(a)], 1) * ArcLoopLong.SPEED
        sim = ArcLoopLong(1)
        sim.STARTS = (SP, SV)
        first = C.probe(sim, hi + 0.2)
        if first is not None and lo <= first <= hi:
            return first, SP, SV
        if first is not None and (best is None or abs(first - 1.1) < abs(best[0] - 1.1)):
            best = (first, SP, SV)
    return best


def build():
    apply_theme()
    first, SP, SV = find_starts(5)
    sim = ArcLoopLong(2)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


if __name__ == "__main__":
    import sys
    sim, first = build()
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(0, 34, 3)])
    if "--preview" in sys.argv:
        R = C.ShortRenderer2(sim)
        ts = [0.3, first + 0.05, 8.0, 16.0, sim.full_t + 0.8, sim.loop_t - 0.4]
        ims = [R.frame(min(sim.n_frames - 1, int(x * FPS))) for x in ts]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a08_check.png")
        ims2 = [R.frame(i) for i in (sim.n_frames - 2, sim.n_frames - 1, 0, 1)]
        out = Image.new("RGB", (360 * 4, 640))
        for i, im in enumerate(ims2):
            out.paste(im.resize((360, 640)), (i * 360, 0))
        out.save("output/preview/a08_seam.png")
    else:
        C.render(sim, "output/a08_arc_squeeze_synthwave_vertical.mp4")
        print("output/a08_arc_squeeze_synthwave_vertical.mp4")
