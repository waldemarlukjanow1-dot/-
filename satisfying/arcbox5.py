"""Arc Squeeze – lange Version (~50 s, nahtloser Loop) mit eigenem Sound (Rizz) bei jeder Quetschung
und neuer Farbwelt: tiefes Petrol, goldener Rahmen, orange Bögen, Bälle von Himbeere bis Zitrone."""
import math

import numpy as np
from PIL import Image

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox4 as D
from .arcbox import FPS

THEME = dict(bg=(3, 12, 16), bg_glow=(6, 30, 34), frame=(255, 228, 165), frame_glow=(255, 170, 60),
             pivot=(45, 28, 5), text=(255, 242, 215))
ARCS = dict(NEON=(255, 140, 20), NEON_HI=(255, 222, 175), NEON_D=(170, 70, 0), NEON_DD=(80, 30, 0))
BALLS = [(255, 60, 110), (255, 110, 200), (190, 110, 255), (90, 140, 255), (0, 210, 255), (40, 240, 160),
         (170, 255, 80), (255, 240, 90)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS


class ArcLoopRizz(C.ArcLoopShort):
    R = 15.0
    N_FULL = 1000
    CAP = 1300
    T_FULL = 44.0
    GAP = 0.25
    SPEED = 470.0
    OM = 2 * math.pi / (180 / 60)      # eine Umdrehung = 180 Bilder (3 s)


def find_starts(seed, n_balls=2, tries=400, lo=0.9, hi=1.8):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        SP = np.array([(rng.uniform(C.X0 + 50, C.X1 - 50), rng.uniform(C.Y0 + 50, C.Y1 - 50)) for _k in range(n_balls)])
        a = rng.uniform(0, 2 * math.pi, n_balls)
        SV = np.stack([np.cos(a), np.sin(a)], 1) * ArcLoopRizz.SPEED
        sim = ArcLoopRizz(1)
        sim.STARTS = (SP, SV)
        first = C.probe(sim, hi + 0.2)
        if first is not None and lo <= first <= hi:
            return first, SP, SV
        if first is not None and (best is None or abs(first - 1.2) < abs(best[0] - 1.2)):
            best = (first, SP, SV)
    return best


def build(seed=7):
    apply_theme()
    first, SP, SV = find_starts(seed)
    sim = ArcLoopRizz(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


def audio(sim, path):
    # mit vielen Quetschungen: höchstens ~3 Sounds pro Sekunde, damit es nicht chaotisch wird
    D.build_sample_audio(sim, path, min_gap=0.32, vol=0.5)


if __name__ == "__main__":
    import sys
    sim, first = build()
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(0, 60, 5)])
    if "--preview" in sys.argv:
        R = C.ShortRenderer2(sim)
        ts = [0.3, 10.0, 20.0, 32.0, sim.full_t + 0.8, sim.loop_t - 0.4]
        ims = [R.frame(min(sim.n_frames - 1, int(x * FPS))) for x in ts]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a10_check.png")
    else:
        old_r, old_a = A.ArcRenderer, A.build_audio
        A.ArcRenderer, A.build_audio = C.ShortRenderer2, audio
        try:
            A.render(sim, "output/a10_arc_squeeze_rizz_vertical.mp4")
        finally:
            A.ArcRenderer, A.build_audio = old_r, old_a
        print("output/a10_arc_squeeze_rizz_vertical.mp4")
