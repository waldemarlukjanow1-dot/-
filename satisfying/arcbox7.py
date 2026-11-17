"""Arc Squeeze a12: Spannungsaufbau – EIN Ball fliegt mehrere Sekunden herum, knappe Beinahe-Quetschungen,
erste echte Quetschung erst nach ~4 s. Rizz-Sound nur bei Quetschungen (jede, überlappend).
Farbwelt: tiefes Grün-Schwarz, minzweißer Rahmen, pinke Bögen, kühle Bälle (Cyan, Blau, Limette, Weiß ...)."""
import math
import numpy as np
from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox5 as E
from . import arcbox6 as F

THEME = dict(bg=(3, 14, 9), bg_glow=(8, 36, 24), frame=(205, 255, 230), frame_glow=(60, 230, 160),
             pivot=(10, 40, 25), text=(225, 255, 240))
ARCS = dict(NEON=(255, 40, 150), NEON_HI=(255, 205, 235), NEON_D=(170, 0, 90), NEON_DD=(70, 0, 40))
BALLS = [(0, 230, 255), (60, 130, 255), (170, 255, 60), (245, 255, 250), (255, 230, 40), (140, 90, 255),
         (40, 255, 170), (255, 140, 40)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS


class ArcLoopWait(E.ArcLoopRizz):
    T_FULL = 46.0
    N_FULL = 900


def probe(sim, T):
    """wie C.probe, aber exakt wie der echte Lauf (Wände nach dem Bogen, Tempo konstant halten)."""
    r = sim.R
    SP, SV = sim.STARTS
    P, V = SP.copy(), SV.copy()
    n = len(P)
    hit = np.zeros(n, np.int64)
    dt = 1.0 / (A.FPS * sim.SUB)
    t = 0.0
    while t < T:
        P += V * dt
        hit[:] = 0
        for k, (qx, qy) in enumerate(A.CORNERS):
            cx_, cy_, a0 = sim.arc_geom(k, t)
            A.arc_hit(P, V, n, r, cx_, cy_, a0, sim.OM, hit, qx, qy, C.ARC_R, k + 1)
        wallnear = ((P[:, 0] < C.X0 + r + 2.5) | (P[:, 0] > C.X1 - r - 2.5) | (P[:, 1] < C.Y0 + r + 2.5) |
                    (P[:, 1] > C.Y1 - r - 2.5))
        if (hit.astype(bool) & wallnear).any():
            return t
        for _ in range(2):
            A.box_walls(P, V, n, r)
        A.keep_speed(V, n, sim.SPEED)
        t += dt
    return None


def find_starts(seed, lo=3.6, hi=4.6, tries=1500):
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        SP = np.array([(rng.uniform(C.X0 + 60, C.X1 - 60), rng.uniform(C.Y0 + 60, C.Y1 - 60))])
        a = rng.uniform(0, 2 * math.pi, 1)
        SV = np.stack([np.cos(a), np.sin(a)], 1) * ArcLoopWait.SPEED
        sim = ArcLoopWait(1)
        sim.STARTS = (SP, SV)
        first = probe(sim, hi + 0.2)
        if first is not None and lo <= first <= hi:
            return first, SP, SV
    raise RuntimeError("kein Start gefunden")


def build(seed=21):
    apply_theme()
    first, SP, SV = find_starts(seed)
    sim = ArcLoopWait(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


if __name__ == "__main__":
    import sys
    from PIL import Image
    sim, first = build()
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"),
          "Bälle", [len(sim.frames[min(sim.n_frames - 1, q * A.FPS)][0]) for q in range(0, 60, 4)])
    if "--preview" in sys.argv:
        R = C.ShortRenderer2(sim)
        ts = [1.0, 3.0, first + 0.3, 20.0, 35.0, sim.loop_t - 0.4]
        ims = [R.frame(min(sim.n_frames - 1, int(x * A.FPS))) for x in ts]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a12_check.png")
    else:
        old_r, old_a = A.ArcRenderer, A.build_audio
        A.ArcRenderer, A.build_audio = C.ShortRenderer2, F.audio
        try:
            A.render(sim, "output/a12_arc_squeeze_wait_vertical.mp4")
        finally:
            A.ArcRenderer, A.build_audio = old_r, old_a
        print("output/a12_arc_squeeze_wait_vertical.mp4")
