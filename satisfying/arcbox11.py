"""Volle Länge (~50 s, nahtloser Loop) der drei neuen Systeme aus arcbox10:
a18 Farbstempel, a19 Mühle, a20 Panzer-Bälle. Dramaturgie wie a14: schneller Hook, Spannungsphase, Explosion."""
import math
import sys

import numpy as np
from PIL import Image

from . import arcbox as A
from . import arcbox3 as C
from . import arcbox9 as H
from . import arcbox10 as I
from .arcbox import FPS


class Full:
    DUR = None
    OM = 2 * math.pi / 5.0
    GAP = 0.2
    allow = H.ArcTriple.allow


class StampFull(Full, I.Stamp):
    N_FULL = 900
    GEN0 = [0, 1, 2]                 # Start: ein roter, ein cyan, ein gelber Ball (Arm-Farben)
    STAMP = I.STAMP[:4]


class MillFull(Full, I.Mill):
    N_FULL = 700
    R = 14.0
    SPEED = 320.0
    T_FULL = 42.0


class ArmorFull(Full, I.Armor):
    N_FULL = 520
    N_HOOK = 9
    T_FULL = 44.0
    GAP = 0.08


CONFIGS = {
    "farbstempel": dict(cls=StampFull, out="output/a18_farbstempel_vertical.mp4", seed=11),
    "muehle": dict(cls=MillFull, out="output/a19_muehle_vertical.mp4", seed=0),
    "panzer": dict(cls=ArmorFull, out="output/a20_panzer_vertical.mp4", seed=5),
}


def build(name):
    cfg = CONFIGS[name]
    I.CONFIGS[name]["theme"]()
    if name == "farbstempel":            # nur die 4 Arm-Farben (Loop-Farbzyklus = 4)
        A_ = I.STAMP[:4]
        I.B.RAINBOW[:] = A_
        C.RAINBOW[:] = A_
    cls = cfg["cls"]
    if name == "muehle":
        SP = np.array([(C.X0 + 250, C.Y0 + 80), (C.X0 + 375, C.Y0 + 50), (C.X0 + 500, C.Y0 + 90)])
        SV = np.array([(-120.0, 0.0), (60.0, 0.0), (150.0, 0.0)])
    else:
        _, SP, SV = H.find_starts(cfg["seed"], lo=0.4, hi=0.9, cls=cls)
    sim = cls(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim


if __name__ == "__main__":
    name = sys.argv[1]
    sim = build(name)
    sp = [e for e in sim.events if e[1] == "split"]
    print(name, "voll", sim.full_t, "Loop", sim.loop_t, "Cracks", len(sp), "erster", round(sp[0][0], 2),
          "Bälle", [int(np.sum(sim.frames[min(sim.n_frames - 1, q * FPS)][5] > 0.5)) for q in range(0, 60, 4)])
    Rcls = I.ArmorRenderer if name == "panzer" else H.MetalRenderer
    if "--preview" in sys.argv:
        R = Rcls(sim)
        ts = [1.0, 8.0, 20.0, 35.0, sim.full_t - 0.5, sim.loop_t - 0.3]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, x in enumerate(ts):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save(f"output/preview/{name}_full_check.png")
    else:
        A.ArcRenderer, A.build_audio = Rcls, H.audio
        A.render(sim, CONFIGS[name]["out"])
        print(CONFIGS[name]["out"])
