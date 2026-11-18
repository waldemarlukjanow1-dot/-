"""Drei neue Systeme (je 8 s Probe) auf Basis von a14 (4 schwergängige Arme, Zerquetschen an der Wand, Split in 3):

  1) farbstempel  – jeder der 4 Arme hat seine eigene Farbe; wen ein Arm zerquetscht, der bekommt seine Farbe.
                    Die Box wird zu einem 4-farbigen Revier-Kampf (welche Farbe gewinnt?).
  2) muehle       – Schwerkraft an: die Bälle fallen nach unten und bilden einen Haufen, die Arme schaufeln
                    wie eine Mühle durch den Haufen und zermahlen ihn an Boden und Wänden.
  3) panzer       – harte Bälle: erst beim 3. Quetschen platzen sie (1. = Haarriss, 2. = großer Riss,
                    3. = Crack in 3 neue, wieder heile Bälle der nächsten Farbe).
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox9 as H
from .arcbox import FPS


def theme(bg, glow, frame, fglow, balls, text=(240, 244, 250)):
    def apply():
        B.THEME.update(dict(bg=bg, bg_glow=glow, frame=frame, frame_glow=fglow, pivot=(12, 14, 20), text=text))
        for k_, v in H.ARCS.items():
            setattr(B, k_, v)
        B.RAINBOW[:] = balls
        C.RAINBOW[:] = balls
    return apply


class Demo(H.ArcTriple):
    DUR = 8.0
    OM = 2 * math.pi / 4.5           # immer noch schwer, aber genug Bewegung für 8 s
    R = 13.0
    N_FULL = 10 ** 6                 # kein Loop-Ende
    N8 = 420
    GAP = 0.12

    def allow(self, n, t):
        return n + 2 < 3 + self.N8 * (min(t, self.DUR) / self.DUR) ** 1.7 + 3


# ---------------------------------------------------------------- 1) Farbstempel
STAMP = [(255, 60, 90), (40, 200, 255), (255, 210, 30), (80, 240, 120), (235, 235, 240)]


class Stamp(Demo):
    def crush(self, i, gen, hitval, t):
        return hitval - 1            # Farbe = Arm (0..3); Startbälle haben Farbe 4 (weiß)


# ---------------------------------------------------------------- 2) Mühle
class Mill(Demo):
    GRAV = 1500.0
    VMAX = 1100.0
    N8 = 380

    def forces(self, P, V, n, dt):
        V[:n, 1] += self.GRAV * dt

    def speed_rule(self, V, n):
        V[:n] *= 0.9993
        sp = np.hypot(V[:n, 0], V[:n, 1])
        f = np.minimum(1.0, self.VMAX / (sp + 1e-9))
        V[:n] *= f[:, None]


# ---------------------------------------------------------------- 3) Panzer-Bälle
ARMOR_COLS = [(255, 120, 40), (60, 230, 255), (255, 70, 200), (190, 255, 60), (150, 110, 255)]


class Armor(Demo):
    N8 = 300
    R = 17.0
    GAP = 0.07

    def crush(self, i, gen, hitval, t):
        stage = gen[i] % 3
        if stage < 2:
            gen[i] += 1
            return None
        return (gen[i] // 3 + 1) * 3


class ArmorRenderer(H.MetalRenderer):
    """Risse auf die Sprites malen: Stufe 1 = ein Haarriss, Stufe 2 = mehrere große Risse."""

    def __init__(self, sim):
        super().__init__(sim)
        r = sim.R * A.SS
        for g, sps in self.sprites.items():
            for idx, sp in enumerate(sps):
                stage = idx % 3
                if stage:
                    sps[idx] = self._crack(sp, r, stage, idx)

    @staticmethod
    def _crack(sp, r, stage, seed):
        sp = sp.copy()
        d = ImageDraw.Draw(sp, "RGBA")
        cx, cy = sp.width / 2, sp.height / 2
        rng = np.random.default_rng(seed)
        n_lines = 1 if stage == 1 else 4
        for j in range(n_lines):
            a = rng.uniform(0, 2 * math.pi) if j else 0.6
            pts = [(cx + rng.uniform(-0.15, 0.15) * r, cy + rng.uniform(-0.15, 0.15) * r)]
            L = r * (0.75 if stage == 1 else 0.95)
            for s in range(1, 5):
                a += rng.uniform(-0.6, 0.6)
                rr = L * s / 4
                pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
            d.line(pts, fill=(15, 10, 10, 235), width=max(3, int(r * (0.2 if stage == 2 else 0.15))))
        return sp


CONFIGS = {
    "farbstempel": dict(cls=Stamp, out="output/a15_idee_farbstempel_8s.mp4",
                        theme=theme((8, 8, 14), (20, 18, 32), (245, 245, 255), (160, 160, 255), STAMP)),
    "muehle": dict(cls=Mill, out="output/a16_idee_muehle_8s.mp4",
                   theme=theme((12, 8, 4), (34, 22, 10), (255, 180, 70), (255, 120, 20),
                               [(255, 235, 200), (255, 170, 60), (240, 90, 40), (200, 40, 60), (120, 30, 90)])),
    "panzer": dict(cls=Armor, out="output/a17_idee_panzer_8s.mp4",
                   theme=theme((4, 10, 12), (10, 30, 34), (120, 255, 200), (0, 220, 150),
                               [c for c in ARMOR_COLS for _ in range(3)])),
}


def build(name):
    cfg = CONFIGS[name]
    cfg["theme"]()
    cls = cfg["cls"]
    if name == "muehle":                      # Bälle starten oben und fallen
        SP = np.array([(C.X0 + 250, C.Y0 + 80), (C.X0 + 375, C.Y0 + 50), (C.X0 + 500, C.Y0 + 90)])
        SV = np.array([(-120.0, 0.0), (60.0, 0.0), (150.0, 0.0)])
    else:
        _, SP, SV = H.find_starts(7, lo=0.3, hi=0.8, cls=cls)
    sim = cls(3)
    sim.STARTS = (SP, SV)
    if name == "farbstempel":
        sim.GEN0 = 4
    sim.run()
    return sim


if __name__ == "__main__":
    name = sys.argv[1]
    sim = build(name)
    sp = [e for e in sim.events if e[1] == "split"]
    print("Risse", sum(1 for e in sim.events if e[1] == "crack"))
    print(name, "Bilder", sim.n_frames, "Cracks", len(sp), "erster", round(sp[0][0], 2) if sp else None,
          "Bälle", [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(0, 9)])
    Rcls = ArmorRenderer if name == "panzer" else H.MetalRenderer
    if "--preview" in sys.argv:
        R = Rcls(sim)
        out = Image.new("RGB", (360 * 3, 640))
        for i, x in enumerate((1.0, 4.0, 7.8)):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), (i * 360, 0))
        out.save(f"output/preview/{name}_check.png")
    else:
        A.ArcRenderer, A.build_audio = Rcls, H.audio
        A.render(sim, CONFIGS[name]["out"])
        print(CONFIGS[name]["out"])
