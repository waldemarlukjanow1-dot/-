"""Greifer (Vorbild: "Satisfying ball crusher", runde Arena mit zwei Greifarmen).
  * runde Neon-Arena, darin zwei L-förmige Greifer (lange Stange vom Rand + kurzer Finger am Gelenk),
    punktsymmetrisch; die ganze Mechanik dreht sich langsam im Uhrzeigersinn
  * Takt: offen (Stangen weit außen am Rand) -> schließen zu einem Rechteck in der Mitte (treibt die Bälle
    dazwischen zusammen) -> SCHNAPPEN (Rechteck klappt zu einem Schlitz zusammen) -> wieder öffnen
  * ein Ball wird NUR zerquetscht, wenn der Greifer ihn in der Mitte schnappt; genau dort teilt er sich
    in 3 (nächste Farbe), die an den Enden herausgedrückt werden
  * Ton: Deep Meow bei jeder Quetschung (überlappend), keine Schrift außer der Ballzahl
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

from . import arcbox as A
from . import arcbox6 as F
from . import jelly as J
from .arcbox import CX, CY, HALF, FPS, SS


class Greifer(J.Drehkreuz):
    SPEED = 200.0
    R = 14.0
    N8 = 320
    GAP = 0.3
    RC = HALF                    # Arena-Radius
    BAR = 8.0                    # halbe Stangendicke
    H = 0.58 * HALF              # halbe Länge des Mittel-Rechtecks
    D_RECT = 0.13 * HALF         # halbe Breite des Rechtecks
    D_OPEN = 0.62 * HALF
    D_SNAP = 8.0 + 14.0 * 0.55    # Schlitz enger als ein Ball: quetscht, ohne dass er durch die Stange rutscht
    P = 2.0                      # Takt in s
    OM_ROT = 2 * math.pi / 12.0  # Drehung der ganzen Mechanik (im Uhrzeigersinn)
    TH0 = 2.0

    def gap(self, t):
        """halber Abstand d der beiden Stangen im Takt."""
        u = (t / self.P) % 1.0
        o, rc, sn = self.D_OPEN, self.D_RECT, self.D_SNAP
        if u < 0.30:                                   # offen -> Rechteck (zusammentreiben)
            e = u / 0.30
            s = e * e * (3 - 2 * e)
            return o + (rc - o) * s
        if u < 0.40:                                   # kurz als Rechteck halten
            return rc
        if u < 0.47:                                   # SCHNAPP!
            e = (u - 0.40) / 0.07
            return rc + (sn - rc) * e ** 1.6
        if u < 0.57:                                   # zugeschnappt halten
            return sn
        e = (u - 0.57) / 0.43                          # wieder öffnen
        s = e * e * (3 - 2 * e)
        return sn + (o - sn) * s

    def geom(self, t):
        th = self.TH0 + self.OM_ROT * t
        ux, uy = math.cos(th), math.sin(th)
        wx, wy = -uy, ux
        d = self.gap(t)
        h = self.H
        U = math.sqrt(max(10.0, self.RC ** 2 - d * d)) - 4
        Lf = 2 * self.D_RECT
        segs = []

        def pt(a, b):
            return (CX + ux * a + wx * b, CY + uy * a + wy * b)
        # Greifer A (oben): Stange bei w=+d von u=-h bis zum Rand, Finger am inneren Ende Richtung -w
        # Greifer B (unten): punktsymmetrisch
        for sgn in (1, -1):
            a0 = pt(-h * sgn, d * sgn)
            a1 = pt(U * sgn, d * sgn)
            if 2 * d <= Lf:
                along = math.sqrt(max(0.0, Lf * Lf - (2 * d) ** 2))
                f1 = pt((-h + along) * sgn, -d * sgn)      # Finger klappt flach ins Rechteck
            else:
                f1 = pt(-h * sgn, (d - Lf) * sgn)          # offen: Finger zeigt nach innen
            segs.append((a0, a1))
            segs.append((a0, f1))
        return segs, (th, d)

    def parts(self, t):
        eps = 1e-3
        s0, _ = self.geom(t)
        s1, _ = self.geom(t + eps)
        out = []
        for (a, b), (a2, b2) in zip(s0, s1):
            ang0 = math.atan2(b[1] - a[1], b[0] - a[0])
            ang1 = math.atan2(b2[1] - a2[1], b2[0] - a2[0])
            dang = (ang1 - ang0 + math.pi) % (2 * math.pi) - math.pi
            out.append((a[0], a[1], b[0], b[1], self.BAR, a[0], a[1], dang / eps,
                        (a2[0] - a[0]) / eps, (a2[1] - a[1]) / eps))
        return out

    def crushed(self, P, n, r, pen, pen0, nx0, ny0):
        th, d = self.geom_now
        if d > self.D_RECT * 0.95:                         # nur beim Schnappen
            return np.zeros(0, np.int64)
        q = P[:n] - (CX, CY)
        uu = q[:, 0] * math.cos(th) + q[:, 1] * math.sin(th)
        ww = -q[:, 0] * math.sin(th) + q[:, 1] * math.cos(th)
        inside = (np.abs(uu) < self.H + r * 0.5) & (np.abs(ww) < self.D_RECT + r)
        c = np.nonzero(((pen[:n] > self.CRUSH * r) | (pen0[:n] > 1.0)) & inside)[0]
        return c[np.argsort(-pen[c])]

    def machine_speed(self, t):
        return 1.0

    N0 = 12

    def allow(self, n, t):
        return n + 2 < self.N0 + self.N8 * (min(t, self.DUR) / self.DUR) ** 1.2 + 3

    def starts(self):
        rng = self.rng
        rr = np.sqrt(rng.uniform(0.02, 0.85, self.N0)) * (self.RC - 40)
        aa = rng.uniform(0, 2 * math.pi, self.N0)
        SP = np.stack([CX + np.cos(aa) * rr, CY + np.sin(aa) * rr], 1)
        a = rng.uniform(0, 2 * math.pi, len(SP))
        return SP, np.stack([np.cos(a), np.sin(a)], 1) * self.SPEED


# geom_now für crushed(): vor jedem Teilschritt aktualisieren
_orig_parts = Greifer.parts


def _parts_with_state(self, t):
    self.geom_now = self.geom(t)[1]
    return _orig_parts(self, t)


Greifer.parts = _parts_with_state


class GreiferRenderer(J.JellyRenderer):
    THEME = dict(bg=(16, 4, 10), bg_glow=(46, 10, 26), frame=(255, 50, 90), text=(255, 232, 236), round=True,
                 balls=[(60, 240, 90), (255, 220, 40), (40, 200, 255), (255, 90, 200), (255, 140, 30),
                        (170, 110, 255)])

    def draw_machine(self, d, parts, fi):
        k = SS
        for (ax, ay, bx, by, rad, *_r) in parts:
            J.glossy_capsule(d, ax, ay, bx, by, rad, (255, 196, 70), (255, 245, 200), (150, 90, 10))
        for (ax, ay, *_r) in parts[::2]:                 # Gelenke
            for rr, col in ((11, (150, 90, 10)), (8, (255, 225, 140)), (3, (150, 90, 10))):
                d.ellipse([(ax - rr) * k, (ay - rr) * k, (ax + rr) * k, (ay + rr) * k], fill=col + (255,))


def audio(sim, path):
    F.audio(sim, path, sample="assets/meow.wav", vol=0.55)


if __name__ == "__main__":
    sim = Greifer(3).run()
    sp = [e for e in sim.events if e[1] == "split"]
    print("Quetschungen", len(sp), "Zeiten", [round(e[0], 2) for e in sp][:12],
          "Bälle", [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(9)])
    if "--preview" in sys.argv:
        R = GreiferRenderer(sim)
        ts = [0.3, 0.8, 1.0, 1.1, 1.3, 3.0, 5.0, 6.5, 7.9]
        out = Image.new("RGB", (360 * 3, 640 * 3))
        for i, x in enumerate(ts):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/greifer_check.png")
    else:
        A.ArcRenderer, A.build_audio = GreiferRenderer, audio
        A.render(sim, "output/g01_greifer_miau_8s.mp4")
        print("output/g01_greifer_miau_8s.mp4")
