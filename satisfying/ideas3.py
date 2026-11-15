"""Fünf neue 8-Sekunden-Ideen (Wünsche aus PREFERENCES.md: keine Überschrift, Zähler unten, keine Ring-Effekte,
kein Schluss-Sound, Bälle bewegen sich nur durch die Maschine, Verdoppeln nur beim echten Einklemmen).

  roller  : eine schwere Walze rollt über die Bodenplatte hin und her und walzt die Bälle platt
  scissor : zwei Klingen schließen sich wie eine Schere zu einem V
  cam     : eine exzentrische Scheibe dreht sich über dem Boden und drückt periodisch auf die Bälle
  grinder : ein dreiarmiger Stern dreht sich an festen Zähnen vorbei (wie ein Mahlwerk)
  rams    : zwei Rammböcke schieben von links und rechts über den Boden zusammen
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from .multiply import CX, CY, R_AR, FPS, SS, poly_body, circle_body, rect_poly
from . import ideas2 as I2
from .ideas2 import Base, LAV, LAV_HI, LAV_D, GOLD, GOLD_HI, GOLD_D

FLOOR = CY + 300


def chord(y):
    return math.sqrt(max(0.0, R_AR ** 2 - (y - CY) ** 2))


def flat_floor(y=FLOOR):
    hw = chord(y) + 60
    return np.array([(CX - hw, y), (CX + hw, y), (CX + hw, CY + R_AR + 80), (CX - hw, CY + R_AR + 80)], np.float64)


def v_floor(half, y=FLOOR, rise=150):
    """Flacher Mittelteil + zwei Schrägen nach außen (Bälle rollen zur Mitte)."""
    h = half
    mid = np.array([(CX - h, y), (CX + h, y), (CX + h, CY + R_AR + 80), (CX - h, CY + R_AR + 80)], np.float64)
    L = np.array([(CX - h, y), (CX - R_AR - 40, y - rise), (CX - R_AR - 40, CY + R_AR + 80), (CX - h, CY + R_AR + 80)],
                 np.float64)
    R = np.array([(CX + h, y), (CX + h, CY + R_AR + 80), (CX + R_AR + 40, CY + R_AR + 80), (CX + R_AR + 40, y - rise)],
                 np.float64)
    return [mid, L, R]


def disc(cx, cy, r, m=28):
    return np.array([(cx + math.cos(2 * math.pi * q / m) * r, cy + math.sin(2 * math.pi * q / m) * r) for q in range(m)],
                    np.float64)


class Idea(Base):
    def floors(self):
        return []

    def static(self, P, V, n, touch, impact):
        for q in self.floors():
            poly_body(P, V, n, self.R, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, touch, 2, impact, 1e9)

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX - (self.n0 - 1) * 14 + 28 * k, FLOOR - 40 - 22 * (k % 2))
            V[k] = (0.0, 0.0)

    def crushed(self, P, V, n, t, touch, impact):
        return np.nonzero((touch[:n] & 3) == 3)[0]


# ------------------------------------------------------------------ 1) Walze

class Roller(Idea):
    RR = 62.0
    AMP, T = 205.0, 1.6

    def floors(self):
        return [flat_floor()]

    def x(self, t):
        return CX + self.AMP * math.sin(2 * math.pi * t / self.T)

    def vx(self, t):
        return self.AMP * 2 * math.pi / self.T * math.cos(2 * math.pi * t / self.T)

    def cy(self):
        return FLOOR - self.RR - 12          # Spalt unter der Walze kleiner als ein Ball

    def machine(self, P, V, n, t, touch, impact):
        self.static(P, V, n, touch, impact)
        x, vx = self.x(t), self.vx(t)
        poly_body(P, V, n, self.R, disc(x, self.cy(), self.RR), vx, 0.0, x, self.cy(), vx / self.RR, 0.2, touch, 1,
                  impact, 1e9)

    def eject(self, P, V, i, j, t):
        """Plattgewalzt -> die zwei Hälften kommen hinter der Walze heraus."""
        s = -1 if self.vx(t) > 0 else 1
        V[j] = V[i]
        x = self.x(t) + s * (self.RR + self.R + 2)
        P[i] = (x, FLOOR - self.R - 0.5)
        P[j] = (x + s * 2 * self.R, FLOOR - self.R - 0.5)

    def state(self, t):
        return (self.x(t), self.x(t) / self.RR)


# ------------------------------------------------------------------ 2) Schere

class Scissor(Idea):
    PX, PY = CX, CY
    LEN, TH = R_AR - 14.0, 16.0
    OPEN, SHUT = math.radians(75), math.radians(3)
    T = 0.85

    def half(self, t):
        u = (t / self.T) % 1.0
        if u < 0.45:
            v = u / 0.45
            f = v * v * (3 - 2 * v)          # langsam auf
        elif u < 0.75:
            f = 1.0
        else:
            v = (u - 0.75) / 0.25
            f = 1 - v * v                    # schnell zu
        return self.SHUT + (self.OPEN - self.SHUT) * f

    def blade(self, a, side):
        ang = math.pi / 2 + side * a
        mx, my = self.PX + math.cos(ang) * self.LEN / 2, self.PY + math.sin(ang) * self.LEN / 2
        return rect_poly(mx, my, self.LEN, self.TH, ang)

    def floors(self):
        return []

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX - 60 + 26 * (k % 5), CY + 250 - 24 * (k // 5))
            V[k] = (0.0, 0.0)

    def machine(self, P, V, n, t, touch, impact):
        self.static(P, V, n, touch, impact)
        a = self.half(t)
        e = 1e-3
        w = (self.half(t + e) - self.half(t - e)) / (2 * e)
        poly_body(P, V, n, self.R, self.blade(a, 1), 0.0, 0.0, self.PX, self.PY, w, 0.2, touch, 1, impact, 1e9)
        poly_body(P, V, n, self.R, self.blade(a, -1), 0.0, 0.0, self.PX, self.PY, -w, 0.2, touch, 8, impact, 1e9)
        circle_body(P, V, n, self.R, CX, CY, 24.0, 0.0, 0.3, touch, 4)

    def crushed(self, P, V, n, t, touch, impact):
        e = 1e-3
        if self.half(t + e) >= self.half(t - e):          # nur beim Zuschnappen
            return np.zeros(0, np.int64)
        both = ((touch[:n] & 1) == 1) & ((touch[:n] & 8) == 8)
        wall_pinch = ((touch[:n] & 9) != 0) & self.on_wall(P, n)      # Klingenspitze drückt gegen die Wand
        return np.nonzero(both | wall_pinch)[0]

    def state(self, t):
        return (self.half(t),)


# ------------------------------------------------------------------ 3) Exzenter

class Cam(Idea):
    DR = 95.0
    ECC = 60.0
    OM = 7.5
    AX, AY = CX, FLOOR - 95.0 - 60.0 - 12.0

    def floors(self):
        return v_floor(55, rise=300)

    def center(self, t):
        a = self.OM * t
        return self.AX + math.sin(a) * self.ECC, self.AY + math.cos(a) * self.ECC

    def machine(self, P, V, n, t, touch, impact):
        self.static(P, V, n, touch, impact)
        cx, cy = self.center(t)
        poly_body(P, V, n, self.R, disc(cx, cy, self.DR), 0.0, 0.0, self.AX, self.AY, -self.OM, 0.2, touch, 1,
                  impact, 1e9)

    def state(self, t):
        return (self.OM * t,)


# ------------------------------------------------------------------ 4) Mahlwerk

class Grinder(Idea):
    ARM, ARM_W = R_AR - 14.0, 30.0
    OM = 2.4
    TEETH = [math.radians(a) for a in (30, 90, 150)]   # feste Zähne unten (Winkel im Kreis, 90 = ganz unten)
    TOOTH_L = R_AR - 300.0 - 14.0 + 6.0                 # Zahn ragt bis knapp an die Armspitzen

    def arms(self, t):
        out = []
        for q in range(3):
            a = self.OM * t + q * 2 * math.pi / 3
            mx, my = CX + math.cos(a) * self.ARM / 2, CY + math.sin(a) * self.ARM / 2
            out.append(rect_poly(mx, my, self.ARM, self.ARM_W, a))
        return out

    def teeth(self):
        out = []
        for a in self.TEETH:
            base = R_AR + 30
            tip = R_AR - (R_AR - self.ARM - 13)
            ca, sa = math.cos(a), math.sin(a)
            tx, ty = -sa, ca
            w0, w1 = 46, 16
            out.append(np.array([(CX + ca * base + tx * w0, CY + sa * base + ty * w0),
                                 (CX + ca * tip + tx * w1, CY + sa * tip + ty * w1),
                                 (CX + ca * tip - tx * w1, CY + sa * tip - ty * w1),
                                 (CX + ca * base - tx * w0, CY + sa * base - ty * w0)], np.float64))
        return out

    def floors(self):
        return []

    def crushed(self, P, V, n, t, touch, impact):
        return np.nonzero(((touch[:n] & 1) == 1) & self.on_wall(P, n))[0]   # Armspitze quetscht an die Wand

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX - 120 + 26 * k, CY + R_AR - 60 - 20 * (k % 2))
            V[k] = (0.0, 0.0)

    def machine(self, P, V, n, t, touch, impact):
        self.static(P, V, n, touch, impact)
        for q in self.arms(t):
            poly_body(P, V, n, self.R, q, 0.0, 0.0, CX, CY, self.OM, 0.2, touch, 1, impact, 1e9)
        circle_body(P, V, n, self.R, CX, CY, 30.0, 0.0, 0.3, touch, 4)

    def state(self, t):
        return (self.OM * t,)


# ------------------------------------------------------------------ 5) Rammböcke

class Rams(Idea):
    RH = 90.0
    T = 1.3
    OPEN = R_AR - 30.0

    def gap_half(self, t):
        u = (t / self.T) % 1.0
        if u < 0.4:
            v = u / 0.4
            f = v * v                         # zusammenschieben
        elif u < 0.5:
            f = 1.0
        else:
            v = (u - 0.5) / 0.5
            f = 1 - v * v * (3 - 2 * v)
        return self.OPEN + (8.0 - self.OPEN) * f

    def floors(self):
        return [flat_floor()]

    def ram(self, g, side):
        x_in = CX + side * g
        x_out = CX + side * (R_AR + 80)
        xa, xb = min(x_in, x_out), max(x_in, x_out)
        return np.array([(xa, FLOOR - self.RH), (xb, FLOOR - self.RH), (xb, FLOOR), (xa, FLOOR)], np.float64)

    def machine(self, P, V, n, t, touch, impact):
        self.static(P, V, n, touch, impact)
        g = self.gap_half(t)
        e = 1e-3
        vg = (self.gap_half(t + e) - self.gap_half(t - e)) / (2 * e)
        poly_body(P, V, n, self.R, self.ram(g, -1), -vg, 0.0, 0.0, 0.0, 0.0, 0.2, touch, 1, impact, 1e9)
        poly_body(P, V, n, self.R, self.ram(g, 1), vg, 0.0, 0.0, 0.0, 0.0, 0.2, touch, 8, impact, 1e9)

    def crushed(self, P, V, n, t, touch, impact):
        if self.gap_half(t) > 4 * self.R:
            return np.zeros(0, np.int64)
        both = ((touch[:n] & 1) == 1) & ((touch[:n] & 8) == 8)
        mid = np.abs(P[:n, 0] - CX) < self.gap_half(t) + self.R
        return np.nonzero(both | (mid & ((touch[:n] & 9) != 0)))[0]

    def eject(self, P, V, i, j, t):
        """Zwischen den Rammböcken kein Platz: die Hälften quellen nach oben heraus."""
        V[j] = V[i]
        P[i] = (CX - self.R * 0.6, FLOOR - self.RH - self.R - 1)
        P[j] = (CX + self.R * 0.6, FLOOR - self.RH - self.R - 1)

    def state(self, t):
        return (self.gap_half(t),)


# ------------------------------------------------------------------ Darstellung

class Renderer(I2.Renderer):
    def machine(self, d, st):
        sc, k = self.sc, SS
        for q in sc.floors():
            self.poly(d, q, LAV)
        if isinstance(sc, Roller):
            x = st[0]
            cy, R = sc.cy(), sc.RR
            d.ellipse([(x - R) * k, (cy - R) * k, (x + R) * k, (cy + R) * k], fill=GOLD + (255,))
            for q in range(8):
                a = st[1] + q * math.pi / 4
                d.line([((x + math.cos(a) * R * 0.3) * k, (cy + math.sin(a) * R * 0.3) * k),
                        ((x + math.cos(a) * R * 0.9) * k, (cy + math.sin(a) * R * 0.9) * k)], fill=GOLD_D + (255,),
                       width=5 * k)
            d.ellipse([(x - 16) * k, (cy - 16) * k, (x + 16) * k, (cy + 16) * k], fill=LAV_D + (255,))
        elif isinstance(sc, Scissor):
            a = st[0]
            for s in (1, -1):
                self.poly(d, sc.blade(a, s), GOLD)
            d.ellipse([(sc.PX - 24) * k, (sc.PY - 24) * k, (sc.PX + 24) * k, (sc.PY + 24) * k], fill=LAV_D + (255,))
        elif isinstance(sc, Cam):
            a = st[0]
            cx, cy = sc.AX + math.sin(a) * sc.ECC, sc.AY + math.cos(a) * sc.ECC
            R = sc.DR
            d.ellipse([(cx - R) * k, (cy - R) * k, (cx + R) * k, (cy + R) * k], fill=GOLD + (255,))
            d.ellipse([(cx - R + 10) * k, (cy - R + 10) * k, (cx + R - 10) * k, (cy + R - 10) * k],
                      outline=GOLD_HI + (255,), width=3 * k)
            d.ellipse([(sc.AX - 15) * k, (sc.AY - 15) * k, (sc.AX + 15) * k, (sc.AY + 15) * k], fill=LAV_D + (255,))
        elif isinstance(sc, Grinder):
            for q in sc.arms(st[0] / sc.OM):
                self.poly(d, q, GOLD)
            d.ellipse([(CX - 30) * k, (CY - 30) * k, (CX + 30) * k, (CY + 30) * k], fill=LAV_D + (255,))
            d.ellipse([(CX - 11) * k, (CY - 11) * k, (CX + 11) * k, (CY + 11) * k], fill=LAV_HI + (255,))
        elif isinstance(sc, Rams):
            g = st[0]
            for s in (-1, 1):
                p = sc.ram(g, s)
                self.poly(d, p, GOLD)
                xi = CX + s * g
                d.line([(xi * k, (FLOOR - sc.RH + 6) * k), (xi * k, (FLOOR - 6) * k)], fill=GOLD_HI + (255,),
                       width=5 * k)


class _Sel:
    pass


def build_audio(sc, path):
    """Weiches Plopp beim Verdoppeln + tiefes Plopp im Takt der Maschine (falls sie einen Takt hat)."""
    I2.build_audio(sc, path)


IDEAS = {"q01_steamroller": lambda: Roller(1, n0=10),
         "q02_scissor": lambda: Scissor(1, n0=14),
         "q03_cam_press": lambda: Cam(1, n0=14),
         "q04_grinder": lambda: Grinder(1, n0=10),
         "q05_side_rams": lambda: Rams(1, n0=10)}

if __name__ == "__main__":
    import sys
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(IDEAS)
    for name in names:
        sc = IDEAS[name]().run()
        print(name, [len(sc.frames[min(sc.n_frames - 1, q * FPS)][0]) for q in range(9)], flush=True)
        if "--preview" in sys.argv:
            R = Renderer(sc)
            ims = [R.frame(min(sc.n_frames - 1, int(x * FPS))) for x in (0.5, 3.0, 5.5, 7.95)]
            out = Image.new("RGB", (360 * 4, 640))
            for i, im in enumerate(ims):
                out.paste(im.resize((360, 640)), (i * 360, 0))
            out.save(f"output/preview/{name}.png")
        else:
            I2._R = Renderer(sc)
            old = I2.Renderer
            I2.Renderer = Renderer
            try:
                I2.render(sc, f"output/{name}_vertical.mp4")
            finally:
                I2.Renderer = old
            print(f"output/{name}_vertical.mp4")
