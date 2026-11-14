"""Drei 8-Sekunden-Konzepte: Bälle vermehren sich durch eine Aktion einer Maschine (ohne Überschrift).

  rollers : zwei gegenläufige Walzen unten ziehen Bälle in den Spalt; zerquetschte Bälle platzen in zwei
  blade   : eine rotierende Klinge in der Mitte schneidet Bälle – jeder geschnittene Ball wird zu zwei Hälften
  piston  : ein Stampfer schlägt im Takt von oben auf den Boden; was darunter liegt, platzt in zwei

Physik: numba, Bälle mit Schwerkraft, Kreiswand, Ball-Ball über Raster; Maschinenteile sind kinematisch.
"""
import bisect
import math
import os
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb
from .overflow import poly_push, ball_col

FPS = 60
W1, H1, SS = 1080, 1920, 2
SR = 44100
F_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
CX, CY, R_AR = 520.0, 900.0, 420.0
G = 1500.0


@njit(cache=True)
def wall(P, V, n, r, rest, kick=0.0):
    for i in range(n):
        dx, dy = P[i, 0] - CX, P[i, 1] - CY
        d = math.sqrt(dx * dx + dy * dy)
        if d > R_AR - r:
            nx, ny = dx / d, dy / d
            P[i, 0] = CX + nx * (R_AR - r)
            P[i, 1] = CY + ny * (R_AR - r)
            vn = V[i, 0] * nx + V[i, 1] * ny
            if vn > 0:
                V[i, 0] -= (1 + rest) * vn * nx
                V[i, 1] -= (1 + rest) * vn * ny
            if kick > 0:   # nach innen schubsen, damit nichts an der Wand klebt
                V[i, 0] -= nx * kick
                V[i, 1] -= ny * kick


@njit(cache=True)
def integrate(P, V, n, dt, g, vmin):
    for i in range(n):
        V[i, 1] += g * dt
        P[i, 0] += V[i, 0] * dt
        P[i, 1] += V[i, 1] * dt
        sp = math.sqrt(V[i, 0] ** 2 + V[i, 1] ** 2)
        if sp > 1800:
            V[i, 0] *= 1800 / sp
            V[i, 1] *= 1800 / sp
        elif sp < vmin and sp > 1e-6:   # Schwerelos: Tempo halten
            V[i, 0] *= vmin / sp
            V[i, 1] *= vmin / sp


@njit(cache=True)
def circle_body(P, V, n, r, bx, by, br, omega, rest, touch, bit):
    """Fester Kreis (z.B. Walze), der sich mit omega dreht: Oberfläche zieht die Bälle mit."""
    for i in range(n):
        dx, dy = P[i, 0] - bx, P[i, 1] - by
        d = math.sqrt(dx * dx + dy * dy)
        if d < br + r and d > 1e-9:
            nx, ny = dx / d, dy / d
            P[i, 0] = bx + nx * (br + r)
            P[i, 1] = by + ny * (br + r)
            sx, sy = -omega * ny * br, omega * nx * br
            rvx, rvy = V[i, 0] - sx, V[i, 1] - sy
            vn = rvx * nx + rvy * ny
            if vn < 0:
                rvx -= (1 + rest) * vn * nx
                rvy -= (1 + rest) * vn * ny
            # Reibung: tangential an die Oberfläche angleichen
            tx, ty = -ny, nx
            vt = rvx * tx + rvy * ty
            rvx -= tx * vt * 0.25
            rvy -= ty * vt * 0.25
            V[i, 0], V[i, 1] = rvx + sx, rvy + sy
            touch[i] |= bit


@njit(cache=True)
def poly_body(P, V, n, r, poly, vx, vy, piv_x, piv_y, omega, rest, touch, bit, impact, imp_thr):
    """Konvexes Polygon mit Geschwindigkeit (vx, vy) und Drehung omega um (piv_x, piv_y)."""
    for i in range(n):
        pen, nx, ny, kx, ky = poly_push(poly, P[i, 0], P[i, 1], r)
        if pen > 0:
            P[i, 0] += nx * pen
            P[i, 1] += ny * pen
            sx = vx - omega * (ky - piv_y)
            sy = vy + omega * (kx - piv_x)
            rvx, rvy = V[i, 0] - sx, V[i, 1] - sy
            vn = rvx * nx + rvy * ny
            if vn < 0:
                rvx -= (1 + rest) * vn * nx
                rvy -= (1 + rest) * vn * ny
                V[i, 0], V[i, 1] = rvx + sx, rvy + sy
                if -vn > imp_thr:
                    impact[i] = 1
            touch[i] |= bit


@njit(cache=True)
def collide(P, V, n, r, cell, head, nxt, gw, gh, x0, y0):
    head[:] = -1
    for i in range(n):
        cx = int((P[i, 0] - x0) / cell)
        cy = int((P[i, 1] - y0) / cell)
        if 0 <= cx < gw and 0 <= cy < gh:
            nxt[i] = head[cy * gw + cx]
            head[cy * gw + cx] = i
        else:
            nxt[i] = -1
    D = 2 * r
    for i in range(n):
        cx = int((P[i, 0] - x0) / cell)
        cy = int((P[i, 1] - y0) / cell)
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                gx, gy = cx + ox, cy + oy
                if gx < 0 or gy < 0 or gx >= gw or gy >= gh:
                    continue
                j = head[gy * gw + gx]
                while j != -1:
                    if j > i:
                        dx, dy = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
                        d2 = dx * dx + dy * dy
                        if 1e-9 < d2 < D * D:
                            d = math.sqrt(d2)
                            nx, ny = dx / d, dy / d
                            ov = (D - d) * 0.5
                            P[i, 0] -= nx * ov
                            P[i, 1] -= ny * ov
                            P[j, 0] += nx * ov
                            P[j, 1] += ny * ov
                            rv = (V[j, 0] - V[i, 0]) * nx + (V[j, 1] - V[i, 1]) * ny
                            if rv < 0:
                                imp = -rv * 0.9 * 0.5
                                V[i, 0] -= imp * nx
                                V[i, 1] -= imp * ny
                                V[j, 0] += imp * nx
                                V[j, 1] += imp * ny
                    j = nxt[j]


def rect_poly(cx, cy, w, h, ang=0.0):
    c, s = math.cos(ang), math.sin(ang)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return np.array([(cx + x * c - y * s, cy + x * s + y * c) for x, y in pts], np.float64)


# ------------------------------------------------------------------ Szenarien

class Scene:
    """Basis: Bälle + Maschine. Unterklassen setzen machine(t) und crushed()."""
    R = 9.0
    CAP = 3000
    DUR = 8.0
    SUB = 8
    GAP = 0.3
    GRAV = G
    VMIN = 0.0
    WALL_REST = 0.6
    KICK = 0.0

    def __init__(self, seed, n0=3):
        self.rng = np.random.default_rng(seed)
        self.n0 = n0

    def allow(self, n, t):
        return True

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX + (k - (self.n0 - 1) / 2) * 60, CY - 250 + self.rng.uniform(-20, 20))
            V[k] = (self.rng.uniform(-150, 150), 0)

    def run(self):
        rng = self.rng
        N, r = self.CAP, self.R
        P, V = np.zeros((N, 2)), np.zeros((N, 2))
        last = np.full(N, -9.0)
        self.start(P, V)
        n = self.n0
        cell = 2 * r
        x0, y0 = CX - R_AR - 10, CY - R_AR - 10
        gw = gh = int((2 * R_AR + 20) / cell) + 1
        head = np.full(gw * gh, -1, np.int64)
        nxt = np.full(N, -1, np.int64)
        touch = np.zeros(N, np.int64)
        impact = np.zeros(N, np.int64)
        dt = 1.0 / (FPS * self.SUB)
        t = 0.0
        self.frames, self.events = [], []
        for fi in range(int(self.DUR * FPS)):
            for _ in range(self.SUB):
                touch[:n] = 0
                impact[:n] = 0
                integrate(P, V, n, dt, self.GRAV, self.VMIN)
                wall(P, V, n, r, self.WALL_REST, self.KICK)
                self.machine(P, V, n, t, touch, impact)
                collide(P, V, n, r, cell, head, nxt, gw, gh, x0, y0)
                for i in self.crushed(P, V, n, t, touch, impact):
                    if n >= N or t - last[i] < self.GAP or not self.allow(n, t):
                        continue
                    last[i] = last[n] = t
                    P[n] = P[i] + rng.normal(0, 1.0, 2)
                    self.eject(P, V, i, n, t)
                    self.events.append((t, "split", {"n": n + 1, "x": float(P[i, 0]), "y": float(P[i, 1])}))
                    n += 1
                t += dt
            self.frames.append((P[:n].astype(np.float32).copy(), self.state(t)))
        self.n_frames = len(self.frames)
        return self


class Rollers(Scene):
    """Zwei gegenläufige Walzen ziehen Bälle in den Spalt. Wer zwischen beiden steckt, platzt in zwei."""
    GAP = 0.12
    RR = 105.0
    GAP_W = 10.0
    W = 6.5

    def __init__(self, seed, n0=3):
        super().__init__(seed, n0)
        self.L = (CX - self.RR - self.GAP_W / 2, CY + 150)
        self.Rr = (CX + self.RR + self.GAP_W / 2, CY + 150)
        self.trough = [np.array([(CX - 210, CY + 150), (CX - R_AR - 20, CY + 30), (CX - R_AR - 20, CY + 500),
                                 (CX - 210, CY + 500)]),
                       np.array([(CX + 210, CY + 150), (CX + 210, CY + 500), (CX + R_AR + 20, CY + 500),
                                 (CX + R_AR + 20, CY + 30)]),
                       np.array([(CX - 30, CY + 205), (CX + 30, CY + 205), (CX + 30, CY + 500),
                                 (CX - 30, CY + 500)])]   # Stopfen unter dem Spalt

    def machine(self, P, V, n, t, touch, impact):
        w = self.W
        circle_body(P, V, n, self.R, self.L[0], self.L[1], self.RR, w, 0.3, touch, 1)
        circle_body(P, V, n, self.R, self.Rr[0], self.Rr[1], self.RR, -w, 0.3, touch, 2)
        for q in self.trough:
            poly_body(P, V, n, self.R, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, touch, 4, impact, 1e9)

    def crushed(self, P, V, n, t, touch, impact):
        return np.nonzero(touch[:n] & 3 == 3)[0]

    def eject(self, P, V, i, j, t):
        rng = self.rng
        for k, s in ((i, -1), (j, 1)):
            a = -math.pi / 2 + s * rng.uniform(0.55, 1.05)
            sp = rng.uniform(620, 860)
            V[k] = (math.cos(a) * sp, math.sin(a) * sp)
            P[k, 1] = self.L[1] - self.RR * 0.55
            P[k, 0] = CX + s * 4

    def state(self, t):
        return (t * self.W,)


class Blade(Scene):
    """Rotierende Klinge durch die Mitte. Ein Schnitt (harter Treffer der Kante) teilt den Ball in zwei Hälften."""
    LEN, TH = 2 * (R_AR - 24), 14.0
    KICK = 260.0
    OM = 3.6
    GAP = 0.09
    GRAV = 0.0
    VMIN = 420.0
    WALL_REST = 1.0

    def blade_poly(self, t):
        return rect_poly(CX, CY, self.LEN, self.TH, self.OM * t)

    def machine(self, P, V, n, t, touch, impact):
        poly_body(P, V, n, self.R, self.blade_poly(t), 0.0, 0.0, CX, CY, self.OM, 0.6, touch, 1, impact,
                  120.0)
        hub = (CX, CY)
        circle_body(P, V, n, self.R, hub[0], hub[1], 22.0, 0.0, 0.5, touch, 2)

    def crushed(self, P, V, n, t, touch, impact):
        return np.nonzero((touch[:n] & 1) == 1)[0]

    def eject(self, P, V, i, j, t):
        # die beiden Hälften fliegen senkrecht zur Klinge auseinander
        a = self.OM * t
        nx, ny = -math.sin(a), math.cos(a)
        vx, vy = V[i]
        for k, s in ((i, 1), (j, -1)):
            V[k] = (vx * 0.6 + s * nx * 380, vy * 0.6 + s * ny * 380)
            P[k] = P[k] + np.array((s * nx * 5, s * ny * 5))

    def state(self, t):
        return (self.OM * t,)


class Piston(Scene):
    """Stampfer schlägt im Takt auf den Boden. Was darunter eingeklemmt wird, platzt in zwei."""
    PW = 380.0
    FLAT = 240.0
    FLOOR = CY + 300
    BEAT = 0.7
    GAP = 0.25

    def __init__(self, seed, n0=3):
        super().__init__(seed, n0)
        self.floor = np.array([(CX - 500, self.FLOOR), (CX + 500, self.FLOOR), (CX + 500, self.FLOOR + 300),
                               (CX - 500, self.FLOOR + 300)], np.float64)
        h = self.FLAT / 2
        self.slopes = [np.array([(CX - h, self.FLOOR), (CX - R_AR - 30, self.FLOOR - 230),
                                 (CX - R_AR - 30, self.FLOOR + 50), (CX - h, self.FLOOR + 50)], np.float64),
                       np.array([(CX + h, self.FLOOR), (CX + h, self.FLOOR + 50), (CX + R_AR + 30, self.FLOOR + 50),
                                 (CX + R_AR + 30, self.FLOOR - 230)], np.float64)]

    def piston_y(self, t):
        """Unterkante des Stampfers: oben warten, schnell runter, kurz halten, langsam hoch."""
        u = self.phase(t) % 1.0
        if u is None:
            u = 0.0
        top, low = CY - 260, self.FLOOR - 2 * self.R * 0.55
        if u < 0.45:
            f = 0.0
        elif u < 0.6:
            f = ((u - 0.45) / 0.15) ** 2
        elif u < 0.7:
            f = 1.0
        else:
            f = 1 - ((u - 0.7) / 0.3)
            f = f * f * (3 - 2 * f)
        return top + (low - top) * f

    def phase(self, t):
        return t / self.BEAT

    def piston_vy(self, t):
        e = 1e-3
        return (self.piston_y(t + e) - self.piston_y(t - e)) / (2 * e)

    def machine(self, P, V, n, t, touch, impact):
        y = self.piston_y(t)
        poly = np.array([(CX - self.PW / 2, y - 700), (CX + self.PW / 2, y - 700), (CX + self.PW / 2, y),
                         (CX - self.PW / 2, y)], np.float64)
        poly_body(P, V, n, self.R, poly, 0.0, self.piston_vy(t), 0.0, 0.0, 0.0, 0.3, touch, 1, impact, 1e9)
        poly_body(P, V, n, self.R, self.floor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, touch, 2, impact, 1e9)
        for q in self.slopes:
            poly_body(P, V, n, self.R, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, touch, 4, impact, 1e9)

    def crushed(self, P, V, n, t, touch, impact):
        if self.piston_y(t) < self.FLOOR - 6 * self.R:
            return np.zeros(0, np.int64)
        m = ((touch[:n] & 1) == 1) & ((touch[:n] & 6) != 0)
        return np.nonzero(m)[0]

    def eject(self, P, V, i, j, t):
        rng = self.rng
        for k in (i, j):
            s = 1 if P[k, 0] > CX else -1
            if k == j:
                s = -s if rng.random() < 0.5 else s
            a = -math.pi / 2 - s * rng.uniform(0.05, 0.3)   # steil hoch und zurück zur Mitte
            sp = rng.uniform(820, 1050)
            V[k] = (math.cos(a) * sp, math.sin(a) * sp)
            P[k, 0] = CX + s * (self.PW / 2 + self.R + 2)
            P[k, 1] = min(P[k, 1], self.FLOOR - self.R - 1)

    def state(self, t):
        return (self.piston_y(t),)

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX - 60 + 60 * k, self.FLOOR - 200 - 40 * k)
            V[k] = (self.rng.uniform(-60, 60), 0)


# ------------------------------------------------------------------ Darstellung (ohne Überschrift)

def _font(path, size, cache={}):
    k = (path, size)
    if k not in cache:
        cache[k] = ImageFont.truetype(path, size)
    return cache[k]


CORAL, CORAL_HI, GOLD = (255, 120, 110), (255, 210, 190), (255, 225, 110)


class MultiplyRenderer:
    def __init__(self, sc):
        self.sc = sc
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        f = (yy / H1)[..., None]
        base = np.array((16, 12, 34), np.float32) * (1 - f) + np.array((34, 12, 44), np.float32) * f
        base = base + np.exp(-(((xx - CX) / 520) ** 2 + ((yy - CY) / 520) ** 2))[..., None] * np.array((30, 20, 60))
        inside = (xx - CX) ** 2 + (yy - CY) ** 2 < R_AR ** 2
        base[inside] = base[inside] * 0.55 + np.array((4, 3, 10))
        self.base = Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(self.base)
        R0, R1 = R_AR * k, (R_AR + 26) * k
        for q in range(72):
            d.arc([CX * k - (R0 + R1) / 2, CY * k - (R0 + R1) / 2, CX * k + (R0 + R1) / 2, CY * k + (R0 + R1) / 2],
                  q * 5 + 0.6, q * 5 + 4.4, fill=(60, 230, 200) if q % 2 else (40, 180, 170), width=int(R1 - R0))
        m = Image.new("L", self.base.size, 0)
        ImageDraw.Draw(m).ellipse([(CX - R_AR) * k, (CY - R_AR) * k, (CX + R_AR) * k, (CY + R_AR) * k], fill=255)
        self.mask = m
        self.splits = [e for e in sc.events if e[1] == "split"]
        self.split_t = [e[0] for e in self.splits]
        self.miles = []
        if isinstance(sc, PistonLong):
            for m in (100, 500, 1000, 2000, 3000):
                hit = next((e[0] for e in self.splits if e[2]["n"] >= m), None)
                if hit is not None:
                    self.miles.append((hit, m))

    def poly(self, d, pts, col, hi=None):
        k = SS
        d.polygon([(x * k, y * k) for x, y in pts], fill=col + (255,))
        if hi is not None:
            d.line([(pts[0][0] * k, pts[0][1] * k), (pts[1][0] * k, pts[1][1] * k)], fill=hi + (255,), width=4 * k)

    def machine(self, d, t, st):
        sc, k = self.sc, SS
        if isinstance(sc, Rollers):
            for q in sc.trough[:2]:
                self.poly(d, q, CORAL, CORAL_HI)
            self.poly(d, sc.trough[2], CORAL)
            for (x, y), sgn in ((sc.L, 1), (sc.Rr, -1)):
                R = sc.RR
                d.ellipse([(x - R) * k, (y - R) * k, (x + R) * k, (y + R) * k], fill=GOLD + (255,))
                for q in range(12):   # Rillen drehen mit
                    a = st[0] * sgn + q * math.pi / 6
                    d.line([((x + math.cos(a) * R * 0.35) * k, (y + math.sin(a) * R * 0.35) * k),
                            ((x + math.cos(a) * R * 0.95) * k, (y + math.sin(a) * R * 0.95) * k)],
                           fill=(205, 160, 40, 255), width=5 * k)
                d.ellipse([(x - R * 0.28) * k, (y - R * 0.28) * k, (x + R * 0.28) * k, (y + R * 0.28) * k],
                          fill=CORAL + (255,))
                d.ellipse([(x - 10) * k, (y - 10) * k, (x + 10) * k, (y + 10) * k], outline=(120, 40, 60, 255),
                          width=3 * k)
        elif isinstance(sc, Blade):
            pts = sc.blade_poly(t)
            self.poly(d, pts, GOLD)
            d.line([(pts[0][0] * k, pts[0][1] * k), (pts[1][0] * k, pts[1][1] * k)], fill=(255, 255, 230, 255),
                   width=3 * k)
            d.ellipse([(CX - 26) * k, (CY - 26) * k, (CX + 26) * k, (CY + 26) * k], fill=CORAL + (255,))
            d.ellipse([(CX - 9) * k, (CY - 9) * k, (CX + 9) * k, (CY + 9) * k], outline=(120, 40, 60, 255),
                      width=3 * k)
        else:
            self.poly(d, sc.floor, CORAL)
            for q in sc.slopes:
                self.poly(d, q, CORAL, CORAL_HI)
            y = st[0]
            x0, x1 = CX - sc.PW / 2, CX + sc.PW / 2
            d.rectangle([x0 * k, (y - 700) * k, x1 * k, y * k], fill=GOLD + (255,))
            xa = x0 + 6
            while xa + 47 < x1 - 4:   # Warnstreifen unten am Stampfer
                d.polygon([(xa * k, (y - 38) * k), ((xa + 17) * k, (y - 38) * k), ((xa + 47) * k, y * k),
                           ((xa + 30) * k, y * k)], fill=(40, 30, 50, 255))
                xa += 34
            d.line([x0 * k, y * k, x1 * k, y * k], fill=(255, 250, 210, 255), width=4 * k)

    def frame(self, fi):
        sc, k = self.sc, SS
        t = (fi + 1) / FPS
        Pn, st = sc.frames[fi]
        n = len(Pn)
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        r = sc.R * k
        for i in range(n):
            x, y = Pn[i, 0] * k, Pn[i, 1] * k
            c = ball_col(i * 3)
            d.ellipse([x - r, y - r, x + r, y + r], fill=c + (255,))
            d.ellipse([x - r * 0.55, y - r * 0.6, x - r * 0.05, y - r * 0.15], fill=(255, 255, 255, 150))
        lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.machine(ImageDraw.Draw(lay, "RGBA"), t, st)
        im.paste(lay, (0, 0), Image.composite(lay.getchannel("A"), Image.new("L", lay.size, 0), self.mask))
        d = ImageDraw.Draw(im, "RGBA")
        lo = bisect.bisect_left(self.split_t, t - 0.3)
        hi = bisect.bisect_right(self.split_t, t)
        for t0, _, dd in self.splits[lo:hi][-80:]:
            e = (t - t0) / 0.3
            x, y = dd["x"] * k, dd["y"] * k
            R = (6 + 24 * e) * k
            d.ellipse([x - R, y - R, x + R, y + R], outline=(255, 255, 255, int(220 * (1 - e))), width=2 * k)
            if e < 0.4:
                R2 = 10 * k * (1 - e / 0.4)
                d.ellipse([x - R2, y - R2, x + R2, y + R2], fill=(255, 255, 255, int(230 * (1 - e / 0.4))))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(g, np.float32) - 50, 0, 255) * 0.35
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(out, "RGBA")
        last = [e[0] for e in self.splits if e[0] <= t]
        pop = 1 + 0.12 * max(0.0, 1 - (t - last[-1]) / 0.12) if last else 1.0
        col = (255, 255, 255)
        ms = [m for m in getattr(self, "miles", []) if m[0] <= t]
        if ms:
            e = t - ms[-1][0]
            if e < 0.6:   # Meilenstein: Zähler wird groß und golden, kurzer Lichtblitz im Ring
                pop += 0.55 * (1 - e / 0.6) ** 2
                col = GOLD
                fl = Image.new("RGBA", out.size, (0, 0, 0, 0))
                ImageDraw.Draw(fl).ellipse([CX - R_AR, CY - R_AR, CX + R_AR, CY + R_AR], outline=GOLD + (255,),
                                           width=18)
                fl = fl.filter(ImageFilter.GaussianBlur(14))
                a_ = np.asarray(fl, np.float32)
                arr = np.asarray(out, np.float32) + a_[..., :3] * (a_[..., 3:] / 255) * (1 - e / 0.6) * 1.4
                out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                d = ImageDraw.Draw(out, "RGBA")
        if getattr(sc, "STOP", None) and t > sc.STOP + 1.2:
            e = min(1.0, (t - sc.STOP - 1.2) / 0.5)
            col = GOLD
            pop = 1 + 0.35 * e
        d.text((CX, CY + R_AR + 74), f"{n:,}", font=_font(F_BOLD, int(54 * pop)), fill=col + (240,),
               anchor="mm")
        return out


def build_audio(sc, path):
    from .breaker import plop
    dur = sc.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + SR * 2)
    last = -1.0
    for t, kind, d in sc.events:
        if t - last < 0.05:
            continue
        last = t
        f = 420 + 380 * min(1.0, math.log10(max(1, d["n"])) / 3.0)
        s = plop(round(f / 20) * 20)
        i = int(t * SR)
        e = min(len(out), i + len(s))
        out[i:e] += s[: e - i] * 0.45
    if isinstance(sc, PistonLong):
        sp = [e for e in sc.events if e[1] == "split"]
        for m in (100, 500, 1000, 2000, 3000):
            hit = next((e[0] for e in sp if e[2]["n"] >= m), None)
            if hit is not None:
                for j, f in enumerate((300, 400, 520)):
                    s_ = plop(f, big=True)
                    i = int((hit + j * 0.08) * SR)
                    out[i:i + len(s_)] += s_[: max(0, len(out) - i)][: len(out[i:i + len(s_)])] * 0.7
        # Stampfer-Aufschlag: tiefes, weiches Plopp bei jedem Schlag
        tt = 0.0
        while tt < sc.STOP + 1:
            ph = sc.phase(tt)
            if ph % 1.0 > 0.58 and sc.phase(tt - 1 / FPS) % 1.0 <= 0.58:
                s_ = plop(160, big=True)
                i = int(tt * SR)
                out[i:i + len(s_)] += s_[: max(0, len(out) - i)][: len(out[i:i + len(s_)])] * 0.35
            tt += 1 / FPS
    out = out[:n]
    mix_ = out + reverb(out, 0.6) * 0.08
    peak = np.max(np.abs(mix_)) or 1
    mix_ = mix_ / peak * 0.6
    st = np.stack([mix_, mix_], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


_R = None


def _chunk(args):
    i0, i1, out = args
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W1}x{H1}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "14M", "-bufsize", "28M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_R.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(sc, out_path, jobs=4):
    global _R
    _R = MultiplyRenderer(sc)
    n = sc.n_frames
    with tempfile.TemporaryDirectory() as tmp:
        bounds = [n * j // jobs for j in range(jobs + 1)]
        parts = [(bounds[j], bounds[j + 1], os.path.join(tmp, f"p{j}.mp4")) for j in range(jobs)]
        with Pool(jobs) as p:
            p.map(_chunk, parts)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            for _, _, pth in parts:
                f.write(f"file '{pth}'\n")
        wav = os.path.join(tmp, "a.wav")
        build_audio(sc, wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-17:TP=-2:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


class PistonLong(Piston):
    """Lange TikTok-Version: Takt wird immer schneller, Meilensteine, am Ende bleibt der Stampfer oben."""
    CAP = 4200
    DUR = 47.0
    B0, B1, TB = 1.0, 0.42, 34.0     # Takt von 1,0 s auf 0,42 s bis Sekunde 34
    STOP = 43.0

    def phase(self, t):
        k = (self.B1 - self.B0) / self.TB
        if t <= self.TB:
            return math.log((self.B0 + k * t) / self.B0) / k
        return math.log(self.B1 / self.B0) / k + (t - self.TB) / self.B1

    def piston_y(self, t):
        if t >= self.STOP:   # Ausklang: erst die laufende Bewegung beenden, dann oben bleiben
            ph = self.phase(self.STOP)
            t_end = self.STOP + (math.ceil(ph) - ph) * self.B1
            if t >= t_end:
                return CY - 260
        return Piston.piston_y(self, t)

    N_END, T_END = 3000, 42.0

    def allow(self, n, t):
        """Wachstum folgt einer glatten Exponentialkurve (zerdrückt wird trotzdem nur echt Eingeklemmtes)."""
        return n < 3 + (self.N_END - 3) * (min(t, self.T_END) / self.T_END) ** 2.4 + 1

    def crushed(self, P, V, n, t, touch, impact):
        if t >= self.STOP + 1.0:
            return np.zeros(0, np.int64)
        return Piston.crushed(self, P, V, n, t, touch, impact)


SCENES = {"m01_crusher_rollers": lambda: Rollers(1, n0=3),
          "m02_spinning_blade": lambda: Blade(2, n0=3),
          "m03_piston_press": lambda: Piston(3, n0=5),
          "m04_piston_press_long": lambda: PistonLong(SEED_LONG, n0=3)}
SEED_LONG = 1

if __name__ == "__main__":
    import sys
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SCENES)
    for name in names:
        sc = SCENES[name]().run()
        print(name, [len(sc.frames[min(sc.n_frames - 1, k * FPS)][0]) for k in range(9)])
        if "--preview" in sys.argv:
            R = MultiplyRenderer(sc)
            ims = [R.frame(min(sc.n_frames - 1, int(x * FPS))) for x in (0.5, 3.0, 5.5, 7.95)]
            out = Image.new("RGB", (360 * 4, 640))
            for i, im in enumerate(ims):
                out.paste(im.resize((360, 640)), (i * 360, 0))
            out.save(f"output/preview/{name}.png")
            tiktok_ui_mock(ims[-1], "vertical").save(f"output/preview/{name}_ui.png")
        else:
            render(sc, f"output/{name}_vertical.mp4")
            print(f"output/{name}_vertical.mp4")
