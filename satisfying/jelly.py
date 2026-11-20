"""Jelly Crush – drei neue Maschinen (je 8 s Probe), gleiche Regeln wie Arc Squeeze:
  * neue Bälle entstehen NUR durch Zerquetschen: genau an der Stelle, wo ein Ball zerquetscht wurde,
    platzt er in 3 Jellys (nächste Farbe), die seitlich herausgedrückt werden
  * "zerquetscht" = die Maschine drückt den Ball gegen Wand/Boden/andere Bälle und er kann nicht ausweichen
    (Restüberlappung mit der Maschine nach dem Auflösen aller Kollisionen)
  * Bälle bewegen sich nur durch Physik/Maschine, keine Schrift außer der Ballzahl
  * Jelly-Optik: glänzend, durchscheinend, wabbelt bei jedem Stoß (gedämpfte Feder pro Ball)
  * Sound: jede Quetschung = "HUSS!" (Sample des Nutzers), Aufprall = leises Jelly-"Blop", Maschine = weiches Brummen

  drehkreuz – Drehkreuz in der Mitte (4 Arme, im Uhrzeigersinn), quetscht Jellys an den Wänden
  presse    – Hydraulikpresse von oben, Schwerkraft: der Jelly-Haufen wird plattgedrückt
  walze     – schwere Walze rollt hin und her über den Boden, Schwerkraft: Jellys darunter werden zerquetscht
"""
import math
import os
import sys
import wave

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFilter

from . import arcbox as A
from .arcbox import CX, CY, HALF, X0, X1, Y0, Y1, FPS, W1, H1, SS
from .multiply import collide as _collide, _font, F_BOLD

SR = 44100


@njit(cache=True)
def capsule(P, V, n, r, ax, ay, bx, by, rad, ox, oy, om, lx, ly, e, pen, nx_, ny_, rec):
    """Ball gegen bewegte Kapsel (Strecke a-b mit Radius rad). Starrkörper-Bewegung: Translation (lx,ly)
    plus Drehung om um (ox,oy). rec=1: Eindringtiefe merken (für 'zerquetscht')."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    D = r + rad
    for i in range(n):
        px, py = P[i, 0], P[i, 1]
        u = 0.0
        if L2 > 1e-9:
            u = ((px - ax) * dx + (py - ay) * dy) / L2
            u = min(1.0, max(0.0, u))
        cx, cy = ax + dx * u, ay + dy * u
        qx, qy = px - cx, py - cy
        d = math.sqrt(qx * qx + qy * qy)
        if d < D and d > 1e-9:
            nx, ny = qx / d, qy / d
            p = D - d
            if rec == 1 and p > pen[i]:
                pen[i] = p
                nx_[i] = nx
                ny_[i] = ny
            P[i, 0] += nx * p
            P[i, 1] += ny * p
            sx = lx - om * (cy - oy)
            sy = ly + om * (cx - ox)
            rvx, rvy = V[i, 0] - sx, V[i, 1] - sy
            vn = rvx * nx + rvy * ny
            if vn < 0:
                rvx -= (1 + e) * vn * nx
                rvy -= (1 + e) * vn * ny
                V[i, 0], V[i, 1] = rvx + sx, rvy + sy


class JellySim:
    R = 17.0
    SUB = 6
    DUR = 8.0
    GRAV = 0.0
    SPEED = None              # konstantes Tempo (ohne Schwerkraft)
    E_WALL = 0.5
    DAMP = 0.0
    GAP = 0.22
    N8 = 260
    CAP = 1400
    CRUSH = 0.28              # Restüberlappung (in r), ab der ein Ball als zerquetscht gilt

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)

    def parts(self, t):
        """Liste der Kapseln: (ax, ay, bx, by, rad, ox, oy, om, lx, ly)"""
        return []

    def allow(self, n, t):
        return n + 2 < 3 + self.N8 * (min(t, self.DUR) / self.DUR) ** 1.5 + 3

    def starts(self):
        raise NotImplementedError

    def run(self):
        N, r = self.CAP, self.R
        P, V = np.zeros((N, 2)), np.zeros((N, 2))
        gen = np.zeros(N, np.int64)
        last = np.full(N, -9.0)
        born = np.full(N, -9.0)
        SP, SV = self.starts()
        n = len(SP)
        P[:n], V[:n] = SP, SV
        cell = 2 * r
        x0, y0 = X0 - 10, Y0 - 10
        gw = gh = int((2 * HALF + 20) / cell) + 1
        head = np.full(gw * gh, -1, np.int64)
        nxt = np.full(N, -1, np.int64)
        pen = np.zeros(N)
        nxa, nya = np.zeros(N), np.zeros(N)
        pen0 = np.zeros(N)
        nx0, ny0 = np.zeros(N), np.zeros(N)
        dt = 1.0 / (FPS * self.SUB)
        self.frames, self.events, self.bounces, self.parts_log, self.mspeed = [], [], [], [], []
        t = 0.0
        while t < self.DUR - 0.5 / FPS:
            for _ in range(self.SUB):
                pts = self.parts(t)
                if self.GRAV:
                    V[:n, 1] += self.GRAV * dt
                    V[:n, 0] += self.rng.standard_normal(n) * 6.0      # winziges Zittern: keine Säulen
                if self.DAMP:
                    V[:n] *= 1 - self.DAMP * dt
                P[:n] += V[:n] * dt
                pen0[:n] = 0
                for q in pts:
                    capsule(P, V, n, r, *q, 0.25, pen0, nx0, ny0, 1)
                for _ in range(2):
                    _collide(P, V, n, r, cell, head, nxt, gw, gh, x0, y0)
                    self.walls(P, V, n, r, t)
                pen[:n] = 0
                for q in pts:                       # nach dem Auflösen: steckt der Ball noch in der Maschine?
                    capsule(P, V, n, r, *q, 0.25, pen, nxa, nya, 1)
                self.walls(P, V, n, r, t)
                if self.SPEED:
                    sp = np.hypot(V[:n, 0], V[:n, 1]) + 1e-9
                    V[:n] *= (self.SPEED / sp)[:, None]
                else:
                    sp = np.hypot(V[:n, 0], V[:n, 1]) + 1e-9
                    V[:n] *= np.minimum(1, 1400 / sp)[:, None]
                cand = self.crushed(P, n, r, pen, pen0, nx0, ny0)
                for i in cand:
                    if n + 2 >= N or t - last[i] < self.GAP or not self.allow(n, t):
                        continue
                    if pen[i] <= 0:
                        nxa[i], nya[i] = nx0[i], ny0[i]
                    tx, ty = -nya[i], nxa[i]                 # Tangente an der Maschine
                    base = P[i].copy()
                    vb = V[i].copy()
                    g = gen[i] + 1
                    for j, o in enumerate((0.0, 1.0, -1.0)):
                        idx = i if j == 0 else n + j - 1
                        P[idx] = base + np.array((tx, ty)) * o * r * 0.95
                        V[idx] = vb * 0.5 + np.array((tx, ty)) * o * 260 + np.array((nxa[i], nya[i])) * 60
                        gen[idx] = g
                        last[idx] = born[idx] = t
                    self.events.append((t, "split", {"n": n + 2, "x": float(base[0]), "y": float(base[1])}))
                    n += 2
                t += dt
            self.frames.append((P[:n].astype(np.float32).copy(), gen[:n].astype(np.int16).copy(),
                                born[:n].astype(np.float32).copy()))
            self.parts_log.append(self.parts(t))
            self.mspeed.append(self.machine_speed(t))
        self.n_frames = len(self.frames)
        self._jiggle()
        return self

    def machine_speed(self, t):
        return 1.0

    def crushed(self, P, n, r, pen, pen0, nx0, ny0):
        c = np.nonzero(pen[:n] > self.CRUSH * r)[0]
        return c[np.argsort(-pen[c])]

    def walls(self, P, V, n, r, t):
        e = self.E_WALL
        lo, hi = P[:n] < np.array((X0 + r, Y0 + r)), P[:n] > np.array((X1 - r, Y1 - r))
        imp = (lo & (V[:n] < 0)) | (hi & (V[:n] > 0))
        if np.any(imp):
            vv = np.abs(V[:n][imp])
            strong = vv > 140
            if np.any(strong):
                self.bounces.append((t, float(vv[strong].max()), int(strong.sum())))
        P[:n] = np.clip(P[:n], (X0 + r, Y0 + r), (X1 - r, Y1 - r))
        V[:n] = np.where(imp, -V[:n] * e, V[:n])

    def _jiggle(self):
        """Wabbel-Feder pro Ball und Achse, angetrieben von Geschwindigkeitsänderungen (Stöße)."""
        w, z, kap = 2 * math.pi * 6.5, 0.16, 0.0125
        dt = 1 / FPS
        s = np.zeros((0, 2))
        sv = np.zeros((0, 2))
        pv = np.zeros((0, 2))
        prevP = None
        self.jig = []
        rng = np.random.default_rng(4)
        for f, (Pf, gf, bf) in enumerate(self.frames):
            n = len(Pf)
            m = len(s)
            if n > m:
                s = np.vstack([s, np.zeros((n - m, 2))])
                sv = np.vstack([sv, rng.uniform(-6, 6, (n - m, 2))])   # Neugeborene wabbeln
                pv = np.vstack([pv, np.zeros((n - m, 2))])
            if prevP is not None:
                mm = len(prevP)
                v = np.zeros((n, 2))
                v[:mm] = (Pf[:mm] - prevP) / dt
                dv = np.abs(v - pv)
                dv[mm:] = 0
                drive = np.minimum(dv, 900) * kap
                sv += drive
                pv = v
            sv += (-w * w * s - 2 * z * w * sv) * dt
            s += sv * dt
            s = np.clip(s, -0.3, 0.3)
            self.jig.append(s.astype(np.float32).copy())
            prevP = Pf


# ------------------------------------------------------------------ Maschinen
class Drehkreuz(JellySim):
    SPEED = 380.0
    OM = 2 * math.pi / 4.2        # im Uhrzeigersinn
    E_WALL = 1.0
    ARM_W = 15.0
    N8 = 300

    RW = HALF                  # runde Arena: Abstand Arm-Spitze -> Rand überall gleich

    def walls(self, P, V, n, r, t):
        q = P[:n] - (CX, CY)
        d = np.hypot(q[:, 0], q[:, 1]) + 1e-9
        out = d > self.RW - r
        if np.any(out):
            nrm = q[out] / d[out][:, None]
            vn = (V[:n][out] * nrm).sum(1)
            hitv = vn > 0
            if np.any(vn > 140):
                self.bounces.append((t, float(vn.max()), int((vn > 140).sum())))
            P[:n][out] = (CX, CY) + nrm * (self.RW - r)
            vv = V[:n][out]
            vv -= (np.where(hitv, vn, 0) * 2)[:, None] * nrm
            V[:n][out] = vv

    def crushed(self, P, n, r, pen, pen0, nx0, ny0):
        q = P[:n] - (CX, CY)
        d = np.hypot(q[:, 0], q[:, 1]) + 1e-9
        outw = (nx0[:n] * q[:, 0] + ny0[:n] * q[:, 1]) / d
        return np.nonzero((pen0[:n] > 0.2) & (d >= self.RW - r - 1.5) & (outw > 0.15))[0]

    BEND = 0.55                # Schaufeln nach hinten gebogen -> drücken Jellys nach außen gegen den Rand

    def parts(self, t):
        L = self.RW - self.R * 1.25 - self.ARM_W
        out = []
        ph = self.OM * t
        for k in range(4):
            a = ph + k * math.pi / 2
            mx, my = CX + math.cos(a) * L * 0.5, CY + math.sin(a) * L * 0.5
            tx, ty = CX + math.cos(a - self.BEND) * L, CY + math.sin(a - self.BEND) * L
            out.append((CX, CY, mx, my, self.ARM_W, CX, CY, self.OM, 0.0, 0.0))
            out.append((mx, my, tx, ty, self.ARM_W, CX, CY, self.OM, 0.0, 0.0))
        out.append((CX, CY, CX, CY, 44.0, CX, CY, self.OM, 0.0, 0.0))     # Nabe
        return out

    def starts(self):
        rng = self.rng
        SP = np.array([(CX - 230, CY - 210), (CX + 240, CY + 120), (CX - 100, CY + 250)], float)
        a = rng.uniform(0, 2 * math.pi, 3)
        return SP, np.stack([np.cos(a), np.sin(a)], 1) * self.SPEED


class Presse(JellySim):
    GRAV = 1700.0
    DAMP = 0.6
    E_WALL = 0.35
    T = 2.0
    PW = 0.5 * HALF            # halbe Breite der Pressplatte
    PT = 24.0                  # halbe Dicke
    N8 = 220

    def crushed(self, P, n, r, pen, pen0, nx0, ny0):
        c = np.nonzero((pen[:n] > self.CRUSH * r) & (P[:n, 1] > self.ypos_now))[0]   # nur unter der Platte
        return c[np.argsort(-pen[c])]

    def parts(self, t):
        self.ypos_now = self.ypos(t)
        return self._parts(t)


    def ypos(self, t):
        u = (t % self.T) / self.T
        top = Y0 + self.PT + self.R * 4.0      # oben bleibt Platz: nichts klemmt an der Decke
        bot = Y1 - self.PT - self.R * 1.25
        if u < 0.55:                          # langsam, schwer nach unten
            e = u / 0.55
            s = e * e * (3 - 2 * e)
        elif u < 0.68:                        # halten (pressen)
            s = 1.0
        else:                                 # zurück nach oben
            e = (u - 0.68) / 0.32
            s = 1 - e * e * (3 - 2 * e)
        return top + (bot - top) * s

    def _parts(self, t):
        y = self.ypos(t)
        vy = (self.ypos(t + 1e-3) - self.ypos(t - 1e-3)) / 2e-3
        return [(CX - self.PW, y, CX + self.PW, y, self.PT, CX, y, 0.0, 0.0, vy)]

    def machine_speed(self, t):
        return abs(self._parts(t)[0][-1]) / 600

    def starts(self):
        SP = np.array([(CX - 120, Y1 - 60), (CX + 20, Y1 - 140), (CX + 140, Y1 - 60)], float)
        return SP, np.zeros((3, 2))


class Walze(JellySim):
    GRAV = 1700.0
    DAMP = 0.5
    E_WALL = 0.35
    RR = 95.0                  # Walzenradius
    T = 3.6
    N8 = 300

    def xpos(self, t):
        A_ = HALF - self.RR - 4
        return CX - A_ * math.cos(2 * math.pi * t / self.T)

    def parts(self, t):
        x = self.xpos(t)
        vx = (self.xpos(t + 1e-3) - self.xpos(t - 1e-3)) / 2e-3
        y = Y1 - self.RR - self.R * 1.2
        return [(x, y, x, y, self.RR, x, y, vx / self.RR, vx, 0.0)]

    def machine_speed(self, t):
        return abs(self.parts(t)[0][-2]) / 450

    def starts(self):
        SP = np.array([(CX - 60, Y1 - 40), (CX + 80, Y1 - 40), (CX + 200, Y1 - 40)], float)
        return SP, np.zeros((3, 2))


# ------------------------------------------------------------------ Optik
def jelly_sprite(r, col):
    """Glänzender, durchscheinender Jelly-Ball (Größe 2.6 r) inkl. weichem Schatten."""
    n = int(r * 2.6)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2 + 0.5
    d = np.hypot(xx, yy) / r
    c = np.array(col, np.float32)
    inside = np.clip((1 - d) * r + 0.7, 0, 1)
    edge = np.clip((d - 0.55) / 0.45, 0, 1)
    rgb = c * (0.78 + 0.25 * (1 - edge[..., None])) * (1 - 0.35 * edge[..., None] ** 2)
    glow = np.exp(-(((xx - r * 0.25) / (r * 0.55)) ** 2 + ((yy - r * 0.32) / (r * 0.45)) ** 2))
    rgb = rgb + glow[..., None] * (255 - rgb) * 0.38                 # Licht scheint durch (unten rechts)
    rim = np.clip(1 - np.abs(d - 0.9) / 0.08, 0, 1) * np.clip(-(xx + yy) / (r * 1.2), 0, 1)
    rgb = rgb + rim[..., None] * (255 - rgb) * 0.45                  # Randlicht oben links
    rot = math.radians(-35)
    ux = (xx + r * 0.36) * math.cos(rot) - (yy + r * 0.42) * math.sin(rot)
    uy = (xx + r * 0.36) * math.sin(rot) + (yy + r * 0.42) * math.cos(rot)
    spec = np.clip(1.4 - ((ux / (r * 0.34)) ** 2 + (uy / (r * 0.17)) ** 2), 0, 1) ** 0.6
    dot = np.exp(-(((xx - r * 0.42) / (r * 0.09)) ** 2 + ((yy - r * 0.5) / (r * 0.07)) ** 2))
    rgb = rgb + (spec * 0.95 + dot * 0.6)[..., None] * (255 - rgb)
    alpha = inside * (0.86 + 0.14 * edge + 0.1 * spec)
    sh = np.clip(1 - np.hypot(xx - r * 0.12, yy - r * 0.3) / (r * 1.12), 0, 1) ** 1.5 * 0.45 * (1 - inside)
    out_rgb = rgb * inside[..., None]
    a = np.clip(alpha + sh, 0, 1)
    out_rgb = np.where(a[..., None] > 0, out_rgb / np.maximum(a, 1e-3)[..., None], 0)
    return Image.fromarray(np.dstack([np.clip(out_rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA")


class JellyRenderer:
    THEME = {}

    def __init__(self, sim):
        self.sim = sim
        th = self.THEME
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        bg = np.array(th["bg"], np.float32) + np.exp(-(((xx - CX) / 560) ** 2 + ((yy - CY) / 620) ** 2))[..., None] \
            * np.array(th["bg_glow"], np.float32)
        self.base = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(self.base, "RGBA")
        fr = th["frame"]
        if th.get("round"):
            for w_, al in ((22, 30), (14, 60), (8, 255)):
                rr = HALF + w_ / 4
                d.ellipse([(CX - rr) * k, (CY - rr) * k, (CX + rr) * k, (CY + rr) * k], outline=fr + (al,),
                          width=int(w_ * k / 2))
            rr = HALF + 1
            d.ellipse([(CX - rr) * k, (CY - rr) * k, (CX + rr) * k, (CY + rr) * k], outline=(255, 255, 255, 120),
                      width=k)
        for w_, al in (() if th.get("round") else ((22, 30), (14, 60), (8, 255))):
            d.rounded_rectangle([(X0 - w_ / 2) * k, (Y0 - w_ / 2) * k, (X1 + w_ / 2) * k, (Y1 + w_ / 2) * k],
                                radius=10 * k, outline=fr + (al,), width=int(w_ * k / 2))
        if not th.get("round"):
            d.rounded_rectangle([(X0 - 3) * k, (Y0 - 3) * k, (X1 + 3) * k, (Y1 + 3) * k], radius=8 * k,
                                outline=(255, 255, 255, 120), width=k)
        self.cache = {}
        self.cols = th["balls"]

    def sprite(self, g, sx, sy):
        key = (g, round(sx * 50), round(sy * 50))
        sp = self.cache.get(key)
        if sp is None:
            base = self.cache.get((g, 50, 50))
            if base is None:
                base = jelly_sprite(self.sim.R * SS, self.cols[g % len(self.cols)])
                self.cache[(g, 50, 50)] = base
            sp = base if key[1:] == (50, 50) else base.resize(
                (max(2, int(base.width * key[1] / 50)), max(2, int(base.height * key[2] / 50))), Image.BICUBIC)
            self.cache[key] = sp
        return sp

    def draw_machine(self, d, parts, fi):
        raise NotImplementedError

    def frame(self, fi):
        sim, k = self.sim, SS
        t = (fi + 1) / FPS
        Pn, gens, born = sim.frames[fi]
        jg = sim.jig[fi]
        im = self.base.copy()
        back = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.draw_back(ImageDraw.Draw(back, "RGBA"), sim.parts_log[fi], fi)
        im.alpha_composite(back)
        for i in range(len(Pn)):
            x, y = Pn[i]
            jx, jy = jg[i]
            e = (t - born[i]) / 0.22
            pop = 1.0
            if 0 <= e < 1:                     # aus der Quetschung heraus aufploppen
                pop = 0.55 + 0.45 * (1 - (1 - e) ** 3) + 0.12 * math.sin(e * math.pi)
            sx = (1 - jx + 0.5 * jy) * pop
            sy = (1 - jy + 0.5 * jx) * pop
            sp = self.sprite(int(gens[i]), sx, sy)
            im.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        top = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.draw_machine(ImageDraw.Draw(top, "RGBA"), sim.parts_log[fi], fi)
        sh = top.filter(ImageFilter.GaussianBlur(8 * k))
        sa = np.asarray(sh).copy()
        sa[..., :3] = 0
        sa[..., 3] = (sa[..., 3] * 0.45).astype(np.uint8)
        im.alpha_composite(Image.fromarray(sa, "RGBA"), (6 * k, 10 * k))
        im.alpha_composite(top)
        out = im.convert("RGB").reduce(SS)
        n_vis = len(Pn)
        d = ImageDraw.Draw(out, "RGBA")
        d.text((CX, Y1 + 120), f"{n_vis:,}", font=_font(F_BOLD, 58), fill=self.THEME["text"] + (240,), anchor="mm")
        return out

    def draw_back(self, d, parts, fi):
        pass


def glossy_capsule(d, ax, ay, bx, by, rad, base, hi, dark, k=SS):
    """Kapsel mit Glanz: dunkler Rand, Grundfarbe, heller Glanzstreifen oben."""
    def seg(r_, col, oy=0.0, shrink=0.0):
        L = math.hypot(bx - ax, by - ay)
        ux, uy = ((bx - ax) / L, (by - ay) / L) if L > 1e-6 else (1.0, 0.0)
        a2 = (ax + ux * shrink, ay + uy * shrink + oy)
        b2 = (bx - ux * shrink, by - uy * shrink + oy)
        d.line([(a2[0] * k, a2[1] * k), (b2[0] * k, b2[1] * k)], fill=col, width=int(2 * r_ * k))
        for (px, py) in (a2, b2):
            d.ellipse([(px - r_) * k, (py - r_) * k, (px + r_) * k, (py + r_) * k], fill=col)
    seg(rad + 2.5, dark + (255,))
    seg(rad, base + (255,))
    seg(rad * 0.55, tuple(int(b + (h - b) * 0.45) for b, h in zip(base, hi)) + (255,), oy=-rad * 0.25,
        shrink=rad * 0.2)
    seg(rad * 0.18, hi + (230,), oy=-rad * 0.5, shrink=rad * 0.6)


class DrehkreuzRenderer(JellyRenderer):
    THEME = dict(bg=(6, 18, 22), bg_glow=(10, 50, 58), frame=(150, 245, 235), text=(225, 255, 250), round=True,
                 balls=[(255, 70, 120), (255, 170, 40), (250, 235, 60), (70, 230, 120), (40, 190, 255),
                        (160, 100, 255)])

    def draw_machine(self, d, parts, fi):
        for (ax, ay, bx, by, rad, *_r) in parts[:8]:
            glossy_capsule(d, ax, ay, bx, by, rad, (215, 222, 232), (255, 255, 255), (60, 70, 82))
        ax, ay, _, _, rad, *_r = parts[8]
        k = SS
        for rr, col in ((rad + 3, (60, 70, 82)), (rad, (200, 208, 220)), (rad * 0.62, (235, 240, 248)),
                        (rad * 0.25, (90, 100, 115))):
            d.ellipse([(ax - rr) * k, (ay - rr) * k, (ax + rr) * k, (ay + rr) * k], fill=col + (255,))
        d.ellipse([(ax - rad * 0.55) * k, (ay - rad * 0.7) * k, (ax - rad * 0.05) * k, (ay - rad * 0.35) * k],
                  fill=(255, 255, 255, 200))


class PresseRenderer(JellyRenderer):
    THEME = dict(bg=(16, 16, 20), bg_glow=(40, 38, 50), frame=(255, 120, 90), text=(255, 235, 225),
                 balls=[(255, 50, 70), (255, 140, 30), (255, 220, 40), (120, 230, 60), (255, 90, 190)])

    def draw_back(self, d, parts, fi):
        k = SS
        ax, ay, bx, by, rad, *_r = parts[0]
        for x_ in (CX - 34, CX + 34):           # Hydraulik-Stangen von oben
            d.rectangle([(x_ - 11) * k, 0, (x_ + 11) * k, ay * k], fill=(120, 126, 138, 255))
            d.rectangle([(x_ - 5) * k, 0, (x_ - 1) * k, ay * k], fill=(230, 236, 245, 255))
        d.rectangle([(CX - 70) * k, 0, (CX + 70) * k, (Y0 - 30) * k], fill=(70, 74, 84, 255))
        d.rectangle([(CX - 64) * k, 0, (CX - 50) * k, (Y0 - 30) * k], fill=(150, 156, 168, 255))

    def draw_machine(self, d, parts, fi):
        ax, ay, bx, by, rad, *_r = parts[0]
        k = SS
        d.rounded_rectangle([(ax - rad) * k, (ay - rad) * k, (bx + rad) * k, (by + rad) * k], radius=10 * k,
                            fill=(48, 52, 62, 255))
        d.rounded_rectangle([(ax - rad + 3) * k, (ay - rad + 3) * k, (bx + rad - 3) * k, (by + rad - 3) * k],
                            radius=8 * k, fill=(150, 156, 170, 255))
        d.rounded_rectangle([(ax - rad + 8) * k, (ay - rad + 6) * k, (bx + rad - 8) * k, (ay - 4) * k],
                            radius=6 * k, fill=(222, 228, 238, 255))
        d.rectangle([(ax - rad + 30) * k, (ay - rad + 8) * k, (bx - 60) * k, (ay - rad + 12) * k],
                    fill=(255, 255, 255, 220))
        for x_ in np.linspace(ax + 10, bx - 10, 9):    # Warnstreifen unten
            d.line([((x_ - 8) * k, (ay + rad - 4) * k), ((x_ + 8) * k, (ay + 6) * k)], fill=(255, 190, 0, 255),
                   width=5 * k)


class WalzeRenderer(JellyRenderer):
    THEME = dict(bg=(20, 12, 30), bg_glow=(60, 30, 80), frame=(255, 170, 230), text=(255, 235, 250),
                 balls=[(120, 240, 255), (180, 140, 255), (255, 130, 220), (255, 200, 120), (140, 255, 170)])

    def draw_machine(self, d, parts, fi):
        x, y, _, _, R, ox, oy, om, *_r = parts[0]
        k = SS
        t = (fi + 1) / FPS
        ang = (self.sim.xpos(t) - CX) / R               # gerollter Winkel
        for rr, col in ((R + 3, (90, 40, 90)), (R, (240, 150, 210)), (R * 0.9, (250, 185, 230))):
            d.ellipse([(x - rr) * k, (y - rr) * k, (x + rr) * k, (y + rr) * k], fill=col + (255,))
        for j in range(8):                               # Rillen drehen sich mit -> man sieht das Rollen
            a = ang + j * math.pi / 4
            d.line([((x + math.cos(a) * R * 0.3) * k, (y + math.sin(a) * R * 0.3) * k),
                    ((x + math.cos(a) * R * 0.86) * k, (y + math.sin(a) * R * 0.86) * k)],
                   fill=(200, 100, 170, 255), width=4 * k)
        for rr, col in ((R * 0.3, (150, 70, 130)), (R * 0.18, (255, 225, 245))):
            d.ellipse([(x - rr) * k, (y - rr) * k, (x + rr) * k, (y + rr) * k], fill=col + (255,))
        d.ellipse([(x - R * 0.72) * k, (y - R * 0.82) * k, (x - R * 0.12) * k, (y - R * 0.5) * k],
                  fill=(255, 255, 255, 170))


# ------------------------------------------------------------------ Ton
def _squelch(rng):
    n = int(SR * 0.28)
    tt = np.arange(n) / SR
    f0 = rng.uniform(0.85, 1.2)
    blob_f = (330 * np.exp(-tt / 0.05) + 110) * f0
    blob = np.sin(2 * np.pi * np.cumsum(blob_f) / SR) * np.exp(-tt / 0.09)
    from scipy.signal import lfilter
    noise = rng.standard_normal(n)
    wet = np.zeros(n)
    fc = 2200 * np.exp(-tt / 0.04) + 380                  # nasses Schmatzen: Resonanz fällt schnell
    y1 = y2 = 0.0
    for i in range(0, n):
        w0 = 2 * math.pi * fc[i] / SR
        a1, a2 = 2 * 0.985 * math.cos(w0), -0.985 * 0.985
        y = noise[i] * 0.06 + a1 * y1 + a2 * y2
        wet[i] = y
        y2, y1 = y1, y
    wet *= np.exp(-tt / 0.06)
    wet /= np.abs(wet).max() + 1e-9
    return (blob * 0.8 + wet * 0.45) * np.minimum(1, tt / 0.003)


def _blop(rng):
    n = int(SR * 0.07)
    tt = np.arange(n) / SR
    f = (620 * np.exp(-tt / 0.02) + 260) * rng.uniform(0.85, 1.2)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.022) * np.minimum(1, tt / 0.002)


def audio(sim, path):
    rng = np.random.default_rng(9)
    dur = sim.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + SR)
    bank = []                                     # "HUSS!" (6 Varianten) bei JEDER Quetschung
    for j in range(1, 7):
        with wave.open(f"assets/huss_{j}.wav") as w:
            bank.append(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768)
    for t, kind, dd in sim.events:
        if kind == "split" and t < dur:
            s = bank[rng.integers(len(bank))]
            i = int(t * SR)
            out[i:i + len(s)] += s * 0.7
    last = -1
    for t, v, c in sim.bounces:
        if t - last < 0.07 or t >= dur:
            continue
        last = t
        fi = min(sim.n_frames - 1, int(t * FPS))
        cnt = len(sim.frames[fi][0])
        s = _blop(rng)
        i = int(t * SR)
        out[i:i + len(s)] += s * min(1.0, v / 700) * 0.22 / (1 + cnt / 150)
    # Maschine: weiches tiefes Brummen, folgt ihrem Tempo
    from scipy.signal import lfilter
    ms = np.interp(np.arange(n) / SR, (np.arange(len(sim.mspeed)) + 1) / FPS, np.array(sim.mspeed))
    hum = np.sin(2 * np.pi * np.cumsum(55 + 20 * ms) / SR) + 0.4 * np.sin(2 * np.pi * np.cumsum(110 + 40 * ms) / SR)
    noise = lfilter([0.02], [1, -0.98], np.random.default_rng(2).standard_normal(n))
    out[:n] += (hum * 0.05 + noise * 0.25) * (0.4 + 0.6 * np.clip(ms, 0, 1.5))
    out = out[:n]
    out *= np.clip(np.arange(n) / (0.02 * SR), 0, 1) * np.clip((n - np.arange(n)) / (0.15 * SR), 0, 1)
    out = np.tanh(out * 1.4) * 0.9
    st = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


CONFIGS = {
    "drehkreuz": (Drehkreuz, DrehkreuzRenderer, "output/j01_jelly_drehkreuz_8s.mp4"),
    "presse": (Presse, PresseRenderer, "output/j02_jelly_presse_8s.mp4"),
    "walze": (Walze, WalzeRenderer, "output/j03_jelly_walze_8s.mp4"),
}

if __name__ == "__main__":
    name = sys.argv[1]
    S, Rc, out = CONFIGS[name]
    sim = S(3).run()
    sp = [e for e in sim.events if e[1] == "split"]
    print(name, "Quetschungen", len(sp), "erste", round(sp[0][0], 2) if sp else None,
          "Bälle", [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(9)])
    if "--preview" in sys.argv:
        R = Rc(sim)
        o = Image.new("RGB", (360 * 3, 640))
        for i, x in enumerate((1.5, 4.5, 7.9)):
            o.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), (i * 360, 0))
        o.save(f"output/preview/jelly_{name}.png")
        R.frame(int(7.9 * FPS) - 1).crop((300, 700, 800, 1250)).save(f"output/preview/jelly_{name}_zoom.png")
    else:
        A.ArcRenderer, A.build_audio = Rc, audio
        A.render(sim, out)
        print(out)
