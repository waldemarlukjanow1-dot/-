"""Drei neue 8-Sekunden-Ideen, alle nach den Wünschen aus PREFERENCES.md:
keine Überschrift, Zähler unter dem Kreis, keine Ring-Effekte, kein Schluss-Sound, die Bälle bewegen sich nur
durch die Maschine (Physik). Ein Ball verdoppelt sich nur, wenn er wirklich eingeklemmt/zerquetscht wird; die
beiden Hälften behalten einfach seine Bewegung (keine künstlich gesetzte Geschwindigkeit).

  pendulum : ein schwerer Pendelhammer schwingt durch den Kreis und quetscht Bälle an der Kreiswand
  drum     : der ganze Kreis dreht sich, nimmt die Bälle an der Wand mit und zieht sie in den Spalt einer Walze
  weight   : ein Fallgewicht wird langsam hochgezogen und fällt frei auf die Bodenplatte
"""
import math
import os
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb
from .multiply import (CX, CY, R_AR, FPS, W1, H1, SS, G, Scene, poly_body, circle_body, rect_poly, ball_col, _font,
                       F_BOLD)

SR = 44100


class Base(Scene):
    R = 11.0
    CAP = 2500
    DUR = 8.0
    SUB = 10
    CITER = 4
    GAP = 0.15

    def eject(self, P, V, i, j, t):
        """Zerquetscht -> zwei Hälften nebeneinander, beide mit der Bewegung des alten Balls."""
        a = self.rng.uniform(0, 2 * math.pi)
        off = np.array((math.cos(a), math.sin(a))) * self.R * 0.55
        P[j] = P[i] + off
        P[i] = P[i] - off
        V[j] = V[i]

    def on_wall(self, P, n):
        return np.hypot(P[:n, 0] - CX, P[:n, 1] - CY) > R_AR - self.R - 1.5


# ------------------------------------------------------------------ 1) Pendelhammer

class Pendulum(Base):
    L = 365.0                 # Abstand Drehpunkt -> Mitte Hammerkopf (Spalt zur Wand < Balldurchmesser)
    HEAD_W, HEAD_H = 64.0, 76.0
    AMP = math.radians(100)
    T = 1.15

    def theta(self, t):
        return self.AMP * math.sin(2 * math.pi * t / self.T)

    def omega(self, t):
        return self.AMP * 2 * math.pi / self.T * math.cos(2 * math.pi * t / self.T)

    def head(self, th):
        hx, hy = CX + math.sin(th) * self.L, CY + math.cos(th) * self.L
        return rect_poly(hx, hy, self.HEAD_W, self.HEAD_H, -th)

    def arm(self, th):
        mx, my = CX + math.sin(th) * self.L / 2, CY + math.cos(th) * self.L / 2
        return rect_poly(mx, my, 16, self.L - self.HEAD_H / 2, -th)

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX - 150 + 30 * k, CY + R_AR - 60 - 20 * (k % 2))
            V[k] = (0.0, 0.0)

    def machine(self, P, V, n, t, touch, impact):
        th, w = self.theta(t), self.omega(t)
        poly_body(P, V, n, self.R, self.head(th), 0.0, 0.0, CX, CY, -w, 0.2, touch, 1, impact, 1e9)
        poly_body(P, V, n, self.R, self.arm(th), 0.0, 0.0, CX, CY, -w, 0.2, touch, 2, impact, 1e9)
        circle_body(P, V, n, self.R, CX, CY, 26.0, 0.0, 0.3, touch, 4)

    def crushed(self, P, V, n, t, touch, impact):
        # zwischen Hammerkopf und Kreiswand eingeklemmt
        m = ((touch[:n] & 1) == 1) & self.on_wall(P, n)
        return np.nonzero(m)[0]

    def state(self, t):
        return (self.theta(t),)


# ------------------------------------------------------------------ 2) Drehende Trommel + Walze

class Drum(Base):
    OM = 2.6                  # Drehung des Kreises (rad/s)
    ROLL_R = 70.0
    ANG = math.radians(-160)  # Lage der Walze (links, etwas über der Mitte)
    GAPW = 15.0               # Spalt zwischen Walze und Wand (< Balldurchmesser)

    def __init__(self, seed, n0=4):
        super().__init__(seed, n0)
        d = R_AR - self.GAPW - self.ROLL_R
        self.rx, self.ry = CX + math.cos(self.ANG) * d, CY + math.sin(self.ANG) * d

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX - 135 + 30 * k, CY + R_AR - 40 - 20 * (k % 2))
            V[k] = (0.0, 0.0)

    def machine(self, P, V, n, t, touch, impact):
        # Wandreibung: Bälle an der Wand werden von der drehenden Trommel mitgenommen
        dx, dy = P[:n, 0] - CX, P[:n, 1] - CY
        d = np.hypot(dx, dy) + 1e-9
        m = d > R_AR - self.R - 2.0
        if m.any():
            tx, ty = -dy[m] / d[m], dx[m] / d[m]
            vt = V[:n][m, 0] * tx + V[:n][m, 1] * ty
            dv = (self.OM * R_AR - vt) * 0.06
            V[np.nonzero(m)[0], 0] += dv * tx
            V[np.nonzero(m)[0], 1] += dv * ty
        circle_body(P, V, n, self.R, self.rx, self.ry, self.ROLL_R, -self.OM * 1.4, 0.2, touch, 1)

    def crushed(self, P, V, n, t, touch, impact):
        m = ((touch[:n] & 1) == 1) & self.on_wall(P, n)
        return np.nonzero(m)[0]

    def eject(self, P, V, i, j, t):
        """Die zwei Hälften kommen auf der anderen Seite des Spalts heraus (Richtung Drehung)."""
        Base.eject(self, P, V, i, j, t)
        ang = math.atan2(self.ry - CY, self.rx - CX) + 0.42
        for k in (i, j):
            rr = R_AR - self.R - 3 - (k == j) * 2 * self.R
            P[k] = (CX + math.cos(ang) * rr, CY + math.sin(ang) * rr)

    def state(self, t):
        return (self.OM * t,)


# ------------------------------------------------------------------ 3) Fallgewicht

class Weight(Base):
    FLOOR = CY + 300
    WW, WH = 230.0, 150.0
    TOP = CY - 230            # Unterkante oben
    LIFT_T, HOLD_T = 0.3, 0.06
    GAP = 0.3

    def __init__(self, seed, n0=4):
        super().__init__(seed, n0)
        h = self.WW / 2 + 4
        self.floor = np.array([(CX - h, self.FLOOR), (CX + h, self.FLOOR), (CX + h, CY + R_AR + 80),
                               (CX - h, CY + R_AR + 80)], np.float64)
        self.slopes = [np.array([(CX - h, self.FLOOR), (CX - R_AR - 40, self.FLOOR - 150), (CX - R_AR - 40, CY + R_AR + 80),
                                 (CX - h, CY + R_AR + 80)], np.float64),
                       np.array([(CX + h, self.FLOOR), (CX + h, CY + R_AR + 80), (CX + R_AR + 40, CY + R_AR + 80),
                                 (CX + R_AR + 40, self.FLOOR - 150)], np.float64)]
        low = self.FLOOR - 2 * self.R * 0.6
        self.fall_T = math.sqrt(2 * (low - self.TOP) / (G * 1.4))    # freier Fall (schweres Gewicht)
        self.cycle = self.LIFT_T + self.HOLD_T + self.fall_T + 0.12
        self.low = low

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX - 90 + 25 * k, self.FLOOR - 40)
            V[k] = (0.0, 0.0)

    def wy(self, t):
        u = t % self.cycle
        if u < self.LIFT_T:                               # langsam hochziehen
            v = u / self.LIFT_T
            return self.low + (self.TOP - self.low) * v * v * (3 - 2 * v)
        u -= self.LIFT_T
        if u < self.HOLD_T:
            return self.TOP
        u -= self.HOLD_T
        if u < self.fall_T:                               # loslassen: freier Fall
            return self.TOP + 0.5 * G * 1.4 * u * u
        return self.low

    def machine(self, P, V, n, t, touch, impact):
        y = self.wy(t)
        e = 1e-3
        vy = (self.wy(t + e) - self.wy(t - e)) / (2 * e)
        poly_body(P, V, n, self.R, rect_poly(CX, y - self.WH / 2, self.WW, self.WH), 0.0, vy, 0.0, 0.0, 0.0, 0.1,
                  touch, 1, impact, 1e9)
        poly_body(P, V, n, self.R, self.floor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, touch, 2, impact, 1e9)
        for q in self.slopes:
            poly_body(P, V, n, self.R, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, touch, 4, impact, 1e9)

    def crushed(self, P, V, n, t, touch, impact):
        if self.wy(t) < self.low - 1.2 * self.R:
            return np.zeros(0, np.int64)
        m = ((touch[:n] & 3) == 3)
        return np.nonzero(m)[0]

    def eject(self, P, V, i, j, t):
        """Unter dem Gewicht ist kein Platz: die Hälften werden seitlich herausgedrückt (an den Rand des Gewichts)."""
        s = 1 if P[i, 0] >= CX else -1
        V[j] = V[i]
        for k, sg in ((i, s), (j, s)):
            P[k, 0] = CX + sg * (self.WW / 2 + self.R + 1 + (k == j) * 2 * self.R)
            P[k, 1] = self.FLOOR - self.R - 0.5

    def state(self, t):
        return (self.wy(t),)


class WeightLong(Weight):
    """Lange Version: das Gewicht fällt immer wieder. Zerquetscht wird jeder Ball direkt unter dem Gewicht, der
    nach unten nicht ausweichen kann (Bodenplatte oder andere Bälle darunter). Am Ende ist der Kreis bis zum Rand voll,
    das Gewicht wird hochgezogen und bleibt oben."""
    DUR = 80.0
    CAP = 3200
    TOP = CY - R_AR + 40
    LIFT_T, HOLD_T = 0.6, 0.08
    N_FULL = 2400
    OUTRO = 4.0

    def __init__(self, seed, n0=8):
        super().__init__(seed, n0)
        self.GAP = self.cycle * 0.9      # jeder Ball höchstens einmal pro Fall
        self.full_t = None

    def wy(self, t):
        if self.full_t is not None and t >= self.full_t:
            e = min(1.0, (t - self.full_t) / 1.2)
            return self._y_full + (self.TOP - self._y_full) * e * e * (3 - 2 * e)
        return Weight.wy(self, t)

    def crushed(self, P, V, n, t, touch, impact):
        if self.full_t is not None:
            return np.zeros(0, np.int64)
        e = 1e-3
        vy = (self.wy(t + e) - self.wy(t - e)) / (2 * e)
        if vy < 200 and self.wy(t) < self.low - 1.2 * self.R:
            return np.zeros(0, np.int64)        # nur während das Gewicht fällt bzw. aufschlägt
        y = self.wy(t)
        under = ((touch[:n] & 1) == 1) & (np.abs(P[:n, 0] - CX) < self.WW / 2) & (P[:n, 1] > y)
        out = []
        for i in np.nonzero(under)[0]:
            if touch[i] & 2:
                out.append(i)
                continue
            dx, dy = P[:n, 0] - P[i, 0], P[:n, 1] - P[i, 1]
            if np.any((dy > self.R * 0.8) & (dy < self.R * 2.3) & (np.abs(dx) < self.R * 1.6)):
                out.append(i)
        return np.array(out, np.int64)

    def eject(self, P, V, i, j, t):
        if P[i, 1] > self.FLOOR - 2 * self.R:   # unterste Reihe: kein Platz -> seitlich am Gewicht heraus
            Weight.eject(self, P, V, i, j, t)
        else:
            Base.eject(self, P, V, i, j, t)

    def frame_hook(self, P, n, t):
        top = int(np.sum(P[:n, 1] < CY - R_AR + 45))
        if top >= 8:
            self._top_since = getattr(self, "_top_since", None) or t
        else:
            self._top_since = None
        if self.full_t is None and (n >= self.N_FULL or (self._top_since and t - self._top_since > 1.5)):
            self._y_full = self.wy(t)
            self.full_t = t
            self.events.append((t, "full", {}))
        return self.full_t is not None and t > self.full_t + self.OUTRO


# ------------------------------------------------------------------ Darstellung

RING_A, RING_B = (255, 90, 190), (90, 200, 255)
LAV, LAV_HI, LAV_D = (178, 168, 235), (225, 220, 255), (120, 110, 190)
GOLD, GOLD_HI, GOLD_D = (255, 214, 120), (255, 240, 200), (215, 160, 70)


class Renderer:
    def __init__(self, sc):
        self.sc = sc
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        f = (yy / H1)[..., None]
        base = np.array((12, 10, 28), np.float32) * (1 - f) + np.array((26, 10, 36), np.float32) * f
        base += np.exp(-(((xx - CX) / 560) ** 2 + ((yy - CY) / 560) ** 2))[..., None] * np.array((26, 18, 50))
        d2 = (xx - CX) ** 2 + (yy - CY) ** 2
        base[d2 < R_AR ** 2] = base[d2 < R_AR ** 2] * 0.5 + np.array((5, 4, 12))
        dist = np.sqrt(d2)
        ang = (np.arctan2(yy - CY, xx - CX) + math.pi) / (2 * math.pi)
        mixv = (0.5 + 0.5 * np.cos(ang * 2 * math.pi))[..., None]
        col = np.array(RING_A, np.float32) * mixv + np.array(RING_B, np.float32) * (1 - mixv)
        band = np.clip(1 - np.abs(dist - (R_AR + 12)) / 12, 0, 1) ** 0.6
        glow = np.exp(-((dist - (R_AR + 12)) / 34) ** 2) * 0.55
        base = base * (1 - band[..., None]) + col * band[..., None] + col * glow[..., None] * 0.6
        self.base = Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("RGBA")
        m = Image.new("L", self.base.size, 0)
        ImageDraw.Draw(m).ellipse([(CX - R_AR) * k, (CY - R_AR) * k, (CX + R_AR) * k, (CY + R_AR) * k], fill=255)
        self.mask = m
        self.splits = [e[0] for e in sc.events if e[1] == "split"]

    def poly(self, d, pts, col):
        d.polygon([(x * SS, y * SS) for x, y in pts], fill=col + (255,))

    def machine(self, d, st):
        sc, k = self.sc, SS
        if isinstance(sc, Pendulum):
            th = st[0]
            self.poly(d, sc.arm(th), LAV)
            h = sc.head(th)
            self.poly(d, h, GOLD)
            # Glanzkante auf der Kopfoberseite
            d.line([(h[0][0] * k, h[0][1] * k), (h[1][0] * k, h[1][1] * k)], fill=GOLD_HI + (255,), width=5 * k)
            d.ellipse([(CX - 26) * k, (CY - 26) * k, (CX + 26) * k, (CY + 26) * k], fill=LAV_D + (255,))
            d.ellipse([(CX - 10) * k, (CY - 10) * k, (CX + 10) * k, (CY + 10) * k], fill=LAV_HI + (255,))
        elif isinstance(sc, Drum):
            # drehende Markierungen am Ring zeigen die Trommeldrehung
            for q in range(24):
                a = st[0] + q * 2 * math.pi / 24
                x0, y0 = CX + math.cos(a) * (R_AR - 4), CY + math.sin(a) * (R_AR - 4)
                x1, y1 = CX + math.cos(a) * (R_AR + 22), CY + math.sin(a) * (R_AR + 22)
                d.line([(x0 * k, y0 * k), (x1 * k, y1 * k)], fill=(255, 255, 255, 110), width=4 * k)
            R = sc.ROLL_R
            d.ellipse([(sc.rx - R) * k, (sc.ry - R) * k, (sc.rx + R) * k, (sc.ry + R) * k], fill=GOLD + (255,))
            for q in range(10):
                a = -st[0] * 1.4 * (R_AR / R) * 0.2 + q * math.pi / 5
                d.line([((sc.rx + math.cos(a) * R * 0.3) * k, (sc.ry + math.sin(a) * R * 0.3) * k),
                        ((sc.rx + math.cos(a) * R * 0.92) * k, (sc.ry + math.sin(a) * R * 0.92) * k)],
                       fill=GOLD_D + (255,), width=5 * k)
            d.ellipse([(sc.rx - 16) * k, (sc.ry - 16) * k, (sc.rx + 16) * k, (sc.ry + 16) * k], fill=LAV_D + (255,))
        else:
            self.poly(d, sc.floor, LAV)
            for q in sc.slopes:
                self.poly(d, q, LAV)
            sl = sc.slopes
            d.line([(sl[0][1][0] * k, sl[0][1][1] * k), (sc.floor[0][0] * k, sc.floor[0][1] * k),
                    (sc.floor[1][0] * k, sc.floor[1][1] * k), (sl[1][3][0] * k, sl[1][3][1] * k)],
                   fill=LAV_HI + (255,), width=5 * k, joint="curve")
            y = st[0]
            d.line([(CX * k, (CY - R_AR - 40) * k), (CX * k, (y - sc.WH) * k)], fill=LAV_HI + (255,), width=5 * k)
            x0, x1, y0 = CX - sc.WW / 2, CX + sc.WW / 2, y - sc.WH
            d.rounded_rectangle([x0 * k, y0 * k, x1 * k, y * k], radius=16 * k, fill=GOLD + (255,))
            d.rounded_rectangle([(x0 + 8) * k, (y0 + 7) * k, (x1 - 8) * k, (y0 + 20) * k], radius=5 * k,
                                fill=GOLD_HI + (255,))
            d.rounded_rectangle([(x0 + 6) * k, (y - 18) * k, (x1 - 6) * k, (y - 5) * k], radius=5 * k,
                                fill=GOLD_D + (255,))
            d.ellipse([(CX - 14) * k, (y0 - 14) * k, (CX + 14) * k, (y0 + 8) * k], outline=LAV_HI + (255,),
                      width=4 * k)

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
        self.machine(ImageDraw.Draw(lay, "RGBA"), st)
        clip = Image.new("L", im.size, 0)
        ImageDraw.Draw(clip).ellipse([(CX - R_AR - 26) * k, (CY - R_AR - 26) * k, (CX + R_AR + 26) * k,
                                      (CY + R_AR + 26) * k], fill=255)
        im.paste(lay, (0, 0), Image.composite(lay.getchannel("A"), Image.new("L", lay.size, 0),
                                              clip if isinstance(sc, (Drum, Weight)) else self.mask))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(g, np.float32) - 50, 0, 255) * 0.35
        ft = getattr(sc, "full_t", None)
        if ft is not None and t > ft:     # Kreis voll: Ring leuchtet sanft auf (ohne Ton)
            e = t - ft
            a = min(1.0, e / 0.8) * (0.7 + 0.3 * math.cos(e * 3.0))
            fl = Image.new("L", (W1, H1), 0)
            ImageDraw.Draw(fl).ellipse([CX - R_AR - 12, CY - R_AR - 12, CX + R_AR + 12, CY + R_AR + 12], outline=255,
                                       width=24)
            fl = np.asarray(fl.filter(ImageFilter.GaussianBlur(14)), np.float32)[..., None] / 255
            arr = arr + fl * np.array((255, 225, 140)) * a * 1.2
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(out, "RGBA")
        last = [s for s in self.splits if s <= t]
        pop = 1 + 0.12 * max(0.0, 1 - (t - last[-1]) / 0.15) if last else 1.0
        d.text((CX, CY + R_AR + 84), f"{n:,}", font=_font(F_BOLD, int(54 * pop)), fill=(255, 255, 255, 235),
               anchor="mm")
        return out


# ------------------------------------------------------------------ Klang: weiches Plopp + tiefes Maschinen-Plopp

def build_audio(sc, path):
    from .breaker import plop
    dur = sc.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + SR * 2)

    def put(t, s_, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s_))
        if i < e:
            out[i:e] += s_[: e - i] * vol

    last = -1.0
    for t, kind, d in sc.events:
        if kind != "split" or t - last < 0.05:
            continue
        last = t
        f = 420 + 380 * min(1.0, math.log10(max(1, d["n"])) / 3.0)
        put(t, plop(round(f / 20) * 20), 0.45)
    if isinstance(sc, Pendulum):           # Hammer am Umkehrpunkt
        q = sc.T / 4
        while q < dur:
            put(q, plop(160, big=True), 0.3)
            q += sc.T / 2
    elif isinstance(sc, Weight):           # Gewicht schlägt auf
        q = sc.LIFT_T + sc.HOLD_T + sc.fall_T
        end = getattr(sc, "full_t", None) or dur
        while q < end:
            put(q, plop(140, big=True), 0.4)
            q += sc.cycle
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
    _R = Renderer(sc)
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


IDEAS = {"n01_pendulum_hammer": lambda: Pendulum(1, n0=10),
         "n02_spinning_drum": lambda: Drum(1, n0=10),
         "n03_drop_weight": lambda: Weight(1, n0=8),
         "n04_drop_weight_long": lambda: WeightLong(1, n0=8)}

if __name__ == "__main__":
    import sys
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(IDEAS)
    for name in names:
        sc = IDEAS[name]().run()
        print(name, [len(sc.frames[min(sc.n_frames - 1, q * FPS)][0]) for q in range(9)])
        if "--preview" in sys.argv:
            R = Renderer(sc)
            ims = [R.frame(min(sc.n_frames - 1, int(x * FPS))) for x in (0.5, 3.0, 5.5, 7.95)]
            out = Image.new("RGB", (360 * 4, 640))
            for i, im in enumerate(ims):
                out.paste(im.resize((360, 640)), (i * 360, 0))
            out.save(f"output/preview/{name}.png")
        else:
            render(sc, f"output/{name}_vertical.mp4")
            print(f"output/{name}_vertical.mp4")
