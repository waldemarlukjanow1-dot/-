"""Kreis-Kette: klassischer Ball-Bounce mit System.

Kreise liegen untereinander und schwingen sanft mit. Jeder Aufprall lässt einen Leuchtpunkt am Kreis
aufleuchten (mit steigendem Ton). Sind alle Punkte an, öffnet sich der Kreis unten – jedes Mal auf eine
andere Art (Schiebetüren, Blütenblätter, Falltür, Iris). Die Bälle fallen durch die obere Öffnung in den
nächsten Kreis und verdoppeln sich dabei: 1 -> 2 -> 4 -> 8 -> 16. Im letzten Kreis kommen alle zur Ruhe.
"""
import colorsys
import math
import os
import random
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tiktok.render import tiktok_ui_mock
from .marble_render import Background, ball_sprite, reverb, soft_sprite

FPS, SUB = 60, 8
DT = 1.0 / (FPS * SUB)
G = 1000.0
R_BALL = 16.0
RADIUS = 300.0
SPACING = 740.0
DOTS = [12, 20, 32, 48, 64]
GATE_TYPES = ["doors", "petals", "trapdoor", "iris"]
RING_HUES = [0.52, 0.86, 0.32, 0.08, 0.72]
EXIT_HALF = 0.4
ENTRY_HALF = 0.75
SWAY = 26.0


def hsv(h, s=0.75, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def back_out(p, c=1.5):
    p = min(max(p, 0.0), 1.0)
    return 1 + (c + 1) * (p - 1) ** 3 + c * (p - 1) ** 2


def angdiff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


# ------------------------------------------------------------------ Simulation

class Circle:
    def __init__(self, k):
        self.k = k
        self.y = k * SPACING
        self.R = RADIUS + (30 if k == len(DOTS) - 1 else 0)
        self.dots = DOTS[k]
        self.lit = []          # (t, farbe)
        self.open_t = None     # Ausgang öffnet
        self.close_t = None    # Ausgang schließt wieder
        self.entry_close_t = None
        self.expected_in = 1 if k == 0 else 2 ** (k - 1)
        self.entered = 0
        self.final = k == len(DOTS) - 1
        self.calm_t = None

    def cx(self, t):
        return SWAY * math.sin(0.7 * t + self.k * 1.7)

    def vx(self, t):
        return SWAY * 0.7 * math.cos(0.7 * t + self.k * 1.7)

    def exit_f(self, t):
        if self.open_t is None or t < self.open_t:
            return 0.0
        if self.close_t is not None and t >= self.close_t:
            return 1 - ease((t - self.close_t) / 0.5)
        return back_out((t - self.open_t) / 0.7, 1.2)

    def entry_f(self, t):
        if self.k == 0:
            return 0.0
        if self.entry_close_t is None or t < self.entry_close_t:
            return 1.0
        return 1 - ease((t - self.entry_close_t) / 0.5)


class OBall:
    def __init__(self, i, x, y, vx, vy, hue, c):
        self.i, self.x, self.y, self.vx, self.vy = i, x, y, vx, vy
        self.hue = hue
        self.color = hsv(hue, 0.7, 1.0)
        self.c = c            # Index des Kreises, in dem (oder zu dem) der Ball gehört
        self.inside = True
        self.last_note = -1.0


class OrbitSim:
    def __init__(self, seed, speed=(470, 640), outro=3.0):
        self.seed = seed
        self.rng = random.Random(seed)
        self.speed, self.outro = speed, outro
        self.t = 0.0
        self.events, self.frames, self.spawns = [], [], []
        self.circles = [Circle(k) for k in range(len(DOTS))]
        a = self.rng.uniform(0.6, 1.2) * self.rng.choice((-1, 1))
        self.hue_next = self.rng.random()
        self.balls = [OBall(0, 0.0, -80.0, math.cos(a) * 560, math.sin(a) * 300, self._hue(), 0)]
        self.finish_t = None

    def _hue(self):
        h = self.hue_next
        self.hue_next = (self.hue_next + 0.618034) % 1
        return h

    def ev(self, kind, **d):
        self.events.append((self.t, kind, d))

    def step(self):
        t, dt, rng = self.t, DT, self.rng
        final = self.circles[-1]
        calm = 0.0 if final.calm_t is None else min(1.0, (t - final.calm_t) / 3.5)
        for b in self.balls:
            b.vy += G * dt
            if calm and b.c == final.k:
                f = 1 - (0.4 + 2.2 * calm) * dt
                b.vx *= f
                b.vy *= f
            b.x += b.vx * dt
            b.y += b.vy * dt
        # Kugel-Kugel
        bs = self.balls
        for i in range(len(bs)):
            a = bs[i]
            for j in range(i + 1, len(bs)):
                c = bs[j]
                dx, dy = c.x - a.x, c.y - a.y
                if abs(dx) > 2 * R_BALL or abs(dy) > 2 * R_BALL:
                    continue
                d = math.hypot(dx, dy)
                if 0 < d < 2 * R_BALL:
                    nx, ny = dx / d, dy / d
                    rel = (a.vx - c.vx) * nx + (a.vy - c.vy) * ny
                    if rel > 0:
                        e = 0.3 if calm else 1.0
                        k = (1 + e) / 2 * rel
                        a.vx -= k * nx
                        a.vy -= k * ny
                        c.vx += k * nx
                        c.vy += k * ny
                    ov = (2 * R_BALL - d) / 2
                    a.x -= nx * ov
                    a.y -= ny * ov
                    c.x += nx * ov
                    c.y += ny * ov
        new = []
        for b in self.balls:
            C = self.circles[b.c]
            ccx, ccy = C.cx(t), C.y
            dx, dy = b.x - ccx, b.y - ccy
            d = math.hypot(dx, dy) or 1e-6
            ang = math.atan2(dy, dx)
            if not b.inside:
                # sanfte Führung zur oberen Öffnung des nächsten Kreises
                top_y = ccy - C.R
                if b.y < top_y:
                    tt = max(0.12, (top_y - b.y) / max(b.vy, 200.0))
                    want = (ccx - b.x) / tt
                    b.vx += (want - b.vx) * min(1.0, 6.0 * dt)
                # fällt zum nächsten Kreis: drinnen?
                if d < C.R - R_BALL and dy > -C.R + 2 * R_BALL:
                    b.inside = True
                    C.entered += 1
                    self.ev("enter", c=C.k, ball=b.i)
                    tw = OBall(len(self.balls) + len(new), b.x + R_BALL * 1.2, b.y, -b.vx + rng.uniform(-60, 60),
                               b.vy, self._hue(), C.k)
                    b.x -= R_BALL * 1.2
                    new.append(tw)
                    self.spawns.append((t, b.x + R_BALL * 1.2, b.y, tw.color))
                    if C.entered >= C.expected_in and C.entry_close_t is None:
                        C.entry_close_t = t + 0.25
                        self.ev("entry_close", c=C.k)
                elif d + R_BALL > C.R and angdiff(ang, -math.pi / 2) > ENTRY_HALF:
                    pass  # außen am oberen Bogen vorbei -> Ring stößt von außen ab (selten)
                continue
            exit_open = C.exit_f(t) > 0.25 and angdiff(ang, math.pi / 2) < EXIT_HALF * min(1.0, C.exit_f(t))
            entry_open = C.entry_f(t) > 0.5 and angdiff(ang, -math.pi / 2) < ENTRY_HALF and False
            if exit_open and d > C.R:
                b.inside = False
                b.c = C.k + 1
                self.ev("leave", c=C.k, ball=b.i)
                continue
            if d + R_BALL > C.R and not (exit_open and d > C.R - R_BALL * 1.5):
                nx, ny = dx / d, dy / d
                b.x, b.y = ccx + nx * (C.R - R_BALL), ccy + ny * (C.R - R_BALL)
                cvx = C.vx(t)
                rvx, rvy = b.vx - cvx, b.vy
                vn = rvx * nx + rvy * ny
                if vn > 0:
                    if calm and C.final:
                        e = 1 - 0.75 * calm
                        rvx -= (1 + e) * vn * nx
                        rvy -= (1 + e) * vn * ny
                        b.vx, b.vy = rvx + cvx, rvy
                    else:
                        rvx -= 2 * vn * nx
                        rvy -= 2 * vn * ny
                        a = rng.uniform(-0.18, 0.18)
                        cs, sn = math.cos(a), math.sin(a)
                        rvx, rvy = rvx * cs - rvy * sn, rvx * sn + rvy * cs
                        sp = math.hypot(rvx, rvy) or 1
                        tgt = min(max(sp, self.speed[0]), self.speed[1])
                        b.vx, b.vy = rvx / sp * tgt + cvx, rvy / sp * tgt
                    self.bounce(b, C, ang, min(1.0, vn / 800))
        self.balls += new
        # Kreis-Logik
        for C in self.circles:
            if C.open_t is not None and C.close_t is None and t > C.open_t + 0.8:
                if not any(b.c == C.k and b.inside for b in self.balls):
                    C.close_t = t
                    self.ev("exit_close", c=C.k)
        if final.calm_t is not None and self.finish_t is None and t > final.calm_t + 3:
            if all(math.hypot(b.vx, b.vy) < 20 for b in self.balls):
                self.finish_t = t
        self.t += dt

    def bounce(self, b, C, ang, strength):
        t = self.t
        counting = (C.k == 0 or C.entry_close_t is not None and t > C.entry_close_t) and len(C.lit) < C.dots
        if counting:
            C.lit.append((t, b.color))
            self.ev("dot", c=C.k, n=len(C.lit), strength=strength)
            if len(C.lit) == C.dots:
                if C.final:
                    C.calm_t = t + 0.4
                    self.ev("calm", c=C.k)
                else:
                    C.open_t = t + 0.35
                    self.ev("open", c=C.k)
        elif strength > 0.1 and t - b.last_note > 0.08:
            b.last_note = t
            self.ev("tick", c=C.k, strength=strength)
        if strength > 0.06:
            self.ev("hit", c=C.k, ang=ang, color=b.color, strength=strength)

    def run(self, max_t=70.0):
        while True:
            for _ in range(SUB):
                self.step()
            self.frames.append(dict(t=self.t, balls=[(b.x, b.y, b.color) for b in self.balls]))
            if self.finish_t is not None and self.t >= self.finish_t + self.outro:
                break
            if self.t > max_t:
                break
        return self


# ------------------------------------------------------------------ Darstellung

W1, H1, SS = 1080, 1920, 2
W2, H2 = W1 * SS, H1 * SS
CX, CY = 540, 900   # Bildschirmmitte des aktiven Kreises


class OrbitRenderer:
    def __init__(self, sim):
        self.sim = sim
        self.bg = Background(seed=9)
        self.spr = {}
        self.shadow = soft_sprite(R_BALL * SS, (6, 2, 20), 140, 5 * SS)
        self.hits = [(t, d) for t, k, d in sim.events if k == "hit"]
        self.hit_t = [t for t, _ in self.hits]
        self.opens = sorted((t, d["c"]) for t, k, d in sim.events if k == "open")
        self.cam = self._camera()

    def _camera(self):
        n = len(self.sim.frames)
        cam = np.zeros(n)
        for i, fr in enumerate(self.sim.frames):
            t = fr["t"]
            k = 0.0
            for to, c in self.opens:
                k = c + ease((t - to - 0.4) / 1.9) if t >= to else k
            cam[i] = k * SPACING
        return cam

    def ball(self, color):
        if color not in self.spr:
            self.spr[color] = ball_sprite(color, R_BALL)
        return self.spr[color]

    def frame(self, i):
        sim = self.sim
        fr = sim.frames[i]
        t = fr["t"]
        cam = self.cam[i]
        k = SS

        def P(x, y):
            return (CX * SS + x * k, CY * SS + (y - cam) * k)

        canvas = Image.new("RGBA", (W2, H2), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        for C in sim.circles:
            sy = CY + (C.y - cam)
            if sy < -500 or sy > H1 + 500:
                continue
            self.draw_circle(canvas, d, C, P, t)
        # Spawn-Pulse
        for t0, x, y, col in sim.spawns:
            e = t - t0
            if 0 <= e < 0.6:
                X, Y = P(x, y)
                r = (R_BALL + 60 * (1 - (1 - e / 0.6) ** 3)) * k
                d.ellipse([X - r, Y - r, X + r, Y + r], outline=col + (int(200 * (1 - e / 0.6)),), width=int(3 * k))
        for x, y, col in fr["balls"]:
            X, Y = P(x, y)
            canvas.alpha_composite(self.shadow, (int(X + 7 * k - self.shadow.width / 2), int(Y + 11 * k - self.shadow.height / 2)))
        for x, y, col in fr["balls"]:
            X, Y = P(x, y)
            s = self.ball(col)
            canvas.alpha_composite(s, (int(round(X - s.width / 2)), int(round(Y - s.height / 2))))
        small = canvas.reduce(SS)
        bg = self.bg.render(cam, 1.0, t)
        bg.alpha_composite(small)
        rgb = bg.convert("RGB")
        g = rgb.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5))
        arr = np.asarray(rgb, np.float32)
        glow = np.asarray(g.resize((W1, H1), Image.BILINEAR), np.float32)
        arr = arr + np.clip(glow - 35, 0, 255) * 0.6
        arr = arr * self.bg.vignette + self.bg.grain[i % len(self.bg.grain)]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def draw_circle(self, canvas, d, C, P, t):
        k = SS
        cx = C.cx(t)
        X, Y = P(cx, C.y)
        R = (C.R + 7) * k
        base = hsv(RING_HUES[C.k], 0.6, 1.0)
        done = C.close_t is not None and t > C.close_t + 0.5
        dim = 0.45 if done else 1.0
        if C.final and C.calm_t is not None:
            dim = 1.0 + 0.25 * ease((t - C.calm_t) / 2.0)
        col = tuple(int(min(255, c * dim)) for c in base)
        fe, fn = C.exit_f(t), C.entry_f(t)
        he = EXIT_HALF * min(1.0, fe) if C.exit_f(t) > 0 else 0.0
        hn = ENTRY_HALF * fn
        # Ring als Bogen ohne Öffnungen (Grad, im Uhrzeigersinn ab 3 Uhr)
        segs = []
        a_exit = (math.pi / 2 - he, math.pi / 2 + he)
        a_entry = (-math.pi / 2 - hn, -math.pi / 2 + hn)
        # Bogen von Ausgang-Ende bis Eingang-Start (links herum) und Eingang-Ende bis Ausgang-Start (rechts)
        segs.append((a_exit[1], a_entry[0] + 2 * math.pi))
        segs.append((a_entry[1], a_exit[0]))
        box = [X - R, Y - R, X + R, Y + R]
        for a0, a1 in segs:
            if a1 - a0 < 0.01:
                continue
            d.arc([b + o for b, o in zip(box, (9 * k, 14 * k, 9 * k, 14 * k))], math.degrees(a0), math.degrees(a1),
                  fill=(6, 2, 20, 70), width=int(16 * k))
            d.arc(box, math.degrees(a0), math.degrees(a1), fill=col + (255,), width=int(13 * k))
            d.arc([X - R + 3 * k, Y - R + 3 * k, X + R - 3 * k, Y + R - 3 * k], math.degrees(a0) + 1, math.degrees(a1) - 1,
                  fill=(255, 255, 255, 110), width=int(3 * k))
        # Öffnungs-Mechanik (Ausgang unten)
        if C.k < len(GATE_TYPES) and fe > 0:
            self.draw_gate(d, GATE_TYPES[C.k], X, Y, R, fe, col, k)
        # Eingangstüren oben: gleiten auseinander/zusammen
        if C.k > 0:
            for sgn in (-1, 1):
                a_mid = -math.pi / 2 + sgn * (ENTRY_HALF / 2 + ENTRY_HALF * fn * 0.9)
                a0, a1 = a_mid - ENTRY_HALF / 2, a_mid + ENTRY_HALF / 2
                Ro = R + 13 * k * fn
                d.arc([X - Ro, Y - Ro, X + Ro, Y + Ro], math.degrees(a0), math.degrees(a1), fill=col + (255,), width=int(11 * k))
        # Leuchtpunkte (drehen langsam mit)
        rot = 0.12 * t * (1 if C.k % 2 == 0 else -1)
        Rd = (C.R + 34) * k
        for j in range(C.dots):
            a = rot + 2 * math.pi * j / C.dots - math.pi / 2
            px, py = X + math.cos(a) * Rd, Y + math.sin(a) * Rd
            if j < len(C.lit) and t >= C.lit[j][0]:
                t0, lc = C.lit[j]
                e = t - t0
                s = 1 + 0.8 * max(0.0, 1 - e / 0.3)
                r = 7 * k * s
                if e < 0.5:
                    rr = (8 + 26 * e / 0.5) * k
                    d.ellipse([px - rr, py - rr, px + rr, py + rr], outline=lc + (int(180 * (1 - e / 0.5)),), width=int(2 * k))
                d.ellipse([px - r, py - r, px + r, py + r], fill=lc + (255,))
                d.ellipse([px - r * 0.45, py - r * 0.45, px + r * 0.45, py + r * 0.45], fill=(255, 255, 255, 220))
            else:
                r = 5 * k
                d.ellipse([px - r, py - r, px + r, py + r], fill=tuple(int(c * 0.35) for c in base) + (255,))
        # Aufprall-Glimmen am Ring
        import bisect
        lo = bisect.bisect_left(self.hit_t, t - 0.35)
        hi = bisect.bisect_right(self.hit_t, t)
        for j in range(max(lo, hi - 30), hi):
            t0, ev = self.hits[j]
            if ev["c"] != C.k:
                continue
            e = (t - t0) / 0.35
            a = math.degrees(ev["ang"])
            w = 6 + 8 * ev["strength"]
            f = (1 - e) * (0.4 + 0.6 * ev["strength"])
            hc = tuple(int(c + (255 - c) * f) for c in col)  # Ringfarbe -> Weiß gemischt (deckend)
            d.arc(box, a - w, a + w, fill=hc + (255,), width=int(13 * k))

    def draw_gate(self, d, kind, X, Y, R, f, col, k):
        h = EXIT_HALF
        mid = math.pi / 2
        if kind == "doors":  # zwei Schiebetüren gleiten außen am Ring entlang zur Seite
            for sgn in (-1, 1):
                a_mid = mid + sgn * (h / 2 + h * f * 0.95)
                Ro = R + 14 * k * min(1.0, f * 2)
                d.arc([X - Ro, Y - Ro, X + Ro, Y + Ro], math.degrees(a_mid - h / 2), math.degrees(a_mid + h / 2),
                      fill=col + (255,), width=int(11 * k))
        elif kind == "petals":  # drei Blütenblätter klappen nach außen auf
            n = 3
            for j in range(n):
                a0 = mid - h + 2 * h * j / n
                a1 = mid - h + 2 * h * (j + 1) / n
                hx, hy = X + math.cos(a0) * R, Y + math.sin(a0) * R
                ex, ey = X + math.cos(a1) * R, Y + math.sin(a1) * R
                rot = f * math.radians(70)
                vx, vy = ex - hx, ey - hy
                rx = vx * math.cos(rot) - vy * math.sin(rot)
                ry = vx * math.sin(rot) + vy * math.cos(rot)
                d.line([hx, hy, hx + rx, hy + ry], fill=col + (255,), width=int(11 * k))
                rr = 5 * k
                d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], fill=(255, 255, 255, 230))
        elif kind == "trapdoor":  # Falltür, links angeschlagen, klappt nach unten
            a0, a1 = mid - h, mid + h
            hx, hy = X + math.cos(a1) * R, Y + math.sin(a1) * R
            ex, ey = X + math.cos(a0) * R, Y + math.sin(a0) * R
            rot = -f * math.radians(95)
            vx, vy = ex - hx, ey - hy
            rx = vx * math.cos(rot) - vy * math.sin(rot)
            ry = vx * math.sin(rot) + vy * math.cos(rot)
            d.line([hx, hy, hx + rx, hy + ry], fill=col + (255,), width=int(12 * k))
            rr = 7 * k
            d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], fill=(255, 255, 255, 240))
        elif kind == "iris":  # sechs Klingen gleiten nach außen und drehen weg
            n = 6
            for j in range(n):
                a = mid - h + 2 * h * (j + 0.5) / n + f * (j - 2.5) * 0.12
                Ro = R + 50 * k * f
                l = 2 * h / n * R * 0.45
                cx, cy = X + math.cos(a) * Ro, Y + math.sin(a) * Ro
                tx, ty = -math.sin(a + f * 1.2), math.cos(a + f * 1.2)
                d.line([cx - tx * l, cy - ty * l, cx + tx * l, cy + ty * l], fill=col + (int(255 * (1 - 0.5 * f)),),
                       width=int(9 * k))


# ------------------------------------------------------------------ Klang

SR = 44100
CHORDS = [(261.63, 329.63, 392.0), (220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 246.94, 293.66),
          (261.63, 329.63, 392.0)]
_cache = {}


def celesta(f):
    key = round(f, 2)
    if key not in _cache:
        n = int(SR * 2.6)
        t = np.arange(n) / SR
        s = np.sin(2 * np.pi * f * t) * np.exp(-t * 2.2) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 5) \
            + 0.18 * np.sin(2 * np.pi * 4.01 * f * t) * np.exp(-t * 11)
        _cache[key] = s * np.minimum(t / 0.003, 1) * 0.42
    return _cache[key]


def bloop():
    n = int(SR * 0.3)
    t = np.arange(n) / SR
    f = 300 + 500 * (1 - np.exp(-t * 30))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 12) * 0.4


def soft_whoosh(rng):
    n = int(SR * 1.0)
    t = np.linspace(0, 1, n)
    x = np.convolve(rng.standard_normal(n), np.ones(30) / 30, mode="same")
    return x * np.sin(np.pi * t) ** 2 * 0.5


def chord_note(c, n):
    ch = CHORDS[c]
    return ch[n % 3] * 2 ** (n // 3)


def pad(total, changes):
    n = int(SR * total)
    t = np.arange(n) / SR
    out = np.zeros(n)
    bounds = [0.0] + [tc for tc, _ in changes] + [total + 3]
    for idx in range(len(bounds) - 1):
        c = 0 if idx == 0 else changes[idx - 1][1]
        t0, t1 = bounds[idx], bounds[idx + 1]
        env = np.clip((t - t0 + 1.5) / 1.5, 0, 1) * np.clip((t1 + 1.5 - t) / 1.5, 0, 1)
        for j, f in enumerate(CHORDS[c]):
            out += env * (np.sin(2 * np.pi * f / 2 * t + j) + 0.3 * np.sin(2 * np.pi * f * t)) / (j + 2)
    out *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.09 * t)
    return out * 0.05 * np.minimum(1, t / 2)


def build_audio(sim, path):
    total = len(sim.frames) / FPS
    n = int(SR * total)
    dry = np.zeros(n + SR * 3)
    rng = random.Random(3)
    nrng = np.random.default_rng(4)

    def put(t, s, vol):
        i = int(t * SR)
        if 0 <= i < len(dry):
            j = min(len(dry), i + len(s))
            dry[i:j] += s[: j - i] * vol

    changes = []
    for t, kind, d in sim.events:
        if kind == "dot":
            c, m = d["c"], d["n"] - 1
            put(t, celesta(chord_note(c, m % 9)), 0.45 + 0.3 * d["strength"])
        elif kind == "tick":
            put(t, celesta(chord_note(min(d["c"], 4), rng.randrange(3, 9))), 0.12 + 0.15 * d["strength"])
        elif kind == "open":
            nc = min(d["c"] + 1, len(CHORDS) - 1)
            for j in range(3):
                put(t + 0.35 + j * 0.08, celesta(CHORDS[nc][j] * 2), 0.4)
            put(t + 0.3, soft_whoosh(nrng), 0.5)
            changes.append((t + 0.35, nc))
        elif kind == "enter":
            put(t, bloop(), 0.35)
        elif kind == "calm":
            for j, f in enumerate((261.63, 329.63, 392.0, 523.25, 659.25)):
                put(t + j * 0.15, celesta(f * 2), 0.35)
    dry = dry[:n]
    wet = reverb(dry, 2.8)
    mix = dry * 0.75 + wet * 0.5 + pad(total, changes)
    mix *= np.clip((total - np.arange(n) / SR) / 1.2, 0, 1)
    peak = np.max(np.abs(mix)) or 1
    mix = np.tanh(mix / peak * 1.1) * 0.9
    st = np.stack([mix, mix], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ Ausgabe

_R = None


def _chunk(args):
    i0, i1, out = args
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W1}x{H1}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-maxrate", "14M", "-bufsize", "28M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_R.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(sim, spec, out_path, preview_dir=None, preview_times=(), jobs=4):
    global _R
    _R = OrbitRenderer(sim)
    n = len(sim.frames)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fr = _R.frame(min(int(pt * FPS), n - 1))
            base = os.path.join(preview_dir, f"{spec['id']}_vertical_{pt:05.1f}")
            fr.save(base + ".png")
            tiktok_ui_mock(fr, "vertical").save(base + "_ui.png")
        return
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
        build_audio(sim, wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path], check=True)
