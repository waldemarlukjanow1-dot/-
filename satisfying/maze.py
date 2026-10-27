"""Parcours-Videos: Bälle müssen sich durch Ebenen mit Schlitzen, Toren und Hindernissen nach unten arbeiten.

Kein Text, keine Schweife. Jede Ebene hat einen eigenen Mechanismus. Wer eine Öffnung findet, fällt eine Ebene
tiefer (Ton steigt mit jeder Ebene). Unten wartet eine goldene Schale, in der alle Bälle zur Ruhe kommen.
Welt-Koordinaten: x in [-400, 400], y von 0 (oben) bis 1150 (unten), y nach unten.
"""
import math
import os
import random
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tiktok.render import tiktok_ui_mock
from .hypno import ease_out, mix
from .sim import DT, FPS, SUB, Sim, hue_rgb

HALF = 400.0
FLOORS = [190.0, 380.0, 570.0, 760.0, 950.0]
BOTTOM = 1150.0
G = 1400.0
PEGS = [(x, 445.0) for x in (-300, -150, 0, 150, 300)] + [(x, 510.0) for x in (-225, -75, 75, 225)]
BAR = dict(c=(0.0, 665.0), half=82.0, om=1.4)
GOAL = [((-HALF, 1010.0), (-40.0, 1138.0)), ((-40.0, 1138.0), (40.0, 1138.0)), ((40.0, 1138.0), (HALF, 1010.0))]
BALL_COLORS = [(255, 70, 200), (60, 220, 255), (255, 220, 60)]
LEVEL_HUES = [0.78, 0.66, 0.54, 0.42, 0.30]


def gaps(level, t, k=1.0):
    """Offene Bereiche (x0, x1) im Boden einer Ebene zur Zeit t. k skaliert die Breiten."""
    if level == 0:  # wandernder Schlitz
        c = 270 * math.sin(0.55 * t + 0.6)
        w = 42 * k
        return [(c - w, c + w)]
    if level == 1:  # drei Tore, die nacheinander kurz öffnen
        out = []
        for j, c in enumerate((-260, 0, 260)):
            if ((t * 0.5 + j / 3) % 1) < 0.30:
                out.append((c - 34 * k, c + 34 * k))
        return out
    if level == 2:  # zwei schmale Randschlitze unter dem Stiftfeld
        w = 30 * k
        return [(-280 - w, -280 + w), (280 - w, 280 + w)]
    if level == 3:  # enger Schlitz direkt unter dem rotierenden Balken
        w = 34 * k
        return [(-w, w)]
    if level == 4:  # Iris: öffnet und schließt rhythmisch
        w = 52 * k * max(0.0, math.sin(0.9 * t))
        return [(-w, w)]
    return []


def gate_slabs(t):
    """Geschlossene Tore der Ebene 1 (für die Darstellung)."""
    out = []
    for j, c in enumerate((-260, 0, 260)):
        ph = (t * 0.5 + j / 3) % 1
        out.append((c, ph < 0.30, ph))
    return out


def bar_segment(t):
    cx, cy = BAR["c"]
    a = BAR["om"] * t
    dx, dy = math.cos(a) * BAR["half"], math.sin(a) * BAR["half"]
    return (cx - dx, cy - dy), (cx + dx, cy + dy)


class MazeBall:
    def __init__(self, x, y, vx, vy, r, color):
        self.x, self.y, self.vx, self.vy, self.r, self.color = x, y, vx, vy, r, color
        self.level = 0
        self.last_note = -1.0


class MazeSim(Sim):
    def __init__(self, seed, n=3, r=16.0, gap_k=1.0):
        super().__init__(seed)
        rng = self.rng
        self.r, self.gap_k = r, gap_k
        self.balls = []
        for i in range(n):
            self.balls.append(MazeBall(-200 + 200 * i + rng.uniform(-40, 40), 40 + rng.uniform(0, 40),
                                       rng.choice((-1, 1)) * rng.uniform(200, 400), rng.uniform(-100, 100), r,
                                       BALL_COLORS[i % len(BALL_COLORS)]))
        self.fx = []  # (t, art, x, y, farbe, stärke)
        self.settle_t = None

    def bounce_note(self, b, strength, kind="drop"):
        if self.t - b.last_note > 0.06 and strength > 0.08:
            b.last_note = self.t
            self.ev(kind, strength=strength, level=b.level)

    def fxa(self, kind, x, y, col, st=1.0):
        self.fx.append((self.t, kind, x, y, col, st))

    def seg_collide(self, b, p, q, vel_at=None, rest=1.0):
        (x0, y0), (x1, y1) = p, q
        ex, ey = x1 - x0, y1 - y0
        L2 = ex * ex + ey * ey
        u = max(0.0, min(1.0, ((b.x - x0) * ex + (b.y - y0) * ey) / L2))
        cx, cy = x0 + ex * u, y0 + ey * u
        dx, dy = b.x - cx, b.y - cy
        d = math.hypot(dx, dy)
        if d >= b.r or d == 0:
            return 0.0
        nx, ny = dx / d, dy / d
        b.x, b.y = cx + nx * b.r, cy + ny * b.r
        svx, svy = vel_at(cx, cy) if vel_at else (0.0, 0.0)
        rvx, rvy = b.vx - svx, b.vy - svy
        vn = rvx * nx + rvy * ny
        if vn >= 0:
            return 0.0
        rvx -= (1 + rest) * vn * nx
        rvy -= (1 + rest) * vn * ny
        b.vx, b.vy = rvx + svx, rvy + svy
        return -vn

    def step(self, dt):
        rng = self.rng
        t = self.t
        for b in self.balls:
            b.vy += G * dt
            b.x += b.vx * dt
            b.y += b.vy * dt
        # Kugel-Kugel
        for i in range(len(self.balls)):
            for j in range(i + 1, len(self.balls)):
                a, c = self.balls[i], self.balls[j]
                dx, dy = c.x - a.x, c.y - a.y
                d = math.hypot(dx, dy)
                if 0 < d < a.r + c.r:
                    nx, ny = dx / d, dy / d
                    rel = (a.vx - c.vx) * nx + (a.vy - c.vy) * ny
                    if rel > 0:
                        e = 0.5 if a.level == 5 else 1.0
                        k = (1 + e) / 2 * rel
                        a.vx -= k * nx
                        a.vy -= k * ny
                        c.vx += k * nx
                        c.vy += k * ny
                    ov = (a.r + c.r - d) / 2
                    a.x -= nx * ov
                    a.y -= ny * ov
                    c.x += nx * ov
                    c.y += ny * ov
        for i, b in enumerate(self.balls):
            L = b.level
            # Seitenwände
            if abs(b.x) + b.r > HALF:
                b.x = math.copysign(HALF - b.r, b.x)
                if b.vx * b.x > 0:
                    b.vx = -b.vx * (0.6 if L == 5 else 1.0)
                    if L < 5:
                        self.bounce_note(b, min(1, abs(b.vx) / 600), "tick")
            # Decke der aktuellen Ebene (von unten immer geschlossen)
            ceil = 0.0 if L == 0 else FLOORS[L - 1]
            if b.y - b.r < ceil:
                b.y = ceil + b.r
                if b.vy < 0:
                    b.vy = abs(b.vy) * 0.9
            if L == 5:
                for p, q in GOAL:
                    vn = self.seg_collide(b, p, q, rest=0.45)
                    if vn > 0:
                        b.vx *= 0.985
                        if vn > 120:
                            self.bounce_note(b, min(1, vn / 900))
                            self.fxa("ripple", b.x, b.y + b.r, b.color, min(1, vn / 900))
                b.vx *= 1 - 0.4 * dt
                continue
            # Hindernisse
            if L == 2:
                for px, py in PEGS:
                    dx, dy = b.x - px, b.y - py
                    d = math.hypot(dx, dy)
                    pr = 9.0
                    if 0 < d < b.r + pr:
                        nx, ny = dx / d, dy / d
                        b.x, b.y = px + nx * (b.r + pr), py + ny * (b.r + pr)
                        vn = b.vx * nx + b.vy * ny
                        if vn < 0:
                            b.vx -= 2 * vn * nx
                            b.vy -= 2 * vn * ny
                            a = rng.uniform(-0.35, 0.35)
                            cs, sn = math.cos(a), math.sin(a)
                            b.vx, b.vy = b.vx * cs - b.vy * sn, b.vx * sn + b.vy * cs
                            self.bounce_note(b, min(1, -vn / 700), "peg")
                            self.fxa("peg", px, py, b.color, min(1, -vn / 700))
            if L == 3:
                p, q = bar_segment(t)
                cx, cy = BAR["c"]
                om = BAR["om"]
                vn = self.seg_collide(b, p, q, vel_at=lambda x, y: (-om * (y - cy), om * (x - cx)))
                if vn > 0:
                    sp = math.hypot(b.vx, b.vy)
                    if sp > 900:
                        b.vx, b.vy = b.vx / sp * 900, b.vy / sp * 900
                    self.bounce_note(b, min(1, vn / 700), "tick")
                    self.fxa("spark", b.x, b.y, b.color, min(1, vn / 700))
            # Boden mit Öffnungen
            yF = FLOORS[L]
            if b.y + b.r > yF and b.y < yF + b.r:
                open_ = any(x0 + b.r * 0.55 < b.x < x1 - b.r * 0.55 for x0, x1 in gaps(L, t, self.gap_k))
                if b.y > yF or open_:
                    if b.y > yF:  # durch!
                        b.level += 1
                        b.vy = min(b.vy, 350.0)
                        self.ev("pass", ball=i, level=b.level)
                        self.fxa("pass", b.x, yF, b.color)
                        if b.level == 5:
                            self.ev("goal", ball=i)
                else:
                    b.y = yF - b.r
                    if b.vy > 0:
                        st = min(1, b.vy / 900)
                        b.vy = -rng.uniform(420, 660)  # zufällige Sprunghöhe
                        b.vx = b.vx * rng.uniform(0.7, 1.2) + rng.uniform(-200, 200)
                        b.vx = math.copysign(min(460, max(130, abs(b.vx))), b.vx)
                        self.bounce_note(b, st)
                        self.fxa("ripple", b.x, yF, b.color, st)
        if self.finish_t is None and all(b.level == 5 for b in self.balls):
            if all(math.hypot(b.vx, b.vy) < 25 for b in self.balls):
                if self.settle_t is None:
                    self.settle_t = self.t
                if self.t - self.settle_t > 0.6:
                    self.finish_t = self.t
            else:
                self.settle_t = None

    def run(self, max_t=60.0):
        self.outro = 1.5
        while True:
            for _ in range(SUB):
                self.step(DT)
                self.t += DT
            self.frames.append(dict(t=self.t, balls=[(b.x, b.y, b.r, b.color, b.level) for b in self.balls]))
            if self.finish_t is not None and self.t >= self.finish_t + self.outro:
                break
            if self.t > max_t:
                break
        return self

    def pass_times(self):
        return [t for t, k, _ in self.events if k == "pass"]


# ------------------------------------------------------------------ Klang (sanft)

SR = 44100
_PENTA = [261.63 * 2 ** (s / 12) for s in (0, 2, 4, 7, 9, 12, 14, 16, 19, 21)]  # C-Dur pentatonisch
_cache = {}


def drop(f):
    """Wassertropfen: kurzer Sinus mit Tonhöhen-Gleiten nach oben."""
    key = ("drop", f)
    if key not in _cache:
        n = int(SR * 0.35)
        t = np.arange(n) / SR
        freq = f * (0.75 + 0.25 * (1 - np.exp(-t * 40)))
        s = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 14) * np.minimum(t / 0.003, 1)
        _cache[key] = s * 0.5
    return _cache[key]


def rhodes(f, dur=2.4):
    """Weiches E-Piano: Grundton + Glockenoberton + leichtes Tremolo."""
    key = ("rhodes", f)
    if key not in _cache:
        n = int(SR * dur)
        t = np.arange(n) / SR
        s = np.sin(2 * np.pi * f * t + 0.6 * np.exp(-t * 4) * np.sin(2 * np.pi * f * t))
        s += 0.18 * np.sin(2 * np.pi * f * 4.0 * t) * np.exp(-t * 7)
        s *= np.exp(-t * 1.3) * (1 + 0.12 * np.sin(2 * np.pi * 4.5 * t)) * np.minimum(t / 0.004, 1)
        _cache[key] = s * 0.45
    return _cache[key]


def tink(f):
    key = ("tink", f)
    if key not in _cache:
        n = int(SR * 0.25)
        t = np.arange(n) / SR
        _cache[key] = np.sin(2 * np.pi * f * t) * np.exp(-t * 22) * 0.3
    return _cache[key]


def warm_pad(total, fade_out_at):
    n = int(SR * total)
    t = np.arange(n) / SR
    chord = (65.41, 98.0, 130.81, 164.81, 196.0)  # C2 G2 C3 E3 G3
    s = sum(np.sin(2 * np.pi * f * t + 0.8 * np.sin(2 * np.pi * 0.05 * (k + 1) * t)) / (k + 1.5)
            for k, f in enumerate(chord))
    s *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.08 * t)
    env = np.minimum(1, t / 2.0) * np.clip(1 - (t - fade_out_at) / 4, 0.25, 1)
    return s * env * 0.07


def build_audio(sim, path):
    total = len(sim.frames) / FPS
    rng = random.Random(sim.seed + 1)
    out = np.zeros(int(SR * (total + 3)))
    pos = 4

    def put(t, s, vol):
        i = int(t * SR)
        j = min(len(out), i + len(s))
        out[i:j] += s[: j - i] * vol

    for t, kind, d in sim.events:
        if kind == "drop":
            pos = min(len(_PENTA) - 1, max(0, pos + rng.choice((-2, -1, 1, 2))))
            put(t, drop(_PENTA[pos] * 2), 0.15 + 0.35 * d["strength"])
        elif kind == "tick":
            put(t, tink(_PENTA[pos] * 4), 0.4 * d["strength"])
        elif kind == "peg":
            put(t, tink(_PENTA[rng.randrange(5, 10)] * 2), 0.5 + 0.3 * d["strength"])
        elif kind == "pass":  # mit jeder Ebene höher: C E G A C
            lv = d["level"]
            f = _PENTA[min(len(_PENTA) - 1, lv + 2)]
            put(t, rhodes(f), 0.9)
            put(t + 0.09, rhodes(f * 1.5), 0.45)
        elif kind == "goal":
            for k, f in enumerate((261.63, 329.63, 392.0, 493.88)):  # Cmaj7, sanft gebrochen
                put(t + k * 0.11, rhodes(f), 0.55)
    out = out[: int(SR * total)]
    out += warm_pad(total, sim.finish_t or total)
    peak = np.max(np.abs(out)) or 1
    out = np.tanh(out / peak * 1.2) * 0.85
    st = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ Darstellung

# Parcours liegt in der Safe-Zone: rechts frei für die Buttons, unten für die Beschreibung.
LAYOUTS = {
    "vertical": dict(W=1080, H=1920, origin=(520, 250), scale=1.0),
    "landscape": dict(W=1920, H=1080, origin=(870, 55), scale=0.85),
}


class MazeScene:
    def __init__(self, sim, fmt):
        self.sim = sim
        L = LAYOUTS[fmt]
        self.W, self.H = L["W"], L["H"]
        self.ox, self.oy = L["origin"]
        self.s = L["scale"]
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        cy = self.oy + BOTTOM * self.s / 2
        d = np.sqrt(((xx - self.ox) / (HALF * self.s * 2.2)) ** 2 + ((yy - cy) / (BOTTOM * self.s * 1.1)) ** 2)
        self.grad = (np.clip(1 - d, 0, 1) ** 1.5)[..., None]
        self.passes = [(t, d) for t, k, d in sim.events if k == "pass"]
        self.fx_t = [f[0] for f in sim.fx]

    def P(self, x, y):
        return self.ox + x * self.s, self.oy + y * self.s

    def line(self, d, p, q, col, w):
        d.line([*self.P(*p), *self.P(*q)], fill=col, width=max(2, int(w * self.s)))

    def level_glow(self, level, t):
        """Ebene leuchtet kurz auf, wenn ein Ball durch ihren Boden fällt."""
        g = 0.0
        for t0, d in self.passes:
            if d["level"] - 1 == level and 0 <= t - t0 < 0.7:
                g = max(g, 1 - (t - t0) / 0.7)
        return g

    def draw(self, d, fi, t):
        sim = self.sim
        k = sim.gap_k
        wall = (140, 100, 255)
        # Außenrahmen
        self.line(d, (-HALF, 0), (-HALF, BOTTOM - 140), wall, 6)
        self.line(d, (HALF, 0), (HALF, BOTTOM - 140), wall, 6)
        self.line(d, (-HALF, 0), (HALF, 0), wall, 6)
        # Böden mit Öffnungen
        for L, yF in enumerate(FLOORS):
            g = self.level_glow(L, t)
            col = mix(hue_rgb(LEVEL_HUES[L], 0.9), 1.0, 0.5 * g)
            xs = [-HALF]
            for x0, x1 in sorted(gaps(L, t, k)):
                xs += [x0, x1]
            xs.append(HALF)
            for a, b in zip(xs[::2], xs[1::2]):
                if b - a > 1:
                    self.line(d, (a, yF), (b, yF), col, 9)
            for x0, x1 in gaps(L, t, k):  # Öffnungskanten leuchten
                for xe in (x0, x1):
                    X, Y = self.P(xe, yF)
                    r = (6 + 4 * g) * self.s
                    d.ellipse([X - r, Y - r, X + r, Y + r], fill=mix((255, 255, 255), 0.9))
            if L == 1:  # geschlossene Tore als eigene Platten
                for c, open_, ph in gate_slabs(t):
                    if not open_:
                        pulse = 0.55 + 0.45 * max(0.0, 1 - (1 - ph) / 0.12)  # glüht kurz vorm Öffnen auf
                        self.line(d, (c - 34 * k, yF), (c + 34 * k, yF), mix((255, 255, 255), pulse), 9)
        # Stiftfeld
        for i, (px, py) in enumerate(PEGS):
            hit = 0.0
            for f in self.recent(t, 0.3):
                if f[1] == "peg" and f[2] == px and f[3] == py:
                    hit = max(hit, 1 - (t - f[0]) / 0.3)
            X, Y = self.P(px, py)
            r = (9 + 4 * hit) * self.s
            d.ellipse([X - r, Y - r, X + r, Y + r], fill=mix(hue_rgb(0.5), 0.8, 0.7 * hit))
        # Rotierender Balken
        p, q = bar_segment(t)
        self.line(d, p, q, hue_rgb(0.42, 0.9), 10)
        X, Y = self.P(*BAR["c"])
        d.ellipse([X - 9, Y - 9, X + 9, Y + 9], fill=(255, 255, 255))
        # Ziel-Schale
        gold = (255, 200, 80)
        n_in = sum(1 for b in sim.frames[fi]["balls"] if b[4] == 5)
        gp = 0.6 + 0.15 * math.sin(t * 2) + 0.08 * n_in
        for a, b in GOAL:
            self.line(d, a, b, mix(gold, min(1.0, gp)), 10)
        # Effekte
        for f in self.recent(t, 0.6):
            t0, kind, x, y, col, st = f
            age = t - t0
            X, Y = self.P(x, y)
            if kind == "ripple" and age < 0.4:
                p_ = age / 0.4
                rx, ry = (8 + 50 * st * ease_out(p_)) * self.s, (3 + 12 * st * ease_out(p_)) * self.s
                d.ellipse([X - rx, Y - ry, X + rx, Y + ry], outline=mix(col, (1 - p_) * (0.4 + 0.6 * st)), width=2)
            elif kind == "pass":
                p_ = age / 0.6
                r = (10 + 70 * ease_out(p_)) * self.s
                d.ellipse([X - r, Y - r, X + r, Y + r], outline=mix(col, 1 - p_, 0.4), width=max(1, int(5 * (1 - p_))))
            elif kind == "spark" and age < 0.25:
                p_ = age / 0.25
                r = (6 + 30 * st * ease_out(p_)) * self.s
                d.ellipse([X - r, Y - r, X + r, Y + r], outline=mix(col, 1 - p_), width=2)
        # Bälle
        for x, y, r, col, lv in sim.frames[fi]["balls"]:
            X, Y = self.P(x, y)
            rr = r * self.s
            d.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=col)
            d.ellipse([X - rr * 0.5, Y - rr * 0.5, X + rr * 0.5, Y + rr * 0.5], fill=(255, 255, 255))

    def recent(self, t, life):
        import bisect
        lo = bisect.bisect_left(self.fx_t, t - life)
        hi = bisect.bisect_right(self.fx_t, t)
        return self.sim.fx[lo:hi]

    def frame(self, fi):
        t = fi / FPS
        geo = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        self.draw(ImageDraw.Draw(geo), fi, t)
        arr = np.asarray(geo, np.float32)
        g1 = geo.resize((self.W // 4, self.H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
        g2 = geo.resize((self.W // 10, self.H // 10), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
        glow = (np.asarray(g1.resize((self.W, self.H), Image.BILINEAR), np.float32) * 1.0 +
                np.asarray(g2.resize((self.W, self.H), Image.BILINEAR), np.float32) * 1.1)
        bgc = np.array(hue_rgb(0.7 + 0.04 * math.sin(t * 0.2), 0.65, 0.18), np.float32)
        out = self.grad * bgc + arr + glow
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def render(sim, spec, fmt, out_path, preview_dir=None, preview_times=()):
    scene = MazeScene(sim, fmt)
    n = len(sim.frames)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fr = scene.frame(min(int(pt * FPS), n - 1))
            base = os.path.join(preview_dir, f"{spec['id']}_{fmt}_{pt:05.1f}")
            fr.save(base + ".png")
            tiktok_ui_mock(fr, fmt).save(base + "_ui.png")
        return
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        build_audio(sim, wav)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{scene.W}x{scene.H}",
               "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "21",
               "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
               "-shortest", "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fi in range(n):
            proc.stdin.write(scene.frame(fi).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg fehlgeschlagen")
