"""Minimalistische Viral-Formate: schwarzer Hintergrund, klare Linien, jeder Aufprall = nächster Melodieton.

spiral: viele verschachtelte Ringe (Weiß -> Rot), Lücken spiralförmig versetzt, drehen gemeinsam. Ein kleiner
        Ball bricht Ring für Ring nach außen. Der äußerste Ring ist geschlossen: dort kommt der Ball zur Ruhe.
strings: Ball im Kreis, jeder Berührungspunkt bleibt per Linie mit dem Ball verbunden.
"""
import math
import os
import random
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb

FPS, SUB = 60, 8
DT = 1.0 / (FPS * SUB)
W1, H1, SS = 1080, 1920, 2
CX, CY = 505, 900          # Mitte des Systems (rechts bleibt Platz für die TikTok-Buttons)
ARENA = 400.0              # Außenradius in Pixeln


def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def angdiff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


# ------------------------------------------------------------------ Simulationen

class BaseSim:
    def __init__(self, seed, slow_at, outro=1.6):
        self.seed, self.rng = seed, random.Random(seed)
        self.t, self.events, self.frames = 0.0, [], []
        self.slow_at, self.outro = slow_at, outro
        self.calm_t, self.finish_t = None, None

    def ev(self, kind, **d):
        self.events.append((self.t, kind, d))

    def calm(self):
        if self.calm_t is None:
            return 0.0
        return min(1.0, max(0.0, (self.t - self.calm_t) / 3.5))

    def wall(self, R, gap=None, half=0.0, jitter=0.2, speed=(520, 800), g=1000.0):
        """Ballphysik in einem Kreis mit optionaler Lücke. Rückgabe: 'bounce'(stärke) / 'escape' / None."""
        b = self.ball
        d = math.hypot(b["x"], b["y"]) or 1e-6
        if gap is not None and angdiff(math.atan2(b["y"], b["x"]), gap) < half:
            return ("escape", 0) if d > R else (None, 0)
        if d + b["r"] <= R:
            return (None, 0)
        nx, ny = b["x"] / d, b["y"] / d
        b["x"], b["y"] = nx * (R - b["r"]), ny * (R - b["r"])
        vn = b["vx"] * nx + b["vy"] * ny
        if vn <= 0:
            return (None, 0)
        c = self.calm()
        if c > 0:  # Ausklang: unelastisch
            e = 1 - 0.7 * c
            b["vx"] -= (1 + e) * vn * nx
            b["vy"] -= (1 + e) * vn * ny
        else:
            b["vx"] -= 2 * vn * nx
            b["vy"] -= 2 * vn * ny
            a = self.rng.uniform(-jitter, jitter)
            cs, sn = math.cos(a), math.sin(a)
            vx, vy = b["vx"] * cs - b["vy"] * sn, b["vx"] * sn + b["vy"] * cs
            if vx * nx + vy * ny > -0.3 * math.hypot(vx, vy):  # muss nach innen zeigen
                vx, vy = b["vx"], b["vy"]
            sp = math.hypot(vx, vy) or 1
            tgt = min(max(sp, speed[0]), speed[1])
            b["vx"], b["vy"] = vx / sp * tgt, vy / sp * tgt
        return ("bounce", min(1.0, vn / 900))

    def move(self, g):
        b = self.ball
        b["vy"] += g * DT
        c = self.calm()
        if c:
            f = 1 - (0.3 + 1.8 * c) * DT
            b["vx"] *= f
            b["vy"] *= f
        b["x"] += b["vx"] * DT
        b["y"] += b["vy"] * DT

    def run(self, max_t=70.0):
        while True:
            for _ in range(SUB):
                self.step()
                self.t += DT
            self.frames.append(self.snapshot())
            if self.finish_t is not None and self.t >= self.finish_t + self.outro:
                break
            if self.t > max_t:
                break
        return self

    def check_rest(self):
        b = self.ball
        if self.calm_t is not None and self.finish_t is None and self.t > self.calm_t + 3:
            if math.hypot(b["vx"], b["vy"]) < 15:
                self.finish_t = self.t


class SpiralSim(BaseSim):
    def __init__(self, seed, n=30, r_in=58.0, half_deg=26.0, twist=0.21, omega=0.9, slow_at=None):
        super().__init__(seed, slow_at)
        rng = self.rng
        self.n = n
        self.radii = [r_in + (ARENA - r_in) * k / (n - 1) for k in range(n)]
        self.half = math.radians(half_deg)
        self.twist, self.omega = twist, omega
        self.th0 = rng.uniform(0, 2 * math.pi)
        self.k = 0
        self.broken = []  # (t, ring)
        a = rng.uniform(0, 2 * math.pi)
        self.ball = dict(x=0.0, y=-10.0, vx=math.cos(a) * 560, vy=math.sin(a) * 560, r=7.0)

    def gap(self, k, t):
        return self.th0 + self.omega * t + k * self.twist

    def step(self):
        self.move(1000.0)
        last = self.n - 1  # äußerster Ring bleibt geschlossen
        while True:
            k = self.k
            gap = self.gap(k, self.t) if k < last else None
            res, st = self.wall(self.radii[k], gap, self.half, speed=(540, 820))
            if res == "escape":
                self.broken.append((self.t, k))
                self.ev("break", ring=k)
                self.k += 1
                if self.k == last:
                    self.calm_t = self.t + 0.8
                continue
            if res == "bounce":
                self.ev("note", strength=st)
            break
        self.check_rest()

    def snapshot(self):
        b = self.ball
        return (b["x"], b["y"], self.k)


class StringSim(BaseSim):
    def __init__(self, seed, slow_at=44.0, r=24.0, speed=(600, 880)):
        super().__init__(seed, slow_at)
        rng = self.rng
        a = rng.uniform(0, 2 * math.pi)
        self.ball = dict(x=rng.uniform(-60, 60), y=-150.0, vx=math.cos(a) * 650, vy=math.sin(a) * 650, r=r)
        self.speed = speed
        self.points = []

    def step(self):
        if self.calm_t is None and self.t >= self.slow_at:
            self.calm_t = self.t
        self.move(1000.0)
        res, st = self.wall(ARENA, speed=self.speed, jitter=0.25)
        if res == "bounce":
            b = self.ball
            d = math.hypot(b["x"], b["y"]) or 1
            if st > 0.08:
                self.points.append((b["x"] / d * ARENA, b["y"] / d * ARENA))
                self.ev("note", strength=st)
        self.check_rest()

    def snapshot(self):
        b = self.ball
        return (b["x"], b["y"], len(self.points))


# ------------------------------------------------------------------ Darstellung

import colorsys as _cs
HUE_LUT = [tuple(int(v * 255) for v in _cs.hsv_to_rgb(i / 256, 0.65, 1.0)) for i in range(256)]


def lerp(c0, c1, f):
    return tuple(int(a + (b - a) * f) for a, b in zip(c0, c1))


class MinimalRenderer:
    def __init__(self, sim, kind):
        self.sim, self.kind = sim, kind
        self.meows = []
        self.cat_meows = sorted((t, dd["cat"], dd["dur"]) for t, kk, dd in sim.events if kk == "meow")
        self.cat_t = [m[0] for m in self.cat_meows]
        if kind == "meow":
            for t, kk, dd in sim.events:
                if kk == "note":
                    self.meows.append((t, 0.42 * (1 / dd["p"]) ** 1.05))

    def open_cats(self, t):
        import bisect
        lo = bisect.bisect_left(self.cat_t, t - 3.0)
        hi = bisect.bisect_right(self.cat_t, t)
        out = {}
        for t0, cat, dur in self.cat_meows[lo:hi]:
            if t <= t0 + dur:
                u = (t - t0) / dur
                out[cat] = max(out.get(cat, 0.0), math.sin(math.pi * min(1.0, u * 1.15)) ** 0.6)
        return out

    def draw_cat(self, d, c, r, t, fur=(255, 160, 70), open_=None):
        """Katzengesicht als Ball; das Maul öffnet sich, solange das Miau klingt."""
        X, Y = c
        dark, pinkn = (40, 25, 20), (255, 140, 170)
        if sum(fur) < 250:
            dark = (230, 220, 120)  # schwarze Katze: helle Augen
        for sgn in (-1, 1):  # Ohren
            d.polygon([(X + sgn * r * 0.95, Y - r * 0.25), (X + sgn * r * 0.75, Y - r * 1.35), (X + sgn * r * 0.2, Y - r * 0.85)], fill=fur)
            d.polygon([(X + sgn * r * 0.78, Y - r * 0.45), (X + sgn * r * 0.68, Y - r * 1.05), (X + sgn * r * 0.36, Y - r * 0.8)], fill=pinkn)
        d.ellipse([X - r, Y - r, X + r, Y + r], fill=fur)
        if open_ is None:
            open_ = 0.0
        for t0, dur in ([] if open_ else self.meows):
            if t0 <= t <= t0 + dur:
                u = (t - t0) / dur
                open_ = max(open_, math.sin(math.pi * min(1.0, u * 1.15)) ** 0.6)
        for sgn in (-1, 1):  # Augen (beim Miauen zusammengekniffen)
            ex, ey = X + sgn * r * 0.38, Y - r * 0.12
            h = r * (0.22 - 0.16 * open_)
            d.ellipse([ex - r * 0.16, ey - h, ex + r * 0.16, ey + h], fill=dark)
            if open_ < 0.5:
                d.ellipse([ex - r * 0.05, ey - h * 0.6, ex + r * 0.04, ey - h * 0.2], fill=(255, 255, 255))
        d.polygon([(X - r * 0.1, Y + r * 0.12), (X + r * 0.1, Y + r * 0.12), (X, Y + r * 0.24)], fill=pinkn)
        if open_ > 0.05:  # Maul
            mh = r * 0.38 * open_
            d.ellipse([X - r * 0.2, Y + r * 0.28, X + r * 0.2, Y + r * 0.28 + mh], fill=(120, 30, 40))
        else:
            d.line([X - r * 0.18, Y + r * 0.34, X, Y + r * 0.26, X + r * 0.18, Y + r * 0.34], fill=dark, width=max(2, int(r * 0.06)))
        for sgn in (-1, 1):  # Schnurrhaare
            for dy in (-0.05, 0.08):
                d.line([X + sgn * r * 0.3, Y + r * (0.22 + dy), X + sgn * r * 1.15, Y + r * (0.15 + dy * 2.2)], fill=(255, 235, 220),
                       width=max(1, int(r * 0.035)))

    def frame(self, i):
        sim = self.sim
        t = (i + 1) / FPS
        img = Image.new("RGB", (W1 * SS, H1 * SS), (0, 0, 0))
        d = ImageDraw.Draw(img)
        k = SS

        def P(x, y):
            return (CX * k + x * k, CY * k + y * k)

        fr = sim.frames[i]
        x, y, state = fr[0], fr[1], fr[2]
        if self.kind in ("spiral", "multiply"):
            n = sim.n
            for ring in range(n):
                R = sim.radii[ring] * k
                f = ring / (n - 1)
                col = lerp((255, 255, 255), (205, 70, 45), f ** 1.3)
                w = max(2, int((sim.radii[1] - sim.radii[0]) * 0.6 * k))
                if ring < state:  # gebrochen: kurzes, sanftes Aufploppen
                    tb = next(tt for tt, r in sim.broken if r == ring)
                    e = (t - tb) / 0.3
                    if e >= 1:
                        continue
                    Rr = R + 18 * k * ease(e)
                    c2 = lerp(col, (0, 0, 0), ease(e))
                    g = sim.gap(ring, tb)
                    a0, a1 = math.degrees(g + sim.half), math.degrees(g - sim.half) + 360
                    d.arc([CX * k - Rr, CY * k - Rr, CX * k + Rr, CY * k + Rr], a0, a1, fill=c2, width=max(1, int(w * (1 - e))))
                    continue
                if ring == n - 1:
                    d.ellipse([CX * k - R, CY * k - R, CX * k + R, CY * k + R], outline=col, width=w)
                    continue
                g = sim.gap(ring, t)
                a0, a1 = math.degrees(g + sim.half), math.degrees(g - sim.half) + 360
                d.arc([CX * k - R, CY * k - R, CX * k + R, CY * k + R], a0, a1, fill=col, width=w)
            if self.kind == "multiply" and fr[3] is not None:
                xs, ys, hs = fr[3]
                r = (sim.rb + 0.5) * k
                for bx, by, h in zip(xs, ys, hs):
                    X, Y = P(float(bx), float(by))
                    d.ellipse([X - r, Y - r, X + r, Y + r], fill=HUE_LUT[int(h * 255) % 256])
            else:
                r = (sim.ball["r"] + 2) * k  # optisch etwas größer, damit er am Handy gut sichtbar ist
                X, Y = P(x, y)
                d.ellipse([X - r, Y - r, X + r, Y + r], fill=(110, 205, 240))
        elif self.kind == "catchaos":
            ring = (255, 175, 205)
            R = (ARENA + 4) * k
            d.ellipse([CX * k - R, CY * k - R, CX * k + R, CY * k + R], outline=ring, width=int(7 * k))
            xs, ys, hs = fr[3]
            t = (i + 1) / FPS
            open_cats = self.open_cats(t)
            for j, (bx, by, h) in enumerate(zip(xs, ys, hs)):
                self.draw_cat(d, P(float(bx), float(by)), sim.rb * k, t, fur=FURS[int(h) % len(FURS)],
                              open_=open_cats.get(j, 0.0))
        elif self.kind == "meow":
            ring = (255, 175, 205)
            R = (ARENA + 4) * k
            d.ellipse([CX * k - R, CY * k - R, CX * k + R, CY * k + R], outline=ring, width=int(7 * k))
            self.draw_cat(d, P(x, y), sim.ball["r"] * k, (i + 1) / FPS)
        else:
            pink = (232, 150, 255)
            R = (ARENA + 3) * k
            d.ellipse([CX * k - R, CY * k - R, CX * k + R, CY * k + R], outline=pink, width=int(5 * k))
            X, Y = P(x, y)
            for px, py in sim.points[:state]:
                d.line([P(px, py), (X, Y)], fill=pink, width=int(2.5 * k))
            r = sim.ball["r"] * k
            d.ellipse([X - r, Y - r, X + r, Y + r], fill=(245, 175, 255))
        return img.reduce(SS)


# ------------------------------------------------------------------ Klang: eigene Melodie, ein Ton pro Aufprall

SR = 44100
A4 = 440.0


def hz(n):  # Halbtöne relativ zu A4
    return A4 * 2 ** (n / 12)


# Originalmelodie (a-Moll), Akkordfolge Am - F - C - G, je 8 Töne
MELODY = [0, 3, 7, 12, 10, 7, 5, 7,
          -4, 0, 3, 8, 7, 3, 0, 3,
          -9, -5, 3, 7, 5, 3, 2, 3,
          -2, 2, 5, 10, 9, 7, 5, 2,
          0, 3, 7, 12, 15, 12, 10, 7,
          -4, 0, 3, 8, 12, 10, 8, 7,
          -9, 3, 7, 10, 8, 7, 5, 3,
          -2, 2, 5, 7, 10, 9, 7, 11]
BASS = [-24, -28, -21, -26]
_cache = {}


def piano(f, bright=1.0):
    key = (round(f, 2), bright)
    if key not in _cache:
        n = int(SR * 2.2)
        t = np.arange(n) / SR
        s = sum(np.sin(2 * np.pi * f * h * t * (1 + 0.0003 * h)) * (bright / h ** 1.4) * np.exp(-t * (1.2 + 0.8 * h))
                for h in range(1, 8))
        s = s * np.minimum(t / 0.004, 1) * 0.35
        _cache[key] = s
    return _cache[key]


def bass(f):
    key = ("b", round(f, 2))
    if key not in _cache:
        n = int(SR * 3.0)
        t = np.arange(n) / SR
        s = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-t * 0.9) * np.minimum(t / 0.01, 1)
        _cache[key] = s * 0.35
    return _cache[key]


def tink(f):
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 9) * 0.25


def build_audio(sim, path, octave=12):
    total = len(sim.frames) / FPS
    n = int(SR * total)
    out = np.zeros(n + SR * 3)

    def put(t, s, vol):
        i = int(t * SR)
        if 0 <= i < len(out):
            j = min(len(out), i + len(s))
            out[i:j] += s[: j - i] * vol

    m = 0
    for t, kind, d in sim.events:
        if kind == "note":
            note = MELODY[m % len(MELODY)] + octave - 12
            put(t, piano(hz(note)), 0.35 + 0.65 * d["strength"])
            if m % 8 == 0:
                put(t, bass(hz(BASS[(m // 8) % 4])), 0.6)
            m += 1
        elif kind == "break":
            note = MELODY[m % len(MELODY)] + octave
            put(t, tink(hz(note)), 0.8)
    out = out[:n]
    wet = reverb(out, 2.2)
    mix = out * 0.85 + wet * 0.35
    mix *= np.clip((total - np.arange(n) / SR) / 1.2, 0, 1)
    peak = np.max(np.abs(mix)) or 1
    mix = np.tanh(mix / peak * 1.15) * 0.9
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
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_R.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(sim, spec, out_path, preview_dir=None, preview_times=(), jobs=4):
    global _R
    _R = MinimalRenderer(sim, spec["kind"])
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
        if spec["kind"] == "meow":
            build_meow_audio(sim, wav)
        elif spec["kind"] == "catchaos":
            build_cat_chaos_audio(sim, wav)
        else:
            build_audio(sim, wav, octave=spec.get("octave", 12))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path], check=True)


# ------------------------------------------------------------------ Spirale + Vermehrung

class MultiplySpiralSim(BaseSim):
    """Phase 1: ein Ball bricht durch dicke Spiral-Ringe. Phase 2: im äußeren Kreis (verschwindet nie) erzeugt
    jeder Treffer einen neuen Ball – bis der Kreis voller Bälle ist. Danach sinken alle zu einem Haufen."""

    def __init__(self, seed, n=15, r_in=80.0, half_deg=34.0, twist=0.22, omega=1.25, p_spawn=0.45, cap=1000,
                 r_ball=7.0, crazy_hold=3.0):
        super().__init__(seed, None)
        rng = self.rng
        self.n = n
        self.radii = [r_in + (ARENA - r_in) * k / (n - 1) for k in range(n)]
        self.half = math.radians(half_deg)
        self.twist, self.omega = twist, omega
        self.th0 = rng.uniform(0, 2 * math.pi)
        self.k = 0
        self.broken = []
        self.p_spawn, self.cap, self.rb, self.crazy_hold = p_spawn, cap, r_ball, crazy_hold
        a = rng.uniform(0, 2 * math.pi)
        self.ball = dict(x=0.0, y=-10.0, vx=math.cos(a) * 560, vy=math.sin(a) * 560, r=r_ball)
        self.np_rng = np.random.default_rng(seed)
        self.X = self.Y = self.VX = self.VY = self.H = None  # Arrays ab Phase 2
        self.phase2_t = None
        self.cap_t = None
        self.last_note = -1.0
        self.spawned = 0

    def gap(self, k, t):
        return self.th0 + self.omega * t + k * self.twist

    def _note(self, strength, minint=0.045):
        if self.t - self.last_note >= minint:
            self.last_note = self.t
            self.ev("note", strength=strength)

    def step(self):
        if self.phase2_t is None:
            self.move(1000.0)
            last = self.n - 1
            while True:
                k = self.k
                res, st = self.wall(self.radii[k], self.gap(k, self.t), self.half, speed=(560, 840))
                if res == "escape":
                    self.broken.append((self.t, k))
                    self.ev("break", ring=k)
                    self.k += 1
                    if self.k == last:  # äußerer Kreis erreicht -> Phase 2
                        b = self.ball
                        self.phase2_t = self.t
                        self.X, self.Y = np.array([b["x"]]), np.array([b["y"]])
                        self.VX, self.VY = np.array([b["vx"]]), np.array([b["vy"]])
                        self.H = np.array([0.55])
                        self.ev("phase2")
                        return
                    continue
                if res == "bounce":
                    self._note(st, 0.0)
                break
            return
        self.step2()

    def step2(self):
        dt, R, rb = DT, ARENA, self.rb
        c = self.calm()
        self.VY += (1000.0 if c else 650.0) * dt  # im Chaos weniger Schwerkraft -> Bälle füllen den ganzen Kreis
        if c:
            f = 1 - (0.3 + 2.5 * c) * dt
            self.VX *= f
            self.VY *= f
        self.X += self.VX * dt
        self.Y += self.VY * dt
        d = np.hypot(self.X, self.Y) + 1e-9
        out = d + rb > R
        if not out.any():
            return
        idx = np.nonzero(out)[0]
        nx, ny = self.X[idx] / d[idx], self.Y[idx] / d[idx]
        self.X[idx], self.Y[idx] = nx * (R - rb), ny * (R - rb)
        vx, vy = self.VX[idx], self.VY[idx]
        vn = vx * nx + vy * ny
        hit = vn > 0
        if not hit.any():
            return
        if c:
            e = 1 - 0.75 * c
            vx = np.where(hit, vx - (1 + e) * vn * nx, vx)
            vy = np.where(hit, vy - (1 + e) * vn * ny, vy)
        else:
            vx = np.where(hit, vx - 2 * vn * nx, vx)
            vy = np.where(hit, vy - 2 * vn * ny, vy)
            a = self.np_rng.uniform(-0.25, 0.25, len(idx))
            vx, vy = vx * np.cos(a) - vy * np.sin(a), vx * np.sin(a) + vy * np.cos(a)
            sp = np.hypot(vx, vy) + 1e-9
            k = np.clip(sp, 520, 820) / sp
            vx, vy = vx * k, vy * k
        self.VX[idx], self.VY[idx] = vx, vy
        hi = idx[hit]
        self._note(float(min(1.0, vn[hit].max() / 900)))
        # Vermehrung: jeder Treffer kann einen neuen Ball erzeugen
        if c == 0 and len(self.X) < self.cap:
            m = self.np_rng.random(len(hi)) < self.p_spawn
            src = hi[m][: self.cap - len(self.X)]
            if len(src):
                ang = self.np_rng.uniform(0, 2 * np.pi, len(src))
                sp = self.np_rng.uniform(560, 800, len(src))
                bx = self.X[src] * 0.97
                by = self.Y[src] * 0.97
                self.X = np.concatenate([self.X, bx])
                self.Y = np.concatenate([self.Y, by])
                # neuer Ball fliegt nach innen weg
                inx, iny = -bx / (np.hypot(bx, by) + 1e-9), -by / (np.hypot(bx, by) + 1e-9)
                base = np.arctan2(iny, inx) + self.np_rng.uniform(-0.9, 0.9, len(src))
                self.VX = np.concatenate([self.VX, np.cos(base) * sp])
                self.VY = np.concatenate([self.VY, np.sin(base) * sp])
                h = (0.55 + (self.spawned + np.arange(len(src))) * 0.013) % 1
                self.spawned += len(src)
                self.H = np.concatenate([self.H, h])
                self.ev("spawn", n=len(self.X))
                if len(self.X) >= self.cap and self.cap_t is None:
                    self.cap_t = self.t
                    self.calm_t = self.t + self.crazy_hold

    def collide(self):
        """Kugel-Kugel-Stöße für viele Bälle (vektorisiert, einmal pro Frame, mehrere Durchläufe)."""
        if self.X is None or len(self.X) < 2:
            return
        rb2 = 2 * self.rb
        c = self.calm()
        for _ in range(2 if c else 1):
            P = np.stack([self.X, self.Y], 1)
            # Gitter-Sortierung: nur Nachbarn in Zellen vergleichen
            cell = np.floor((P + 500) / rb2).astype(np.int64)
            key = cell[:, 0] * 10000 + cell[:, 1]
            order = np.argsort(key)
            ks = key[order]
            pairs_i, pairs_j = [], []
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    nk = key + ox * 10000 + oy
                    lo = np.searchsorted(ks, nk, "left")
                    hi = np.searchsorted(ks, nk, "right")
                    cnt = hi - lo
                    if cnt.max() == 0:
                        continue
                    rep = np.repeat(np.arange(len(key)), cnt)
                    offs = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi) if b > a]) if cnt.sum() else np.array([], int)
                    j = order[offs]
                    m = rep < j
                    pairs_i.append(rep[m])
                    pairs_j.append(j[m])
            if not pairs_i:
                return
            I, J = np.concatenate(pairs_i), np.concatenate(pairs_j)
            dx, dy = self.X[J] - self.X[I], self.Y[J] - self.Y[I]
            dd = np.hypot(dx, dy)
            m = (dd < rb2) & (dd > 1e-6)
            if not m.any():
                return
            I, J, dx, dy, dd = I[m], J[m], dx[m], dy[m], dd[m]
            nx, ny = dx / dd, dy / dd
            ov = (rb2 - dd) / 2
            np.add.at(self.X, I, -nx * ov)
            np.add.at(self.Y, I, -ny * ov)
            np.add.at(self.X, J, nx * ov)
            np.add.at(self.Y, J, ny * ov)
            rel = (self.VX[I] - self.VX[J]) * nx + (self.VY[I] - self.VY[J]) * ny
            rel = np.where(rel > 0, rel, 0) * ((1 + (0.2 if c else 0.9)) / 2)
            np.add.at(self.VX, I, -rel * nx)
            np.add.at(self.VY, I, -rel * ny)
            np.add.at(self.VX, J, rel * nx)
            np.add.at(self.VY, J, rel * ny)
        if not c:  # Energie halten: niemand bleibt liegen, solange das Chaos läuft
            self.keep_energy()
        # nicht aus dem Kreis schieben lassen
        d = np.hypot(self.X, self.Y) + 1e-9
        o = d > ARENA - self.rb
        self.X[o] *= (ARENA - self.rb) / d[o]
        self.Y[o] *= (ARENA - self.rb) / d[o]

    def keep_energy(self, vmin=480.0):
        """Zu langsame Bälle auf Mindesttempo bringen; fast stehende bekommen eine neue Zufallsrichtung."""
        sp = np.hypot(self.VX, self.VY)
        slow = sp < vmin
        if not slow.any():
            return
        tiny = sp < 30
        if tiny.any():
            a = self.np_rng.uniform(0, 2 * np.pi, int(tiny.sum()))
            self.VX[tiny] = np.cos(a) * vmin
            self.VY[tiny] = np.sin(a) * vmin
        m = slow & ~tiny
        k = vmin / sp[m]
        self.VX[m] *= k
        self.VY[m] *= k

    def check_rest2(self):
        if self.calm_t is not None and self.finish_t is None and self.t > self.calm_t + 4.0:
            if np.percentile(np.hypot(self.VX, self.VY), 90) < 40 or self.t > self.calm_t + 6.5:
                self.finish_t = self.t

    def run(self, max_t=70.0):
        while True:
            for _ in range(SUB):
                self.step()
                self.t += DT
            self.collide()
            if self.phase2_t is not None:
                self.check_rest2()
            self.frames.append(self.snapshot())
            if self.finish_t is not None and self.t >= self.finish_t + self.outro:
                break
            if self.t > max_t:
                break
        return self

    def snapshot(self):
        if self.phase2_t is None:
            b = self.ball
            return (b["x"], b["y"], self.k, None)
        return (0.0, 0.0, self.k, (self.X.astype(np.float32).copy(), self.Y.astype(np.float32).copy(),
                                     self.H.astype(np.float32).copy()))


# ------------------------------------------------------------------ Miau: synthetische Katze, wird tiefer + langsamer

def meow(p, seed=0):
    """Synthetisches Miau. p=1: kleine Katze (hoch, kurz) ... p~0.3: tief und gedehnt."""
    rng = np.random.default_rng(seed)
    D = 0.42 * (1 / p) ** 1.05
    n = int(SR * D)
    u = np.linspace(0, 1, n)
    # Tonhöhe: steigt, Bogen, fällt am Ende ab + Vibrato
    f0 = 720 * p * (0.82 + 0.38 * np.sin(np.pi * np.clip(u * 1.1, 0, 1)) ** 0.7 - 0.12 * u ** 3)
    f0 *= 1 + 0.018 * np.sin(2 * np.pi * 6.0 * u * D)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    # Formanten (Hz, Bandbreite, Gewicht): m -> i -> a -> u
    keys = np.array([0.0, 0.12, 0.35, 0.68, 1.0])
    F1 = np.array([260, 380, 900, 820, 420]) * p ** 0.45
    F2 = np.array([1000, 2300, 1550, 1250, 900]) * p ** 0.45
    F3 = np.array([2500, 3000, 2600, 2500, 2400]) * p ** 0.45
    g1 = np.interp(u, keys, F1)
    g2 = np.interp(u, keys, F2)
    g3 = np.interp(u, keys, F3)
    nasal = np.interp(u, keys, [1.0, 0.15, 0.0, 0.0, 0.2])
    out = np.zeros(n)
    hmax = int(7000 / max(f0.min(), 60))
    for h in range(1, max(2, hmax)):
        fh = h * f0
        amp = (np.exp(-((fh - g1) / 160) ** 2) * 1.0 + np.exp(-((fh - g2) / 220) ** 2) * 0.55
               + np.exp(-((fh - g3) / 300) ** 2) * 0.25 + 0.02) / h ** 0.3
        amp *= 1 - 0.7 * nasal * (fh > 500)
        out += amp * np.sin(h * phase)
    breath = np.convolve(rng.standard_normal(n), np.ones(14) / 14, mode="same") * 0.025
    env = np.minimum(u * D / 0.035, 1) * np.minimum((1 - u) * D / 0.12, 1)
    s = (out + breath) * env
    return s / (np.max(np.abs(s)) or 1) * 0.6


class MeowSim(BaseSim):
    """Katzen-Ball im Kreis. Die Zeit läuft immer langsamer (Zeitlupe) – passend dazu wird jedes Miau tiefer und
    gedehnter. Am Ende kommt der Ball zur Ruhe."""

    def __init__(self, seed, active=44.0, r=52.0, s_end=0.33, speed=(820, 980), g=550.0):
        super().__init__(seed, None)
        rng = self.rng
        a = rng.uniform(0, 2 * math.pi)
        self.ball = dict(x=0.0, y=-120.0, vx=math.cos(a) * 850, vy=math.sin(a) * 850, r=r)
        self.active, self.s_end, self.speed = active, s_end, speed
        self.g = g

    def slowmo(self):
        return 1 - (1 - self.s_end) * min(1.0, self.t / self.active)

    def step(self):
        if self.calm_t is None and self.t >= self.active:
            self.calm_t = self.t
        sm = self.slowmo()
        b = self.ball
        dt = DT * sm  # Zeitlupe: Physik läuft langsamer als die Uhr
        b["vy"] += self.g * dt
        c = self.calm()
        if c:
            f = 1 - (0.3 + 1.8 * c) * dt
            b["vx"] *= f
            b["vy"] *= f
        b["x"] += b["vx"] * dt
        b["y"] += b["vy"] * dt
        res, st = self.wall(ARENA, speed=self.speed, jitter=0.3)
        if res == "bounce" and st > 0.12:
            self.ev("note", strength=st, p=sm)
        self.check_rest()

    def check_rest(self):
        b = self.ball
        if self.calm_t is not None and self.finish_t is None and self.t > self.calm_t + 4:
            if math.hypot(b["vx"], b["vy"]) < 15 or self.t > self.calm_t + 9:
                self.finish_t = self.t

    def snapshot(self):
        b = self.ball
        return (b["x"], b["y"], 0, None)


def build_meow_audio(sim, path):
    total = len(sim.frames) / FPS
    n = int(SR * total)
    out = np.zeros(n + SR * 4)
    notes = [(t, d) for t, k, d in sim.events if k == "note"]
    for j, (t, d) in enumerate(notes):
        p = d["p"]  # Zeitlupen-Faktor = Tonhöhe/Tempo des Miaus
        s = meow(p, seed=j) * (0.55 + 0.45 * d["strength"])
        i = int(t * SR)
        k = min(len(out), i + len(s))
        out[i:k] += s[: k - i]
        thump = np.sin(2 * np.pi * 90 * np.arange(int(SR * 0.12)) / SR) * np.exp(-np.arange(int(SR * 0.12)) / SR * 30) * 0.25
        out[i:i + len(thump)] += thump[: max(0, min(len(thump), len(out) - i))]
    out = out[:n]
    wet = reverb(out, 1.4)
    mix = out * 0.9 + wet * 0.2
    mix *= np.clip((total - np.arange(n) / SR) / 1.0, 0, 1)
    peak = np.max(np.abs(mix)) or 1
    mix = np.tanh(mix / peak * 1.1) * 0.9
    st = np.stack([mix, mix], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ Katzen-Chaos: Katzen vermehren sich, alle miauen

FURS = [(255, 160, 70), (170, 170, 180), (250, 245, 235), (60, 55, 60), (240, 200, 140), (200, 120, 60), (120, 110, 115)]


class CatChaosSim(MultiplySpiralSim):
    """Start mit einer Katze. Jeder Aufprall miaut (eigene Stimmlage je Katze) und kann eine neue Katze erzeugen.
    Ist der Kreis voll, wird alles zur Zeitlupe: die Miaus werden tiefer und langsamer, die Katzen sinken ab."""

    def __init__(self, seed, cap=100, p_spawn=0.11, r_cat=22.0, hold=6.0, slow_end=0.35, max_rate=26.0):
        super().__init__(seed, n=2, cap=cap, p_spawn=p_spawn, r_ball=r_cat, crazy_hold=hold)
        rng = self.rng
        a = rng.uniform(0, 2 * math.pi)
        self.phase2_t = 0.0
        self.X, self.Y = np.array([0.0]), np.array([-100.0])
        self.VX, self.VY = np.array([math.cos(a) * 700]), np.array([math.sin(a) * 700])
        self.H = np.array([0.0])
        self.PITCH = np.array([1.0])
        self.LAST = np.array([-9.0])
        self.slow_end, self.max_rate = slow_end, max_rate
        self.meow_times = []

    def slowmo(self):
        if self.calm_t is None:
            return 1.0
        return 1 - (1 - self.slow_end) * min(1.0, max(0.0, (self.t - self.calm_t) / 2.5))

    def step(self):
        sm = self.slowmo()
        dt, R, rb = DT * sm, ARENA, self.rb
        c = self.calm()
        self.VY += (1000.0 if c else 650.0) * dt
        if c:
            f = 1 - (0.2 + 0.9 * c) * dt
            self.VX *= f
            self.VY *= f
        self.X += self.VX * dt
        self.Y += self.VY * dt
        d = np.hypot(self.X, self.Y) + 1e-9
        out = d + rb > R
        if not out.any():
            return
        idx = np.nonzero(out)[0]
        nx, ny = self.X[idx] / d[idx], self.Y[idx] / d[idx]
        self.X[idx], self.Y[idx] = nx * (R - rb), ny * (R - rb)
        vx, vy = self.VX[idx], self.VY[idx]
        vn = vx * nx + vy * ny
        hit = vn > 0
        if not hit.any():
            return
        if c:
            e = 1 - 0.45 * c
            vx = np.where(hit, vx - (1 + e) * vn * nx, vx)
            vy = np.where(hit, vy - (1 + e) * vn * ny, vy)
        else:
            vx = np.where(hit, vx - 2 * vn * nx, vx)
            vy = np.where(hit, vy - 2 * vn * ny, vy)
            a = self.np_rng.uniform(-0.3, 0.3, len(idx))
            vx, vy = vx * np.cos(a) - vy * np.sin(a), vx * np.sin(a) + vy * np.cos(a)
            sp = np.hypot(vx, vy) + 1e-9
            k = np.clip(sp, 560, 820) / sp
            vx, vy = vx * k, vy * k
        self.VX[idx], self.VY[idx] = vx, vy
        hi = idx[hit]
        st = np.minimum(1.0, vn[hit] / 900)
        # Miau bei jedem Aufprall (je Katze nicht überlappend, insgesamt begrenzte Dichte)
        for j, s_ in zip(hi, st):
            if s_ < (0.02 if c else 0.1):  # im Ausklang miauen auch sanfte Aufpralle (tief + langsam)
                continue
            p = float(self.PITCH[j]) * sm
            dur = 0.42 * (1 / p) ** 1.05
            if self.t - self.LAST[j] < dur * 0.8:
                continue
            recent = [x for x in self.meow_times[-60:] if self.t - x < 1.0]
            if len(recent) >= self.max_rate * (0.6 if c else 1.0):
                continue
            self.LAST[j] = self.t
            self.meow_times.append(self.t)
            self.ev("meow", cat=int(j), p=p, dur=dur, strength=float(s_))
        # Vermehrung
        if c == 0 and len(self.X) < self.cap:
            m = self.np_rng.random(len(hi)) < self.p_spawn
            src = hi[m][: self.cap - len(self.X)]
            if len(src):
                L = len(src)
                bx, by = self.X[src] * 0.9, self.Y[src] * 0.9
                inx, iny = -bx / (np.hypot(bx, by) + 1e-9), -by / (np.hypot(bx, by) + 1e-9)
                base = np.arctan2(iny, inx) + self.np_rng.uniform(-0.9, 0.9, L)
                sp = self.np_rng.uniform(600, 800, L)
                self.X = np.concatenate([self.X, bx])
                self.Y = np.concatenate([self.Y, by])
                self.VX = np.concatenate([self.VX, np.cos(base) * sp])
                self.VY = np.concatenate([self.VY, np.sin(base) * sp])
                self.H = np.concatenate([self.H, self.np_rng.integers(0, len(FURS), L).astype(float)])
                self.PITCH = np.concatenate([self.PITCH, self.np_rng.uniform(0.72, 1.35, L)])
                self.LAST = np.concatenate([self.LAST, np.full(L, -9.0)])
                self.ev("spawn", n=len(self.X))
                if len(self.X) >= self.cap and self.cap_t is None:
                    self.cap_t = self.t
                    self.calm_t = self.t + self.crazy_hold


    def check_rest2(self):
        if self.calm_t is not None and self.finish_t is None and self.t > self.calm_t + 5.0:
            if np.percentile(np.hypot(self.VX, self.VY), 90) < 30 or self.t > self.calm_t + 7.0:
                self.finish_t = self.t


def build_cat_chaos_audio(sim, path):
    total = len(sim.frames) / FPS
    n = int(SR * total)
    out = np.zeros(n + SR * 4)
    cache = {}
    for t, kind, d in sim.events:
        if kind != "meow":
            continue
        key = round(d["p"], 2)
        if key not in cache:
            cache[key] = meow(key, seed=int(key * 100))
        s = cache[key] * (0.5 + 0.5 * d["strength"])
        i = int(t * SR)
        j = min(len(out), i + len(s))
        out[i:j] += s[: j - i]
    out = out[:n]
    wet = reverb(out, 1.6)
    mix = out * 0.85 + wet * 0.25
    mix *= np.clip((total - np.arange(n) / SR) / 1.0, 0, 1)
    peak = np.max(np.abs(mix)) or 1
    mix = np.tanh(mix / peak * 1.6) * 0.9
    st = np.stack([mix, mix], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
