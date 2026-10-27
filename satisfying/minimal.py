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
        return min(1.0, (self.t - self.calm_t) / 3.5)

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

def lerp(c0, c1, f):
    return tuple(int(a + (b - a) * f) for a, b in zip(c0, c1))


class MinimalRenderer:
    def __init__(self, sim, kind):
        self.sim, self.kind = sim, kind

    def frame(self, i):
        sim = self.sim
        t = (i + 1) / FPS
        img = Image.new("RGB", (W1 * SS, H1 * SS), (0, 0, 0))
        d = ImageDraw.Draw(img)
        k = SS

        def P(x, y):
            return (CX * k + x * k, CY * k + y * k)

        x, y, state = sim.frames[i]
        if self.kind == "spiral":
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
            r = (sim.ball["r"] + 2) * k  # optisch etwas größer, damit er am Handy gut sichtbar ist
            X, Y = P(x, y)
            d.ellipse([X - r, Y - r, X + r, Y + r], fill=(110, 205, 240))
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
        build_audio(sim, wav, octave=spec.get("octave", 12))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path], check=True)
