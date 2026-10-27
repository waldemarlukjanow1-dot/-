"""Zwei virale Satisfying-Formate.

poly: Polyrhythmus-Bögen – 21 Halbkreise, Punkt i schwingt n_i-mal hin und her; nach genau T Sekunden sind alle
      wieder ausgerichtet (perfekter Loop). Jeder Treffer an der Grundlinie = Ton seines Bogens + Aufleuchten.
pong: Pong Wars – zwei Teams färben Felder um. Weitere Bälle kommen dazu, am Ende wird es langsamer bis Stillstand.
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
from PIL import Image, ImageDraw

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb

FPS = 60
W1, H1, SS = 1080, 1920, 2
CX = 505  # horizontale Mitte (rechts Platz für TikTok-Buttons)
SR = 44100


def hsv(h, s=0.7, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def mix(c0, c1, f):
    return tuple(int(a + (b - a) * f) for a, b in zip(c0, c1))


# ------------------------------------------------------------------ Klangfarben

_cache = {}


def vibes(f, dur=2.8):
    """Weiches Vibraphon/Glocke."""
    key = ("v", round(f, 2))
    if key not in _cache:
        n = int(SR * dur)
        t = np.arange(n) / SR
        s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 1.6) + 0.25 * np.sin(2 * np.pi * 4.0 * f * t) * np.exp(-t * 7)
             + 0.1 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t * 5))
        s *= (1 + 0.1 * np.sin(2 * np.pi * 5.5 * t)) * np.minimum(t / 0.003, 1)
        _cache[key] = s * 0.35
    return _cache[key]


def wood(f):
    key = ("w", round(f, 2))
    if key not in _cache:
        n = int(SR * 0.35)
        t = np.arange(n) / SR
        s = np.sin(2 * np.pi * f * t) * np.exp(-t * 18) + 0.4 * np.sin(2 * np.pi * 2.3 * f * t) * np.exp(-t * 35)
        _cache[key] = s * np.minimum(t / 0.001, 1) * 0.4
    return _cache[key]


def write_wav(mix_, path):
    peak = np.max(np.abs(mix_)) or 1
    mix_ = np.tanh(mix_ / peak * 1.2) * 0.9
    st = np.stack([mix_, mix_], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ Polyrhythmus-Bögen

class Poly:
    def __init__(self, n=21, T=50.0, n_outer=16, intro=0.6, outro=2.4):
        self.n, self.T, self.intro, self.outro = n, T, intro, outro
        # Bogen 0 = außen (langsam, tief), Bogen n-1 = innen (schnell, hoch)
        self.cycles = [n_outer + i for i in range(n)]
        self.total = intro + T + outro
        self.base_y = 1230
        self.r_out, self.r_in = 415.0, 70.0
        self.hits = []  # (t, arc, side)
        for i, c in enumerate(self.cycles):
            per = T / c  # eine volle Hin-und-Her-Bewegung
            k = 0
            while True:
                th = intro + k * per / 2
                if th > intro + T + 1e-6:
                    break
                self.hits.append((th, i, k % 2))
                k += 1
        self.hits.sort()

    def radius(self, i):
        return self.r_out - (self.r_out - self.r_in) * i / (self.n - 1)

    def angle(self, i, t):
        """Winkel auf dem Halbkreis: pi = links an der Grundlinie, 0 = rechts."""
        e = min(max(t - self.intro, 0.0), self.T)
        ph = (e * self.cycles[i] / self.T) % 1.0  # 0..1 je Hin-und-Her
        tri = 1 - abs(1 - 2 * ph)  # 0 -> 1 -> 0
        return math.pi * (1 - tri)

    def scale_freq(self, i):
        # Dur-Pentatonik, außen tief -> innen hoch
        steps = [0, 2, 4, 7, 9]
        k = i
        semi = 12 * (k // 5) + steps[k % 5]
        return 130.81 * 2 ** (semi / 12)  # ab C3

    def frame(self, fi):
        t = fi / FPS
        k = SS
        img = Image.new("RGB", (W1 * k, H1 * k), (8, 8, 14))
        d = ImageDraw.Draw(img)
        by = self.base_y * k
        flash = [0.0] * self.n
        import bisect
        lo = bisect.bisect_left(self.hits, (t - 0.6, -1, -1))
        hi = bisect.bisect_right(self.hits, (t, 99, 99))
        for th, i, side in self.hits[lo:hi]:
            flash[i] = max(flash[i], 1 - (t - th) / 0.6)
        # Grundlinie
        d.line([(CX - self.r_out - 20) * k, by, (CX + self.r_out + 20) * k, by], fill=(200, 200, 215), width=3 * k)
        for i in range(self.n):
            R = self.radius(i) * k
            col = hsv(0.0 + 0.78 * i / (self.n - 1), 0.75, 1.0)
            f = flash[i]
            arc_col = mix(mix((8, 8, 14), col, 0.38), (255, 255, 255) if f > 0.85 else col, f)
            d.arc([CX * k - R, by - R, CX * k + R, by + R], 180, 360, fill=arc_col, width=int((4 + 3 * f) * k))
        # Treffer-Wellen an der Grundlinie
        for th, i, side in self.hits[lo:hi]:
            e = (t - th) / 0.6
            if e < 1:
                R = self.radius(i)
                x = CX + (-R if side == 0 else R)
                col = hsv(0.78 * i / (self.n - 1), 0.75, 1.0)
                rr = (6 + 26 * e) * k
                d.ellipse([x * k - rr, by - rr, x * k + rr, by + rr], outline=mix((8, 8, 14), col, 1 - e), width=2 * k)
        for i in range(self.n):
            R = self.radius(i)
            a = self.angle(i, t)
            x, y = CX + math.cos(a) * R, self.base_y - math.sin(a) * R
            col = hsv(0.78 * i / (self.n - 1), 0.75, 1.0)
            r = 11 * k
            d.ellipse([x * k - r * 1.6, y * k - r * 1.6, x * k + r * 1.6, y * k + r * 1.6], fill=mix((8, 8, 14), col, 0.3))
            d.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=mix(col, (255, 255, 255), 0.55))
        return img.reduce(SS)

    def frames_count(self):
        return int(self.total * FPS)

    def audio(self, path):
        n = int(SR * self.total)
        out = np.zeros(n + SR * 3)
        for th, i, side in self.hits:
            s = vibes(self.scale_freq(i))
            j = int(th * SR)
            e = min(len(out), j + len(s))
            out[j:e] += s[: e - j] * 0.5
        out = out[:n]
        mix_ = out * 0.8 + reverb(out, 2.6) * 0.45
        mix_ *= np.clip((self.total - np.arange(n) / SR) / 0.8, 0, 1)
        write_wav(mix_, path)


# ------------------------------------------------------------------ Pong Wars

DAY, NIGHT = (217, 232, 227), (17, 76, 90)


class PongWars:
    def __init__(self, seed, grid=30, size=840, total=52.0, adds=(8.0, 16.0, 24.0, 32.0), slow_at=46.0, speed=650.0):
        self.rng = random.Random(seed)
        self.g, self.size, self.total = grid, size, total
        self.cell = size / grid
        self.x0, self.y0 = CX - size / 2, 960 - size / 2
        self.adds, self.slow_at, self.speed = list(adds), slow_at, speed
        self.state = np.zeros((grid, grid), np.uint8)  # 0 = Tag, 1 = Nacht
        self.state[:, grid // 2:] = 1
        self.balls = []  # [x, y, vx, vy, team]
        self._add(0)
        self._add(1)
        self.events, self.frames = [], []

    def _add(self, team):
        rng = self.rng
        # jeder Ball startet im eigenen Gebiet (Tag links, Nacht rechts) und kämpft an der Grenze
        x = self.size * (0.25 if team == 0 else 0.75) + rng.uniform(-60, 60)
        y = self.size * rng.uniform(0.2, 0.8)
        a = rng.uniform(0, 2 * math.pi)
        self.balls.append([x, y, math.cos(a), math.sin(a), team])

    def run(self):
        dt_sub = 1 / (FPS * 6)
        r = self.cell * 0.42
        n = int(self.total * FPS)
        added = 0
        for fi in range(n):
            t = fi / FPS
            while added < len(self.adds) and t >= self.adds[added]:
                self._add(0)
                self._add(1)
                self.events.append((t, "add", {}))
                added += 1
            sp = self.speed * (1 + 0.25 * min(1.0, t / self.slow_at))
            if t > self.slow_at:
                sp *= max(0.0, 1 - (t - self.slow_at) / 3.5) ** 1.5
            for _ in range(6):
                for b in self.balls:
                    b[0] += b[2] * sp * dt_sub
                    b[1] += b[3] * sp * dt_sub
                    # Wände
                    if b[0] < r and b[2] < 0 or b[0] > self.size - r and b[2] > 0:
                        b[2] = -b[2]
                    if b[1] < r and b[3] < 0 or b[1] > self.size - r and b[3] > 0:
                        b[3] = -b[3]
                    # Felder des Gegners am Ballrand prüfen
                    for ang in range(0, 360, 45):
                        px = b[0] + math.cos(math.radians(ang)) * r
                        py = b[1] + math.sin(math.radians(ang)) * r
                        ci, cj = int(px // self.cell), int(py // self.cell)
                        if 0 <= ci < self.g and 0 <= cj < self.g and self.state[cj, ci] != b[4]:
                            self.state[cj, ci] = b[4]
                            if abs(math.cos(math.radians(ang))) > abs(math.sin(math.radians(ang))):
                                b[2] = -b[2]
                            else:
                                b[3] = -b[3]
                            a = self.rng.uniform(-0.08, 0.08)  # winzige Störung gegen Endlosschleifen
                            vx, vy = b[2] * math.cos(a) - b[3] * math.sin(a), b[2] * math.sin(a) + b[3] * math.cos(a)
                            L = math.hypot(vx, vy)
                            b[2], b[3] = vx / L, vy / L
                            if sp > 60:
                                self.events.append((t, "flip", {"team": b[4], "row": cj, "col": ci}))
                            break
            self.frames.append((self.state.copy(), [tuple(b) for b in self.balls]))
        return self

    def frame(self, fi):
        k = SS
        state, balls = self.frames[fi]
        img = Image.new("RGB", (W1 * k, H1 * k), (10, 12, 16))
        lut = np.array([DAY, NIGHT], np.uint8)
        grid = Image.fromarray(lut[state], "RGB").resize((int(self.size * k), int(self.size * k)), Image.NEAREST)
        img.paste(grid, (int(self.x0 * k), int(self.y0 * k)))
        d = ImageDraw.Draw(img)
        r = self.cell * 0.42 * k
        for x, y, vx, vy, team in balls:
            X, Y = (self.x0 + x) * k, (self.y0 + y) * k
            d.ellipse([X - r, Y - r, X + r, Y + r], fill=DAY if team == 1 else NIGHT)
        return img.reduce(SS)

    def frames_count(self):
        return len(self.frames)

    def audio(self, path):
        n = int(SR * self.total)
        out = np.zeros(n + SR * 2)
        penta = [0, 2, 4, 7, 9]
        last = {}
        for t, kind, d in self.events:
            if kind == "add":
                for j, f in enumerate((523.25, 659.25, 783.99)):
                    s = vibes(f)
                    i = int((t + j * 0.07) * SR)
                    out[i:i + len(s)] += s[: max(0, len(out) - i)][: len(out[i:i + len(s)])] * 0.4
                continue
            key = (d["team"], d["row"])
            if t - last.get(key, -1) < 0.05:
                continue
            last[key] = t
            semi = 12 * (d["row"] // 5) + penta[d["row"] % 5]
            if d["team"] == 0:  # Tag: hell, Holz-Glocke
                s = wood(392.0 * 2 ** (semi / 24))
            else:  # Nacht: weicher, tiefer
                s = vibes(196.0 * 2 ** (semi / 24)) * 0.7
            i = int(t * SR)
            e = min(len(out), i + len(s))
            out[i:e] += s[: e - i] * 0.45
        out = out[:n]
        mix_ = out * 0.85 + reverb(out, 1.8) * 0.3
        mix_ *= np.clip((self.total - np.arange(n) / SR) / 1.0, 0, 1)
        write_wav(mix_, path)


# ------------------------------------------------------------------ Ausgabe

_V = None


def _chunk(args):
    i0, i1, out = args
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W1}x{H1}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "12M", "-bufsize", "24M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_V.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(obj, name, out_path, preview_dir=None, preview_times=(), jobs=4):
    global _V
    _V = obj
    n = obj.frames_count()
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fr = obj.frame(min(int(pt * FPS), n - 1))
            base = os.path.join(preview_dir, f"{name}_vertical_{pt:05.1f}")
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
        obj.audio(wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


VIRAL = {
    "x01_polyrhythmus": lambda: Poly(),
    "x02_pong_wars": lambda: PongWars(7).run(),
}

if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(VIRAL)
    preview = "--preview" in names
    names = [n for n in names if not n.startswith("--")] or list(VIRAL)
    for name in names:
        obj = VIRAL[name]()
        if preview:
            render(obj, name, None, preview_dir="output/preview",
                   preview_times=[0.0, 0.7, 3.0, 8.0, 15.0, 22.0, 30.0, 38.0, 45.0, 49.5, 50.6, obj.frames_count() / FPS - 0.1])
            print(name, "Vorschau")
        else:
            out = f"output/{name}_vertical.mp4"
            render(obj, name, out)
            print(out)
