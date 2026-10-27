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

# ------------------------------------------------------------------ Pong Wars 2: Bälle vermehren sich, es gibt einen Sieger

def soft_tone(f):
    """Runder, weicher Ton (kaum Obertöne, sanfter Anschlag)."""
    key = ("soft", round(f, 2))
    if key not in _cache:
        n = int(SR * 1.4)
        t = np.arange(n) / SR
        s = (np.sin(2 * np.pi * f * t) + 0.12 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 6)) * np.exp(-t * 3.2)
        _cache[key] = s * np.minimum(t / 0.012, 1) * 0.3
    return _cache[key]


def warm_pad(total, fade_from):
    n = int(SR * total)
    t = np.arange(n) / SR
    chords = [(130.81, 196.0, 261.63, 329.63), (110.0, 164.81, 220.0, 261.63), (87.31, 130.81, 174.61, 220.0),
              (98.0, 146.83, 196.0, 246.94)]
    out = np.zeros(n)
    seg = 6.0
    for ci in range(int(total / seg) + 2):
        t0 = ci * seg
        env = np.clip((t - t0 + 1.5) / 1.5, 0, 1) * np.clip((t0 + seg + 1.5 - t) / 1.5, 0, 1)
        for k, f in enumerate(chords[ci % 4]):
            out += env * np.sin(2 * np.pi * f * t + k) / (k + 2)
    out *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * t)
    return out * 0.06 * np.minimum(1, t / 2) * np.clip(1 - (t - fade_from) / 8, 0.4, 1)


class PongWars2(PongWars):
    def __init__(self, seed, p_split=0.06, cap=40, decide_at=41.0, flood=3.2, total=50.0, speed=600.0):
        super().__init__(seed, grid=30, total=total, adds=(), slow_at=decide_at + flood + 1.0, speed=speed)
        self.p_split, self.cap, self.decide_at, self.flood = p_split, cap, decide_at, flood
        self.winner = None
        self.area = []
        self.gone = {}  # Ball-Index -> Zeitpunkt des Verschwindens

    def run(self):
        dt_sub = 1 / (FPS * 6)
        r = self.cell * 0.42
        n = int(self.total * FPS)
        g = self.g
        for fi in range(n):
            t = fi / FPS
            sp = self.speed
            if t > self.slow_at:
                sp *= max(0.0, 1 - (t - self.slow_at) / 3.5) ** 1.5
            if t >= self.decide_at and self.winner is None:
                day = int((self.state == 0).sum())
                self.winner = 0 if day >= g * g - day else 1
                self.events.append((t, "decide", {"winner": self.winner}))
                # Flutwelle: Reihenfolge nach Abstand zum Schwerpunkt des Siegers
                ys, xs = np.nonzero(self.state == self.winner)
                cy, cx = ys.mean(), xs.mean()
                yy, xx = np.mgrid[0:g, 0:g]
                dist = np.hypot(yy - cy, xx - cx)
                self.flood_t = self.decide_at + dist / dist.max() * self.flood
            if self.winner is not None:
                conv = (self.state != self.winner) & (self.flood_t <= t)
                if conv.any():
                    rows = np.nonzero(conv)[0]
                    self.state[conv] = self.winner
                    self.events.append((t, "flood", {"n": int(conv.sum()), "row": int(rows.mean())}))
                for bi, b in enumerate(self.balls):  # Verlierer-Bälle verschwinden, wenn die Welle sie erreicht
                    if b[4] != self.winner and bi not in self.gone:
                        ci, cj = min(g - 1, int(b[0] // self.cell)), min(g - 1, int(b[1] // self.cell))
                        if self.flood_t[cj, ci] <= t:
                            self.gone[bi] = t
                            self.events.append((t, "pop", {}))
            new = []
            for _ in range(6):
                for bi, b in enumerate(self.balls):
                    if bi in self.gone:
                        continue
                    b[0] += b[2] * sp * dt_sub
                    b[1] += b[3] * sp * dt_sub
                    if b[0] < r and b[2] < 0 or b[0] > self.size - r and b[2] > 0:
                        b[2] = -b[2]
                    if b[1] < r and b[3] < 0 or b[1] > self.size - r and b[3] > 0:
                        b[3] = -b[3]
                    if self.winner is not None:
                        continue  # nach der Entscheidung wird nicht mehr erobert
                    for ang in range(0, 360, 45):
                        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                        ci, cj = int((b[0] + ca * r) // self.cell), int((b[1] + sa * r) // self.cell)
                        if 0 <= ci < g and 0 <= cj < g and self.state[cj, ci] != b[4]:
                            self.state[cj, ci] = b[4]
                            if abs(ca) > abs(sa):
                                b[2] = -b[2]
                            else:
                                b[3] = -b[3]
                            a = self.rng.uniform(-0.1, 0.1)
                            vx, vy = b[2] * math.cos(a) - b[3] * math.sin(a), b[2] * math.sin(a) + b[3] * math.cos(a)
                            L = math.hypot(vx, vy)
                            b[2], b[3] = vx / L, vy / L
                            self.events.append((t, "flip", {"team": b[4], "row": cj, "col": ci}))
                            # Treffer -> neuer Ball für dieses Team (Abspaltung in Gegenrichtung)
                            team_n = sum(1 for x in self.balls + new if x[4] == b[4])
                            if team_n < self.cap and self.rng.random() < self.p_split:
                                aa = self.rng.uniform(0, 2 * math.pi)
                                new.append([b[0], b[1], math.cos(aa), math.sin(aa), b[4]])
                                self.events.append((t, "split", {"team": b[4]}))
                            break
            self.balls += new
            self.area.append(int((self.state == 0).sum()) / (g * g))
            self.frames.append((self.state.copy(), [tuple(b) + (self.gone.get(bi),) for bi, b in enumerate(self.balls)],
                                self.area[-1]))
        return self

    def lead_changes(self):
        lead, ch = None, 0
        for a in self.area[FPS * 3: int(self.decide_at * FPS)]:
            l = 0 if a > 0.53 else 1 if a < 0.47 else lead
            if l is not None and lead is not None and l != lead:
                ch += 1
            lead = l if l is not None else lead
        return ch

    def frame(self, fi):
        k = SS
        state, balls, area = self.frames[fi]
        t = fi / FPS
        img = Image.new("RGB", (W1 * k, H1 * k), (10, 12, 16))
        lut = np.array([DAY, NIGHT], np.uint8)
        grid = Image.fromarray(lut[state], "RGB").resize((int(self.size * k), int(self.size * k)), Image.NEAREST)
        img.paste(grid, (int(self.x0 * k), int(self.y0 * k)))
        d = ImageDraw.Draw(img)
        # Kräfte-Balken über dem Feld (ohne Text)
        bx0, by0 = self.x0 * k, (self.y0 - 46) * k
        bw, bh = self.size * k, 18 * k
        d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=bh // 2, fill=NIGHT)
        d.rounded_rectangle([bx0, by0, bx0 + bw * area, by0 + bh], radius=bh // 2, fill=DAY)
        d.line([bx0 + bw / 2, by0 - 6 * k, bx0 + bw / 2, by0 + bh + 6 * k], fill=(240, 120, 110), width=2 * k)
        r = self.cell * 0.42 * k
        for x, y, vx, vy, team, gone in balls:
            X, Y = (self.x0 + x) * k, (self.y0 + y) * k
            col = DAY if team == 1 else NIGHT
            rr = r
            if gone is not None:
                e = (t - gone) / 0.35
                if e >= 1:
                    continue
                rr = r * (1 + 0.8 * e)
                col = mix(col, lut[team].tolist() and tuple(lut[1 - team]), e)
            d.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=col)
        return img.reduce(SS)

    def audio(self, path):
        n = int(SR * self.total)
        out = np.zeros(n + SR * 3)
        penta = [0, 2, 4, 7, 9]
        last = -1.0

        def put(t, s, vol):
            i = int(t * SR)
            e = min(len(out), i + len(s))
            if i < e:
                out[i:e] += s[: e - i] * vol

        for t, kind, d in self.events:
            if kind == "flip":
                if t - last < 0.085:  # höchstens ~12 Töne pro Sekunde
                    continue
                last = t
                row = d["row"] % 10
                semi = 12 * (row // 5) + penta[row % 5]
                base = 261.63 if d["team"] == 0 else 196.0
                put(t, soft_tone(base * 2 ** (semi / 12)), 0.5)
            elif kind == "split":
                put(t, soft_tone(1046.5 if d["team"] == 0 else 783.99), 0.12)
            elif kind == "flood":
                if t - last < 0.06:
                    continue
                last = t
                row = d["row"]
                put(t, soft_tone(523.25 * 2 ** ((penta[row % 5] + 12 * (row % 2)) / 12)), 0.4)
            elif kind == "decide":
                for j, f in enumerate((261.63, 329.63, 392.0, 523.25)):
                    put(t + j * 0.12, vibes(f), 0.35)
        win_t = self.decide_at + self.flood
        for j, f in enumerate((261.63, 329.63, 392.0, 523.25, 659.25)):
            put(win_t + j * 0.09, vibes(f), 0.4)
        out = out[:n]
        mix_ = out * 0.8 + reverb(out, 2.4) * 0.4 + warm_pad(self.total, win_t)
        mix_ *= np.clip((self.total - np.arange(n) / SR) / 1.2, 0, 1)
        write_wav(mix_, path)


VIRAL["x03_pong_wars_sieger"] = lambda: PongWars2(4, p_split=0.08).run()


# ------------------------------------------------------------------ Pong Wars – Angriff (klare Schlacht mit Sieger)

def asmr_tap(f, seed=0):
    """Leises, trockenes Tippen wie eine Glasmurmel auf Holz."""
    key = ("tap", round(f, 1), seed)
    if key not in _cache:
        rng = np.random.default_rng(seed)
        n = int(SR * 0.25)
        t = np.arange(n) / SR
        click = np.convolve(rng.standard_normal(n), np.ones(5) / 5, mode="same") * np.exp(-t * 260)
        body = np.sin(2 * np.pi * f * t) * np.exp(-t * 28) + 0.3 * np.sin(2 * np.pi * f * 2.7 * t) * np.exp(-t * 60)
        _cache[key] = (click * 0.6 + body * 0.5) * 0.4
    return _cache[key]


def asmr_pop():
    n = int(SR * 0.25)
    t = np.arange(n) / SR
    f = 420 + 380 * np.exp(-t * 40)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 22) * 0.45


def asmr_thup():
    n = int(SR * 0.3)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (90 + 60 * np.exp(-t * 30)) * t) * np.exp(-t * 16) * 0.6


class PongBattle(PongWars):
    """Hell (Team 0) greift an, Dunkel (Team 1) verteidigt. Neue Bälle nur in festen Wellen aus der eigenen Basis.
    Eingekesselte Bälle scheiden aus. Wer keine Bälle mehr hat, verliert; der Sieger überflutet den Rest."""

    # (Zeit, neue helle Bälle, neue dunkle Bälle, Biss-Radius hell, Biss-Radius dunkel)
    WAVES = [(0.0, 1, 1, 0, 0), (5.0, 1, 1, 0, 0), (12.0, 0, 3, 0, 1), (19.0, 3, 0, 1, 0), (27.0, 3, 0, 1, 0),
             (34.0, 4, 0, 2, 0), (39.0, 3, 0, 3, 0)]

    def __init__(self, seed, speed=520.0, waves=None, grid=30):
        super().__init__(seed, grid=grid, total=70.0, adds=(), slow_at=99, speed=speed)
        self.balls = []
        self.waves = list(waves or self.WAVES)
        self.dead = {}
        self.winner, self.win_t = None, None
        self.area = []
        self.power = {}
        self.bites = []

    def spawn(self, team, t, k_in_wave, power=0):
        g = self.g
        # Startfeld: eigenes Feld, möglichst nah an der Mitte der eigenen Basis-Wand
        own = np.argwhere(self.state == team)
        if len(own) == 0:
            return
        home = np.array([g / 2, 0 if team == 0 else g - 1])
        cj, ci = own[np.argmin(np.hypot(own[:, 0] - home[0], own[:, 1] - home[1]) + 0.001 * np.arange(len(own)))]
        x, y = (ci + 0.5) * self.cell, (cj + 0.5) * self.cell
        spread = (k_in_wave - 0.5) * 0.35 if k_in_wave else 0.0
        a = (0.0 if team == 0 else math.pi) + spread + self.rng.uniform(-0.25, 0.25)
        self.power[len(self.balls)] = power
        self.balls.append([x, y, math.cos(a), math.sin(a), team])
        self.events.append((t, "spawn", {"team": team, "power": power}))

    def run(self):
        dt_sub = 1 / (FPS * 6)
        r = self.cell * 0.42
        g = self.g
        fi = 0
        pending = []  # (Zeit, Team, Nr in der Welle) – Bälle einer Welle kommen kurz nacheinander
        for wt, n0, n1, p0, p1 in self.waves:
            for k in range(n0):
                pending.append((wt + 0.35 * k, 0, k, p0))
            for k in range(n1):
                pending.append((wt + 0.35 * k, 1, k, p1))
        pending.sort()
        while True:
            t = fi / FPS
            while pending and pending[0][0] <= t and self.winner is None:
                _, team, k, pw = pending.pop(0)
                self.spawn(team, t, k, pw)
            sp = self.speed
            if self.win_t is not None:
                sp *= max(0.0, 1 - (t - self.win_t - 2.0) / 3.5) ** 1.5 if t > self.win_t + 2.0 else 1.0
            for _ in range(6):
                for bi, b in enumerate(self.balls):
                    if bi in self.dead:
                        continue
                    b[0] += b[2] * sp * dt_sub
                    b[1] += b[3] * sp * dt_sub
                    if b[0] < r and b[2] < 0 or b[0] > self.size - r and b[2] > 0:
                        b[2] = -b[2]
                    if b[1] < r and b[3] < 0 or b[1] > self.size - r and b[3] > 0:
                        b[3] = -b[3]
                    if self.winner is not None:
                        continue
                    for ang in range(0, 360, 45):
                        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                        ci, cj = int((b[0] + ca * r) // self.cell), int((b[1] + sa * r) // self.cell)
                        if 0 <= ci < g and 0 <= cj < g and self.state[cj, ci] != b[4]:
                            pw = self.power.get(bi, 0)
                            n_bite = 0
                            for dj in range(-pw, pw + 1):  # Biss: Raute mit Radius pw
                                for di in range(-(pw - abs(dj)), pw - abs(dj) + 1):
                                    yj, xi = cj + dj, ci + di
                                    if 0 <= yj < g and 0 <= xi < g and self.state[yj, xi] != b[4]:
                                        self.state[yj, xi] = b[4]
                                        n_bite += 1
                            if pw:
                                self.bites.append((t, ci, cj, pw, b[4]))
                            if abs(ca) > abs(sa):
                                b[2] = -b[2]
                            else:
                                b[3] = -b[3]
                            a = self.rng.uniform(-0.06, 0.06)
                            vx, vy = b[2] * math.cos(a) - b[3] * math.sin(a), b[2] * math.sin(a) + b[3] * math.cos(a)
                            L = math.hypot(vx, vy)
                            b[2], b[3] = vx / L, vy / L
                            self.events.append((t, "flip", {"team": b[4], "row": cj, "bite": n_bite}))
                            break
            # Einkesselung: steht ein Ball auf gegnerischem Feld oder hat kaum eigene Felder um sich -> raus
            if self.winner is None:
                for bi, b in enumerate(self.balls):
                    if bi in self.dead:
                        continue
                    ci, cj = min(g - 1, int(b[0] // self.cell)), min(g - 1, int(b[1] // self.cell))
                    nb = self.state[max(0, cj - 1):cj + 2, max(0, ci - 1):ci + 2]
                    if (nb == b[4]).sum() <= 1 and self.state[cj, ci] != b[4]:  # eingekesselt
                        self.dead[bi] = t
                        self.events.append((t, "pop", {"team": b[4]}))
                # Zusammenbruch: unter 8 % Fläche ploppen die letzten Bälle nacheinander
                for tm in (0, 1):
                    frac = float((self.state == tm).mean())
                    if frac < 0.08 and any(b[4] == tm for b in self.balls):
                        last_pop = max([tt for tt in self.dead.values()] + [-9])
                        if t - last_pop > 0.45:
                            for bi, b in enumerate(self.balls):
                                if bi not in self.dead and b[4] == tm:
                                    self.dead[bi] = t
                                    self.events.append((t, "pop", {"team": tm}))
                                    break
                alive = [sum(1 for bi, b in enumerate(self.balls) if bi not in self.dead and b[4] == tm) for tm in (0, 1)]
                spawned = [any(b[4] == tm for b in self.balls) for tm in (0, 1)]
                for tm in (0, 1):
                    if spawned[tm] and alive[tm] == 0 and not any(p[1] == tm for p in pending):
                        self.winner, self.win_t = 1 - tm, t
                        ys, xs = np.nonzero(self.state == self.winner)
                        yy, xx = np.mgrid[0:g, 0:g]
                        dist = np.hypot(yy - ys.mean(), xx - xs.mean())
                        self.flood_t = t + 0.3 + dist / dist.max() * 2.2
                        self.events.append((t, "win", {"winner": self.winner}))
                        break
            if self.winner is not None:
                conv = (self.state != self.winner) & (self.flood_t <= t)
                if conv.any():
                    self.state[conv] = self.winner
                    self.events.append((t, "flood", {"row": int(np.nonzero(conv)[0].mean())}))
            self.area.append(int((self.state == 0).sum()) / (g * g))
            self.frames.append((self.state.copy(), [tuple(b) + (self.dead.get(bi),) for bi, b in enumerate(self.balls)],
                                self.area[-1]))
            fi += 1
            if self.win_t is not None and t >= self.win_t + 6.5:
                break
            if t > 62:
                break
        self.total = len(self.frames) / FPS
        return self

    def frame(self, fi):
        img = PongWars2.frame(self, fi)
        # Biss-Effekt: kurzes helles Aufleuchten der eroberten Raute
        t = fi / FPS
        d = ImageDraw.Draw(img)
        c = self.cell
        for t0, ci, cj, pw, team in self.bites[-200:]:
            e = (t - t0) / 0.35
            if 0 <= e < 1:
                cx, cy = self.x0 + (ci + 0.5) * c, self.y0 + (cj + 0.5) * c
                R = (pw + 0.5) * c * (1 + 0.25 * e)
                col = mix(DAY if team == 0 else NIGHT, (255, 255, 255), 0.6 * (1 - e))
                d.polygon([(cx, cy - R), (cx + R, cy), (cx, cy + R), (cx - R, cy)], outline=col, width=2)
        return img

    def audio(self, path):
        n = int(SR * self.total)
        out = np.zeros(n + SR * 3)
        penta = [0, 2, 4, 7, 9]
        last = -1.0

        def put(t, s, vol):
            i = int(t * SR)
            e = min(len(out), i + len(s))
            if i < e:
                out[i:e] += s[: e - i] * vol

        for t, kind, d in self.events:
            if kind == "flip":
                if t - last < 0.06:
                    continue
                last = t
                semi = penta[d["row"] % 5] + 12 * ((d["row"] // 5) % 2)
                f = (880.0 if d["team"] == 0 else 660.0) * 2 ** (semi / 12)
                put(t, asmr_tap(f, seed=d["row"] % 7), 0.45)
            elif kind == "spawn":
                put(t, asmr_thup(), 0.5)
            elif kind == "pop":
                put(t, asmr_pop(), 0.6)
            elif kind == "flood":
                if t - last < 0.05:
                    continue
                last = t
                put(t, asmr_tap(1046.5 * 2 ** (penta[d["row"] % 5] / 12), seed=3), 0.35)
            elif kind == "win":
                for j, f in enumerate((261.63, 329.63, 392.0, 523.25)):
                    put(t + 2.6 + j * 0.15, soft_tone(f), 0.6)
        out = out[:n]
        tt = np.arange(n) / SR
        drone = (np.sin(2 * np.pi * 65.41 * tt) + 0.5 * np.sin(2 * np.pi * 98.0 * tt)) * 0.025 * np.minimum(1, tt / 3)
        mix_ = out * 0.9 + reverb(out, 0.9) * 0.15 + drone
        mix_ *= np.clip((self.total - tt) / 1.2, 0, 1)
        write_wav(mix_, path)


VIRAL["x04_pong_angriff"] = lambda: PongBattle(8).run()

if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(VIRAL)
    preview = "--preview" in names
    names = [n for n in names if not n.startswith("--")] or list(VIRAL)
    for name in names:
        obj = VIRAL[name]()
        if preview:
            render(obj, name, None, preview_dir="output/preview",
                   preview_times=[0.0, 6.0, 12.5, 16.0, 20.0, 27.0, 34.5, 39.5, 43.0, 46.5, 48.0, obj.frames_count() / FPS - 0.1])
            print(name, "Vorschau")
        else:
            out = f"output/{name}_vertical.mp4"
            render(obj, name, out)
            print(out)

