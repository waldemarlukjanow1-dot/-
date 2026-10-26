"""Flow-Videos: zufällige Sprungphysik, Muster-Schweife, ruhiges Ende (Bälle bleiben liegen).

Kein Text, kein Feuerwerk. Kreis exakt in der Bildmitte. Jeder Ball zieht ein eigenes Muster hinter sich her.
Am Ende werden die Bälle durch Luftwiderstand + unelastische Stöße langsamer und kommen unten zur Ruhe.
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
from .sim import ARENA, DT, FPS, SUB, Ball, Sim, collide_pairs, hue_rgb

STYLES = ["ribbon", "dots", "rings", "helix", "zigzag", "stars", "squares", "beads"]

# Kreis exakt mittig; Radius so, dass rechts die TikTok-Buttons frei bleiben.
LAYOUTS = {
    "vertical": dict(W=1080, H=1920, arena=(540, 960, 395)),
    "landscape": dict(W=1920, H=1080, arena=(960, 540, 440)),
}


# ------------------------------------------------------------------ Simulation

class FlowSim(Sim):
    def __init__(self, seed, n_start=2, n_max=9, spawn_every=4.5, slow_at=44.0, speed=(700, 1250), r=15.0,
                 jitter=0.6, styles=STYLES, speedup=0.0):
        super().__init__(seed)
        self.n_max, self.spawn_every, self.slow_at = n_max, spawn_every, slow_at
        self.speed, self.r, self.jitter, self.styles, self.speedup = speed, r, jitter, styles, speedup
        self.balls = []
        self.next_spawn = spawn_every
        self.hue0 = self.rng.random()
        self.paths = []
        self.last_clack = -1.0
        for _ in range(n_start):
            self.spawn(initial=True)

    def spawn(self, initial=False):
        rng = self.rng
        i = len(self.balls)
        a = rng.uniform(0, 2 * math.pi)
        sp = rng.uniform(*self.speed)
        off = rng.uniform(0, 80) if initial else 0
        b = Ball(math.cos(a + 1) * off, math.sin(a + 1) * off, math.cos(a) * sp, math.sin(a) * sp, self.r)
        b.g = rng.uniform(550, 1450)  # eigene Schwerkraft je Ball -> unterschiedliche Sprungbögen
        b.hue = (self.hue0 + i * 0.13) % 1
        b.style = self.styles[i % len(self.styles)]
        b.path = []
        self.balls.append(b)
        if not initial:
            self.ev("spawn", ball=i)

    def slow(self):
        return min(1.0, max(0.0, (self.t - self.slow_at) / 4.0))

    def step(self, dt):
        rng = self.rng
        k = self.slow()
        if (self.t >= self.next_spawn and len(self.balls) < self.n_max and self.t < self.slow_at - 6):
            self.spawn()
            self.next_spawn += self.spawn_every
        boost = 1 + self.speedup * min(1.0, self.t / max(self.slow_at, 1))
        for b in self.balls:
            b.vy += b.g * dt
            if k > 0:
                f = 1 - (0.3 + 2.6 * k) * dt
                b.vx *= f
                b.vy *= f
            b.x += b.vx * dt
            b.y += b.vy * dt
        for i, j in collide_pairs(self.balls):
            a, c = self.balls[i], self.balls[j]
            rel = math.hypot(a.vx - c.vx, a.vy - c.vy)
            if rel > 250 and self.t - self.last_clack > 0.08:
                self.last_clack = self.t
                self.ev("clack", strength=min(1.0, rel / 1500))
        R = ARENA
        for i, b in enumerate(self.balls):
            d = math.hypot(b.x, b.y) or 1e-6
            if d + b.r <= R:
                continue
            nx, ny = b.x / d, b.y / d
            b.x, b.y = nx * (R - b.r), ny * (R - b.r)
            vn = b.vx * nx + b.vy * ny
            if vn <= 0:
                continue
            if k == 0:
                b.vx -= 2 * vn * nx
                b.vy -= 2 * vn * ny
                a = rng.uniform(-self.jitter, self.jitter)  # zufällige Ablenkung
                c, s = math.cos(a), math.sin(a)
                vx, vy = b.vx * c - b.vy * s, b.vx * s + b.vy * c
                if vx * nx + vy * ny > -0.25 * math.hypot(vx, vy):  # muss nach innen zeigen
                    vx, vy = b.vx, b.vy
                    if vx * nx + vy * ny > -0.25 * math.hypot(vx, vy):
                        vx, vy = vx - 0.6 * math.hypot(vx, vy) * nx, vy - 0.6 * math.hypot(vx, vy) * ny
                sp = math.hypot(vx, vy) or 1
                tgt = rng.uniform(*self.speed) * boost
                b.vx, b.vy = vx / sp * tgt, vy / sp * tgt
            else:  # Ausklang: unelastisch + Reibung
                e = 1 - 0.6 * k
                b.vx -= (1 + e) * vn * nx
                b.vy -= (1 + e) * vn * ny
                tx, ty = -ny, nx
                vt = b.vx * tx + b.vy * ty
                b.vx -= vt * tx * 0.08 * k
                b.vy -= vt * ty * 0.08 * k
            strength = min(1.0, vn / 1100)
            if vn > 70:
                b.hue = (b.hue + 0.04) % 1
                self.hits.append((self.t, nx * R, ny * R, hue_rgb(b.hue), strength))
                if self.t - b.last_note > 0.05:
                    b.last_note = self.t
                    self.ev("note", ball=i, strength=strength)
        for b in self.balls:
            b.path.append((b.x, b.y))
        if self.finish_t is None and self.t > self.slow_at + 3:
            if all(math.hypot(b.vx, b.vy) < 12 for b in self.balls) or self.t > self.slow_at + 10:
                self.finish_t = self.t

    def run(self, max_t=60.0):
        self.outro = 1.5
        while True:
            for b in self.balls:
                b.path = []
            for _ in range(SUB):
                self.step(DT)
                self.t += DT
            self.frames.append(dict(t=self.t, balls=[(b.x, b.y, b.r, b.hue, b.style, list(b.path)) for b in self.balls]))
            if self.finish_t is not None and self.t >= self.finish_t + self.outro:
                break
            if self.t > max_t:
                break
        return self


# ------------------------------------------------------------------ Klang

SR = 44100
_SCALES = {
    "bell": [329.63 * 2 ** (s / 12) for s in (0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24)],   # E-Dur pentatonisch
    "pluck": [293.66 * 2 ** (s / 12) for s in (-12, -9, -7, -5, -2, 0, 3, 5, 7, 10, 12, 15)],  # d-Moll pentatonisch
}
_cache = {}


def bell(f):
    key = ("bell", f)
    if key not in _cache:
        n = int(SR * 2.2)
        t = np.arange(n) / SR
        idx = 2.4 * np.exp(-t * 5)
        car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * 3.5 * t))
        s = car * np.exp(-t * 2.0) + 0.25 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t * 4)
        _cache[key] = s * np.minimum(t / 0.002, 1) * 0.45
    return _cache[key]


def pluck(f):
    key = ("pluck", f)
    if key not in _cache:
        n = int(SR * 1.4)
        N = max(2, int(SR / f))
        rng = np.random.default_rng(int(f))
        y = np.zeros(n)
        y[:N] = np.convolve(rng.uniform(-1, 1, N), np.ones(3) / 3, mode="same")
        for i in range(N, n):  # Karplus-Strong
            y[i] = 0.4985 * (y[i - N] + y[i - N - 1])
        t = np.arange(n) / SR
        tine = np.sin(2 * np.pi * f * t) * np.exp(-t * 3.5) * 0.5
        _cache[key] = (y * 0.8 + tine) * 0.5
    return _cache[key]


def pad(total, root, fade_from):
    n = int(SR * total)
    t = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * root * m * t + np.sin(2 * np.pi * 0.07 * t * (i + 1)))
            for i, (m, a) in enumerate(((0.5, 0.5), (0.75, 0.3), (1.0, 0.25), (1.5, 0.12))))
    s *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.11 * t)
    env = np.minimum(1, t / 1.5) * np.clip(1 - (t - fade_from) / 6, 0, 1)
    return s * env * 0.05


def build_audio(sim, instrument, path):
    total = len(sim.frames) / FPS
    rng = random.Random(sim.seed)
    voice = bell if instrument == "bell" else pluck
    scale = _SCALES[instrument]
    mix_ = np.zeros(int(SR * (total + 2.5)))
    pos = len(scale) // 2

    def put(t, s, vol):
        i = int(t * SR)
        j = min(len(mix_), i + len(s))
        mix_[i:j] += s[: j - i] * vol

    for t, kind, d in sim.events:
        if kind == "note":
            pos = min(len(scale) - 1, max(0, pos + rng.choice((-2, -1, 1, 1, 2))))  # Zufallsmelodie in der Skala
            put(t, voice(scale[pos]), 0.2 + 0.8 * d["strength"])
        elif kind == "clack":
            put(t, voice(scale[min(len(scale) - 1, pos + 3)]), 0.25 * d["strength"])
        elif kind == "spawn":
            for k, st in enumerate((0, 2, 4)):
                put(t + k * 0.06, voice(scale[st + 3]), 0.35)
    mix_ = mix_[: int(SR * total)]
    mix_ += pad(total, scale[0] / 2, sim.slow_at)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = np.tanh(mix_ / peak * 1.3) * 0.85
    st = np.stack([mix_, mix_], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ Darstellung

class FlowScene:
    def __init__(self, sim, mode, fmt):
        self.sim, self.mode = sim, mode
        L = LAYOUTS[fmt]
        self.W, self.H = L["W"], L["H"]
        self.cx, self.cy, ar = L["arena"]
        self.s = ar / ARENA
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        d = np.sqrt((xx - self.cx) ** 2 + (yy - self.cy) ** 2) / (ar * 2.0)
        self.grad = (np.clip(1 - d, 0, 1) ** 1.6)[..., None]
        self.hit_t = [h[0] for h in sim.hits]
        self.spawns = [(t, d["ball"]) for t, k, d in sim.events if k == "spawn"]
        self.paint = np.zeros((self.H, self.W, 3), np.float32) if mode == "paint" else None
        self.paint_fi = -1
        self.counter = {}

    def P(self, x, y):
        return self.cx + x * self.s, self.cy + y * self.s

    # ---- Muster

    def pattern(self, d, style, pts, hue, R, t, fade=True, base=0):
        """pts: Bildschirmpunkte alt -> neu. Zeichnet das Muster eines Balls."""
        n = len(pts)
        if n < 2:
            return
        for k in range(n):
            f = (k + 1) / n if fade else 1.0
            gk = base + k
            x, y = pts[k]
            px, py = pts[max(0, k - 1)]
            tx, ty = x - px, y - py
            L = math.hypot(tx, ty)
            if k and L < 0.8:
                continue  # ruhender Ball: kein Muster stapeln
            L = L or 1
            nx, ny = -ty / L, tx / L
            col = hue_rgb(hue + (1 - f) * 0.25)
            if style == "ribbon":
                if k:
                    d.line([px, py, x, y], fill=mix(col, f), width=max(2, int(R * 1.5 * f)))
            elif style == "dots" and gk % 3 == 0:
                r = max(1.5, R * 0.55 * f)
                d.ellipse([x - r, y - r, x + r, y + r], fill=mix(col, f))
            elif style == "rings" and gk % 6 == 0:
                r = R * (0.5 + 1.8 * (1 - f))
                d.ellipse([x - r, y - r, x + r, y + r], outline=mix(col, f), width=3)
            elif style == "helix" and k:
                ph0, ph1 = (gk - 1) * 0.45, gk * 0.45
                for sgn, hc in ((1, 0.0), (-1, 0.5)):
                    o0, o1 = math.sin(ph0) * R * 1.3 * sgn, math.sin(ph1) * R * 1.3 * sgn
                    d.line([px + nx * o0, py + ny * o0, x + nx * o1, y + ny * o1],
                           fill=mix(hue_rgb(hue + hc), f), width=3)
                if gk % 5 == 0:
                    o = math.sin(ph1) * R * 1.3
                    d.line([x + nx * o, y + ny * o, x - nx * o, y - ny * o], fill=mix((255, 255, 255), f * 0.6), width=2)
            elif style == "zigzag" and k and gk % 2 == 0:
                o = R * 1.1 * (1 if (gk // 2) % 2 else -1)
                q = pts[max(0, k - 2)]
                o_prev = -o
                d.line([q[0] + nx * o_prev, q[1] + ny * o_prev, x + nx * o, y + ny * o], fill=mix(col, f), width=3)
            elif style == "stars" and gk % 4 == 0:
                s = R * 0.8 * f
                a = gk * 0.6 + t * 3
                for aa in (a, a + math.pi / 2):
                    d.line([x - math.cos(aa) * s, y - math.sin(aa) * s, x + math.cos(aa) * s, y + math.sin(aa) * s],
                           fill=mix(col, f, 0.3), width=3)
            elif style == "squares" and gk % 5 == 0:
                s = R * 0.9 * f
                a = gk * 0.4
                q = [(x + math.cos(a + j * math.pi / 2) * s, y + math.sin(a + j * math.pi / 2) * s) for j in range(4)]
                d.polygon(q, outline=mix(col, f))
            elif style == "beads" and gk % 4 == 0:
                r = R * 0.45
                d.ellipse([x - r, y - r, x + r, y + r], outline=mix(col, f), width=2)

    def trail_points(self, fi, bi, frames_back):
        pts = []
        for j in range(max(0, fi - frames_back), fi + 1):
            balls = self.sim.frames[j]["balls"]
            if bi < len(balls):
                pts += balls[bi][5][::2]
        return [self.P(x, y) for x, y in pts]

    # ---- Effekte

    def hit_effects(self, d, t):
        import bisect
        lo = bisect.bisect_left(self.hit_t, t - 0.45)
        hi = bisect.bisect_right(self.hit_t, t)
        for i in range(max(lo, hi - 40), hi):
            t0, x, y, col, st = self.sim.hits[i]
            if st < 0.12:
                continue
            age = t - t0
            p = age / 0.45
            X, Y = self.P(x, y)
            a = math.degrees(math.atan2(y, x))
            rr = ARENA * self.s + 6
            if age < 0.25:
                d.arc([self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr], a - 10 * st, a + 10 * st,
                      fill=mix(col, (1 - age / 0.25) * st, 0.6), width=max(6, int(12 * self.s)))
            r = (6 + 60 * st * ease_out(p)) * self.s
            d.ellipse([X - r, Y - r, X + r, Y + r], outline=mix(col, (1 - p) * st, 0.3), width=max(1, int(5 * (1 - p))))
            if age < 0.3:
                rng = random.Random(i * 7919)
                base = math.atan2(-y, -x)
                for _ in range(int(3 + 7 * st)):
                    aa = base + rng.uniform(-1.3, 1.3)
                    sp = rng.uniform(250, 750) * st
                    x1, y1 = self.P(x + math.cos(aa) * sp * age, y + math.sin(aa) * sp * age)
                    x0, y0 = self.P(x + math.cos(aa) * sp * max(0, age - 0.035), y + math.sin(aa) * sp * max(0, age - 0.035))
                    d.line([x0, y0, x1, y1], fill=mix(col, 1 - age / 0.3, 0.5), width=3)

    def draw(self, d, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        R = ARENA * self.s + 6
        ring_col = hue_rgb(0.75 + 0.1 * math.sin(t * 0.25) + t * 0.01, 0.75)
        d.ellipse([self.cx - R, self.cy - R, self.cx + R, self.cy + R], outline=ring_col, width=max(6, int(8 * self.s)))
        for t0, bi in self.spawns:  # neuer Ball: sanfte Welle aus der Mitte
            age = t - t0
            if 0 <= age < 0.8:
                r = (10 + 300 * ease_out(age / 0.8)) * self.s
                d.ellipse([self.cx - r, self.cy - r, self.cx + r, self.cy + r],
                          outline=mix(hue_rgb(fr["balls"][bi][3]), 1 - age / 0.8, 0.4), width=4)
        if self.mode == "trail":
            for bi, b in enumerate(fr["balls"]):
                back = 26
                base = max(0, fi - back) * (SUB // 2)  # globaler Punktindex -> Muster bleibt am Pfad verankert
                self.pattern(d, b[4], self.trail_points(fi, bi, back), b[3], b[2] * self.s * 1.3, t, base=base)
        self.hit_effects(d, t)
        for x, y, r, hue, style, _ in fr["balls"]:
            X, Y = self.P(x, y)
            rr = r * self.s
            d.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=hue_rgb(hue, 0.7))
            d.ellipse([X - rr * 0.55, Y - rr * 0.55, X + rr * 0.55, Y + rr * 0.55], fill=(255, 255, 255))

    def update_paint(self, fi, t):
        """Malmodus: Muster wird auf eine langsam verblassende Leinwand gemalt (muss Frame für Frame laufen)."""
        sim = self.sim
        while self.paint_fi < fi:
            self.paint_fi += 1
            j = self.paint_fi
            fr = sim.frames[j]
            stamp = Image.new("RGB", (self.W, self.H), (0, 0, 0))
            d = ImageDraw.Draw(stamp)
            for bi, b in enumerate(fr["balls"]):
                prev = sim.frames[j - 1]["balls"] if j else []
                pts = ([prev[bi][5][-1]] if bi < len(prev) and prev[bi][5] else []) + b[5]
                spts = [self.P(x, y) for x, y in pts]
                self.pattern(d, b[4], spts, b[3], b[2] * self.s * 0.8, j / FPS, fade=False, base=self.counter.get(bi, 0))
                self.counter[bi] = self.counter.get(bi, 0) + len(spts)
            k = min(1.0, max(0.0, (j / FPS - sim.slow_at) / 4))
            decay = 0.985 + 0.015 * k  # im Ausklang bleibt das Bild stehen
            self.paint *= decay
            self.paint += np.asarray(stamp, np.float32) * 0.55
            np.minimum(self.paint, 255, out=self.paint)

    def frame(self, fi):
        t = fi / FPS
        geo = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        d = ImageDraw.Draw(geo)
        self.draw(d, fi, t)
        arr = np.asarray(geo, np.float32)
        if self.paint is not None:
            self.update_paint(fi, t)
            arr = np.maximum(arr, self.paint)
            geo = Image.fromarray(arr.astype(np.uint8), "RGB")
        g1 = geo.resize((self.W // 4, self.H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
        g2 = geo.resize((self.W // 10, self.H // 10), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
        glow = (np.asarray(g1.resize((self.W, self.H), Image.BILINEAR), np.float32) * 1.1 +
                np.asarray(g2.resize((self.W, self.H), Image.BILINEAR), np.float32) * 1.2)
        bgc = np.array(hue_rgb(0.7 + 0.06 * math.sin(t * 0.25), 0.7, 0.2), np.float32)
        out = self.grad * bgc + arr + glow
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def render(sim, spec, fmt, out_path, preview_dir=None, preview_times=()):
    scene = FlowScene(sim, spec["mode"], fmt)
    n = len(sim.frames)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in sorted(preview_times):
            fr = scene.frame(min(int(pt * FPS), n - 1))
            base = os.path.join(preview_dir, f"{spec['id']}_{fmt}_{pt:05.1f}")
            fr.save(base + ".png")
            tiktok_ui_mock(fr, fmt).save(base + "_ui.png")
        return
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        build_audio(sim, spec["instrument"], wav)
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
