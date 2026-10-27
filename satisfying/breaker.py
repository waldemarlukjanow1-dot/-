"""Auto-Brick-Breaker: ein Feld voller Klötze wird von Bällen aufgelöst (ohne Schläger).

Regeln
- Ein Block verschwindet nur, wenn ein Ball ihn wirklich berührt (exakte Kreis-Rechteck-Kollision, Substeps
  kleiner als der Ballradius -> kein Durchtunneln).
- Graue Stahlblöcke brauchen 3 Treffer.
- Power-up-Blöcke: gelb "+" -> 3 neue Bälle aus genau diesem Block, cyan "x" -> alle Bälle verdoppeln (gedeckelt).
- Keine Bälle entstehen zufällig. Bälle stoßen nicht miteinander zusammen (wie im Original).
- Ist das Feld leer, werden die Bälle langsamer und bleiben liegen.
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

FPS = 60
W1, H1, SS = 1080, 1920, 2
SR = 44100

# Spielfeld (Bildschirm-Pixel), rechts bleibt Platz für die TikTok-Buttons
FX0, FY0, FX1, FY1 = 40, 240, 930, 1420
CELL = 18
COLS = (FX1 - FX0) // CELL           # 49
ROWS = 52
BX0 = FX0 + ((FX1 - FX0) - COLS * CELL) // 2
BY0 = FY0

EMPTY, GREEN, STEEL, PLUS, MULT = 0, 1, 2, 3, 4
GREEN_C, GREEN_D = (46, 150, 70), (22, 70, 35)
STEEL_C = (150, 150, 158)
PLUS_C, MULT_C = (255, 210, 40), (60, 220, 240)
BG = (30, 28, 34)


class BreakerSim:
    def __init__(self, seed, n_plus=26, n_mult=5, cap=160, speed=1150.0, r=6.0, outro=3.0, n_start=2, n_low=4):
        self.rng = random.Random(seed)
        rng = self.rng
        self.kind = np.full((ROWS, COLS), GREEN, np.uint8)
        self.hp = np.ones((ROWS, COLS), np.uint8)
        # Stahlwände: ein paar waagrechte/senkrechte Linien wie im Original
        for _ in range(5):
            if rng.random() < 0.6:
                rr, c0 = rng.randrange(8, ROWS - 4), rng.randrange(2, COLS - 20)
                cs = slice(c0, c0 + rng.randrange(10, 20))
                self.kind[rr, cs] = STEEL
                self.hp[rr, cs] = 3
            else:
                cc, r0 = rng.randrange(4, COLS - 4), rng.randrange(6, ROWS - 18)
                rs = slice(r0, r0 + rng.randrange(8, 16))
                self.kind[rs, cc] = STEEL
                self.hp[rs, cc] = 3
        cells = [(r_, c_) for r_ in range(ROWS) for c_ in range(COLS) if self.kind[r_, c_] == GREEN]
        rng.shuffle(cells)
        low = [rc for rc in cells if rc[0] >= ROWS - 10]  # ein paar Power-ups in den unteren Reihen -> früher Schwung
        for r_, c_ in low[:n_low]:
            self.kind[r_, c_] = PLUS
        cells = [rc for rc in cells if self.kind[rc] == GREEN]
        for r_, c_ in cells[:n_plus]:
            self.kind[r_, c_] = PLUS
        for r_, c_ in cells[n_plus:n_plus + n_mult]:
            self.kind[r_, c_] = MULT
        self.cap, self.speed, self.r, self.outro = cap, speed, r, outro
        self.balls = []
        for j in range(n_start):  # Start: Bälle schießen von unten ins Feld
            a = math.radians(-90 + (j - (n_start - 1) / 2) * 24 + rng.uniform(-6, 6))
            self.balls.append([(FX0 + FX1) / 2 + (j - (n_start - 1) / 2) * 30, FY1 - 40.0, math.cos(a), math.sin(a)])
        self.t = 0.0
        self.events, self.frames = [], []
        self.broken = []  # (t, row, col, kind)
        self.clear_t = None
        self.finish_t = None

    def ev(self, kind, **d):
        self.events.append((self.t, kind, d))

    def spawn(self, x, y, n):
        rng = self.rng
        for k in range(n):
            if len(self.balls) >= self.cap:
                return
            a = 2 * math.pi * (k + rng.random() * 0.5) / max(n, 1)
            self.balls.append([x, y, math.cos(a), math.sin(a)])

    def hit_brick(self, r_, c_, b):
        k = self.kind[r_, c_]
        if k == STEEL:
            self.hp[r_, c_] -= 1
            self.ev("steel", row=r_, hp=int(self.hp[r_, c_]))
            if self.hp[r_, c_] > 0:
                return
        self.kind[r_, c_] = EMPTY
        self.broken.append((self.t, r_, c_, int(k)))
        cx, cy = BX0 + (c_ + 0.5) * CELL, BY0 + (r_ + 0.5) * CELL
        if k == PLUS:
            self.spawn(cx, cy, 3)
            self.ev("plus", row=r_, n=len(self.balls))
        elif k == MULT:
            n = min(len(self.balls), self.cap - len(self.balls))
            src = [list(bb) for bb in self.balls[:n]]
            for bb in src:  # jeder vorhandene Ball bekommt einen Zwilling in Spiegel-Richtung
                self.balls.append([bb[0], bb[1], -bb[2], bb[3]])
            self.ev("mult", row=r_, n=len(self.balls))
        else:
            self.ev("break", row=r_, col=c_)

    def step_ball(self, b, dist):
        r = self.r
        b[0] += b[2] * dist
        b[1] += b[3] * dist
        # Wände
        if b[0] < FX0 + r and b[2] < 0 or b[0] > FX1 - r and b[2] > 0:
            b[2] = -b[2]
            b[0] = min(max(b[0], FX0 + r), FX1 - r)
            self.ev("wall", strength=0.3)
        if b[1] < FY0 + r and b[3] < 0 or b[1] > FY1 - r and b[3] > 0:
            b[3] = -b[3]
            b[1] = min(max(b[1], FY0 + r), FY1 - r)
            self.ev("wall", strength=0.3)
        # Blöcke: nur die Zellen unter dem Ball prüfen, exakte Kreis-Rechteck-Berührung
        c0, c1 = int((b[0] - r - BX0) // CELL), int((b[0] + r - BX0) // CELL)
        r0, r1 = int((b[1] - r - BY0) // CELL), int((b[1] + r - BY0) // CELL)
        nx_sum = ny_sum = 0.0
        touched = []
        for rr in range(max(0, r0), min(ROWS - 1, r1) + 1):
            for cc in range(max(0, c0), min(COLS - 1, c1) + 1):
                if self.kind[rr, cc] == EMPTY:
                    continue
                x0, y0 = BX0 + cc * CELL, BY0 + rr * CELL
                px = min(max(b[0], x0), x0 + CELL)
                py = min(max(b[1], y0), y0 + CELL)
                dx, dy = b[0] - px, b[1] - py
                d2 = dx * dx + dy * dy
                if d2 >= r * r:
                    continue
                if d2 > 1e-9:
                    d = math.sqrt(d2)
                    nx, ny = dx / d, dy / d
                    pen = r - d
                else:  # Mittelpunkt im Block: kürzeste Achse nach außen
                    opts = [(b[0] - x0, -1, 0), (x0 + CELL - b[0], 1, 0), (b[1] - y0, 0, -1), (y0 + CELL - b[1], 0, 1)]
                    pen0, nx, ny = min(opts)
                    pen = pen0 + r
                b[0] += nx * pen
                b[1] += ny * pen
                nx_sum += nx
                ny_sum += ny
                touched.append((rr, cc))
        if touched:
            L = math.hypot(nx_sum, ny_sum)
            if L > 1e-9:
                nx, ny = nx_sum / L, ny_sum / L
                vn = b[2] * nx + b[3] * ny
                if vn < 0:
                    b[2] -= 2 * vn * nx
                    b[3] -= 2 * vn * ny
            for rr, cc in touched:
                self.hit_brick(rr, cc, b)
        # flache Winkel vermeiden (sonst ewiges Hin und Her)
        if abs(b[3]) < 0.18:
            b[3] = math.copysign(0.18, b[3] if b[3] else 1.0)
            L = math.hypot(b[2], b[3])
            b[2], b[3] = b[2] / L, b[3] / L
        if abs(b[2]) < 0.12:
            b[2] = math.copysign(0.12, b[2] if b[2] else 1.0)
            L = math.hypot(b[2], b[3])
            b[2], b[3] = b[2] / L, b[3] / L

    def run(self, max_t=75.0):
        frame_dt = 1 / FPS
        while True:
            sp = self.speed
            if self.clear_t is not None:  # Ausklang: langsamer werden
                sp *= max(0.0, 1 - (self.t - self.clear_t) / 2.5) ** 1.3
            dist = sp * frame_dt
            n_sub = max(1, int(math.ceil(dist / (self.r * 0.7))))
            for _ in range(n_sub):
                for b in list(self.balls):
                    self.step_ball(b, dist / n_sub)
                self.t += frame_dt / n_sub
            if self.clear_t is None and not (self.kind != EMPTY).any():
                self.clear_t = self.t
                self.finish_t = self.t + 2.5
                self.ev("clear")
            self.frames.append((self.kind.copy(), self.hp.copy(), np.array([(b[0], b[1]) for b in self.balls], np.float32)))
            if self.finish_t is not None and self.t >= self.finish_t + self.outro:
                break
            if self.t > max_t:
                break
        self.total = len(self.frames) / FPS
        return self


# ------------------------------------------------------------------ Darstellung

def _tile(color, dark, size):
    t = np.zeros((size, size, 3), np.uint8)
    t[:] = dark
    b = max(1, size // 9)
    t[b:size - b, b:size - b] = color
    hl = tuple(min(255, v + 35) for v in color)
    t[b:b + max(1, size // 10), b:size - b] = hl
    return t


class BreakerRenderer:
    def __init__(self, sim):
        self.sim = sim
        s = CELL * SS
        self.tiles = {GREEN: _tile(GREEN_C, GREEN_D, s), STEEL: _tile(STEEL_C, (70, 70, 78), s),
                      PLUS: _tile(PLUS_C, (150, 110, 10), s), MULT: _tile(MULT_C, (20, 110, 130), s)}
        self.broken_t = [b[0] for b in sim.broken]

    def frame(self, fi):
        sim = self.sim
        kind, hp, balls = sim.frames[fi]
        t = (fi + 1) / FPS
        k = SS
        img = np.zeros((H1 * k, W1 * k, 3), np.uint8)
        img[:] = (12, 12, 16)
        img[FY0 * k:FY1 * k, FX0 * k:FX1 * k] = BG
        s = CELL * k
        # Blöcke als Kacheln (schnell per numpy)
        field = np.zeros((ROWS * s, COLS * s, 3), np.uint8)
        field[:] = BG
        for kd, tile in self.tiles.items():
            mask = kind == kd
            if not mask.any():
                continue
            big = np.kron(mask, np.ones((s, s), bool))
            field[big] = np.tile(tile, (ROWS, COLS, 1))[big]
        img[BY0 * k:BY0 * k + ROWS * s, BX0 * k:BX0 * k + COLS * s] = field
        im = Image.fromarray(img, "RGB")
        d = ImageDraw.Draw(im)
        # Symbole auf Power-ups, Risse im Stahl
        for rr, cc in np.argwhere(kind >= STEEL):
            x0, y0 = (BX0 + cc * CELL) * k, (BY0 + rr * CELL) * k
            cx, cy = x0 + s / 2, y0 + s / 2
            kd = kind[rr, cc]
            if kd == PLUS:
                d.line([cx - s * 0.28, cy, cx + s * 0.28, cy], fill=(255, 255, 255), width=int(s * 0.16))
                d.line([cx, cy - s * 0.28, cx, cy + s * 0.28], fill=(255, 255, 255), width=int(s * 0.16))
            elif kd == MULT:
                d.line([cx - s * 0.25, cy - s * 0.25, cx + s * 0.25, cy + s * 0.25], fill=(255, 255, 255), width=int(s * 0.15))
                d.line([cx - s * 0.25, cy + s * 0.25, cx + s * 0.25, cy - s * 0.25], fill=(255, 255, 255), width=int(s * 0.15))
            elif kd == STEEL and hp[rr, cc] < 3:
                d.line([x0 + s * 0.2, y0 + s * 0.3, cx, cy, x0 + s * 0.75, y0 + s * 0.8], fill=(60, 60, 66), width=k)
                if hp[rr, cc] < 2:
                    d.line([x0 + s * 0.7, y0 + s * 0.2, cx, cy, x0 + s * 0.3, y0 + s * 0.85], fill=(60, 60, 66), width=k)
        # zerbrechende Blöcke: kurz aufhellen und schrumpfen
        import bisect
        lo = bisect.bisect_left(self.broken_t, t - 0.2)
        hi = bisect.bisect_right(self.broken_t, t)
        cols = {GREEN: GREEN_C, STEEL: STEEL_C, PLUS: PLUS_C, MULT: MULT_C}
        for t0, rr, cc, kd in sim.broken[lo:hi]:
            e = (t - t0) / 0.2
            x0, y0 = (BX0 + cc * CELL) * k, (BY0 + rr * CELL) * k
            m = s * 0.5 * e
            c = tuple(int(v + (255 - v) * (1 - e) * 0.6) for v in cols.get(kd, GREEN_C))
            d.rectangle([x0 + m, y0 + m, x0 + s - m, y0 + s - m], fill=c)
            if kd in (PLUS, MULT) and t - t0 < 0.2:
                pass
        # Power-up-Ringe (länger sichtbar)
        lo2 = bisect.bisect_left(self.broken_t, t - 0.6)
        for t0, rr, cc, kd in sim.broken[lo2:hi]:
            if kd in (PLUS, MULT):
                e = (t - t0) / 0.6
                cx, cy = (BX0 + (cc + 0.5) * CELL) * k, (BY0 + (rr + 0.5) * CELL) * k
                R = (10 + 70 * e) * k
                col = PLUS_C if kd == PLUS else MULT_C
                d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=tuple(int(v * (1 - e)) for v in col), width=3 * k)
        r = sim.r * k
        for x, y in balls:
            X, Y = x * k, y * k
            d.ellipse([X - r, Y - r, X + r, Y + r], fill=(250, 250, 250))
        return im.reduce(SS)


# ------------------------------------------------------------------ Klang (ASMR)

_cache = {}


def tick(f, dull=False):
    key = (round(f, 1), dull)
    if key not in _cache:
        rng = np.random.default_rng(int(f))
        n = int(SR * 0.2)
        t = np.arange(n) / SR
        click = np.convolve(rng.standard_normal(n), np.ones(4) / 4, mode="same") * np.exp(-t * 300)
        dec = 45 if dull else 30
        body = np.sin(2 * np.pi * f * t) * np.exp(-t * dec) + 0.25 * np.sin(2 * np.pi * f * 2.4 * t) * np.exp(-t * 70)
        _cache[key] = (click * (0.8 if dull else 0.5) + body * 0.6) * 0.35
    return _cache[key]


def chime(f):
    key = ("c", round(f, 1))
    if key not in _cache:
        n = int(SR * 1.6)
        t = np.arange(n) / SR
        s = np.sin(2 * np.pi * f * t) * np.exp(-t * 2.5) + 0.3 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 5)
        _cache[key] = s * np.minimum(t / 0.004, 1) * 0.3
    return _cache[key]


def build_audio(sim, path):
    total = sim.total
    n = int(SR * total)
    out = np.zeros(n + SR * 3)
    penta = [0, 2, 4, 7, 9]
    last = -1.0

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    for t, kind, d in sim.events:
        if kind == "break":
            if t - last < 0.03:  # höchstens ~33 Klicks pro Sekunde
                continue
            last = t
            row = ROWS - 1 - d["row"]
            semi = penta[row % 5] + 12 * ((row // 5) % 3)
            put(t, tick(392.0 * 2 ** (semi / 12)), 0.5)
        elif kind == "steel":
            put(t, tick(180.0, dull=True), 0.5)
        elif kind == "plus":
            for j, f in enumerate((659.25, 783.99, 1046.5)):
                put(t + j * 0.06, chime(f), 0.45)
        elif kind == "mult":
            for j, f in enumerate((523.25, 659.25, 783.99, 1046.5, 1318.5)):
                put(t + j * 0.05, chime(f), 0.5)
        elif kind == "clear":
            for j, f in enumerate((261.63, 329.63, 392.0, 523.25, 659.25)):
                put(t + j * 0.12, chime(f), 0.6)
    out = out[:n]
    mix_ = out * 0.9 + reverb(out, 1.0) * 0.15
    mix_ *= np.clip((total - np.arange(n) / SR) / 1.2, 0, 1)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = np.tanh(mix_ / peak * 1.3) * 0.9
    st = np.stack([mix_, mix_], axis=1)
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


def render(sim, name, out_path, preview_dir=None, preview_times=(), jobs=4):
    global _R
    _R = BreakerRenderer(sim)
    n = len(sim.frames)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fr = _R.frame(min(int(pt * FPS), n - 1))
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
        build_audio(sim, wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


SEED = 14
BREAKER = {"k01_block_breaker": lambda: BreakerSim(SEED).run()}

if __name__ == "__main__":
    import sys
    preview = "--preview" in sys.argv
    for name, mk in BREAKER.items():
        sim = mk()
        if preview:
            ts = [0.0, 3.0, 8.0, 14.0, 20.0, 26.0, 32.0, 38.0, 44.0, sim.total - 3, sim.total - 0.1]
            render(sim, name, None, preview_dir="output/preview", preview_times=ts)
            print(name, "Vorschau", round(sim.total, 1), "clear", sim.clear_t)
        else:
            out = f"output/{name}_vertical.mp4"
            render(sim, name, out)
            print(out, round(sim.total, 1))
