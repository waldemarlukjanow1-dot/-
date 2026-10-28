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
FX0, FY0, FX1, FY1 = 100, 290, 940, 1460
CELL = 22
COLS = (FX1 - FX0) // CELL           # 49
ROWS = 43
BX0 = FX0 + ((FX1 - FX0) - COLS * CELL) // 2
BY0 = FY0

EMPTY, GREEN, STEEL, PLUS, MULT, WALL = 0, 1, 2, 3, 4, 5
WALL_C, WALL_D = (128, 128, 136), (70, 70, 78)
# unzerstörbare Wände wie im Vorbild: waagrechte Lagen mit schmaler Lücke + Korridor-Stücke
WALL_ROWS = [(34, 30, 36), (25, 3, 9), (16, 17, 23), (7, 31, 37)]  # (Reihe, Lücke von, Lücke bis)
WALL_SEGS = [("v", 29, 29, 34), ("v", 37, 27, 34), ("v", 10, 20, 25), ("v", 3, 18, 25), ("v", 17, 11, 16),
             ("v", 24, 11, 16), ("h", 28, 18, 26)]
GREEN_C, GREEN_D = (46, 150, 70), (22, 70, 35)
STEEL_C = (150, 150, 158)
PLUS_C, MULT_C = (255, 210, 40), (60, 220, 240)
BG = (30, 28, 34)


class BreakerSim:
    def __init__(self, seed, cap=220, speed=1900.0, r=7.0, outro=3.0, n_start=3, n_low=4):
        self.rng = random.Random(seed)
        rng = self.rng
        self.kind = np.full((ROWS, COLS), GREEN, np.uint8)
        self.hp = np.ones((ROWS, COLS), np.uint8)
        for row, g0, g1 in WALL_ROWS:
            for c_ in range(COLS):
                if not (g0 <= c_ < g1):
                    self.kind[row, c_] = WALL
        for kd, a, b0, b1 in WALL_SEGS:
            for k_ in range(b0, b1):
                if kd == "v":
                    self.kind[k_, a] = WALL
                else:
                    self.kind[a, k_] = WALL
        # Korridor hinter jeder Lücke (wie im Vorbild schon frei) + Power-ups am Korridor-Ende = Durchbruch-Moment
        for row, g0, g1 in WALL_ROWS:
            mid = (g0 + g1) // 2
            top = max(0, row - 5)
            for r_ in range(top, row):
                for c_ in range(g0, g1):
                    if self.kind[r_, c_] == GREEN:
                        self.kind[r_, c_] = EMPTY
            for dc, kd in ((-1, PLUS), (0, MULT), (1, PLUS)):
                c_ = mid + dc * 2
                if 0 <= top - 1 < ROWS and 0 <= c_ < COLS and self.kind[top - 1, c_] == GREEN:
                    self.kind[top - 1, c_] = kd
        # Power-ups je Abteil (zwischen den Wandlagen): unten wenige, oben belohnen sie den Durchbruch
        bounds = [ROWS] + [w[0] for w in WALL_ROWS] + [-1]
        plan = [(n_low, 1), (2, 0), (2, 0), (2, 0), (1, 0)]
        for (hi, lo), (np_, nm) in zip(zip(bounds, bounds[1:]), plan):
            cells = [(r_, c_) for r_ in range(lo + 1, hi) for c_ in range(COLS) if self.kind[r_, c_] == GREEN]
            rng.shuffle(cells)
            for r_, c_ in cells[:np_]:
                self.kind[r_, c_] = PLUS
            for r_, c_ in cells[np_:np_ + nm]:
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
                if self.kind[rr, cc] != WALL:
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
            ramp = getattr(self, "ramp", None)
            if ramp:  # langsam starten, mit der Zeit schneller
                v0, v1, tr = ramp
                p_ = min(1.0, self.t / tr)
                sp = v0 + (v1 - v0) * p_ * p_ * (3 - 2 * p_)
            if self.clear_t is not None:  # Ausklang: langsamer werden
                sp *= max(0.0, 1 - (self.t - self.clear_t) / 2.5) ** 1.3
            dist = sp * frame_dt
            n_sub = max(1, int(math.ceil(dist / (self.r * 0.7))))
            for _ in range(n_sub):
                for b in list(self.balls):
                    self.step_ball(b, dist / n_sub)
                self.t += frame_dt / n_sub
            if self.clear_t is None and not ((self.kind != EMPTY) & (self.kind != WALL)).any():
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
                      PLUS: _tile(PLUS_C, (150, 110, 10), s), MULT: _tile(MULT_C, (20, 110, 130), s),
                      WALL: _tile(WALL_C, WALL_D, s)}
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
        for rr, cc in np.argwhere((kind >= STEEL) & (kind != WALL)):
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


# ------------------------------------------------------------------ Klang: nur leichtes Ploppen

_cache = {}


def plop(f, big=False):
    """Weiches Plopp wie eine kleine Blase: kurzer Sinus mit schnellem Tonhöhen-Sprung, ohne Melodie."""
    key = (round(f), big)
    if key not in _cache:
        n = int(SR * (0.16 if big else 0.09))
        t = np.arange(n) / SR
        fr = f * (0.7 + 0.6 * (1 - np.exp(-t * (60 if big else 90))))
        s = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * (30 if big else 55))
        s *= np.minimum(t / 0.002, 1)
        _cache[key] = s * (0.5 if big else 0.32)
    return _cache[key]


def build_audio(sim, path):
    total = sim.total
    n = int(SR * total)
    out = np.zeros(n + SR * 2)
    rng = random.Random(5)
    last = -1.0

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    for t, kind, d in sim.events:
        if kind == "break":
            if t - last < 0.028:
                continue
            last = t
            put(t, plop(rng.choice((520, 560, 600, 640, 680))), 0.6)
        elif kind in ("plus", "mult"):
            put(t, plop(330, big=True), 0.9)
    for t, x, y, st in getattr(sim, "waves", []):  # Welle: tiefes, weiches Plopp
        put(t, plop(210, big=True), 0.8 * st)
    out = out[:n]
    mix_ = out + reverb(out, 0.6) * 0.08
    mix_ *= np.clip((total - np.arange(n) / SR) / 1.2, 0, 1)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = np.tanh(mix_ / peak * 1.2) * 0.9
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
    _R = (FxRenderer(sim) if isinstance(sim, SymbolBreakerSim) else ViceRenderer(sim) if isinstance(sim, ViceBreakerSim) else GlossyRenderer(sim) if isinstance(sim, SlowHeartSim)
          else NeonRenderer(sim) if isinstance(sim, HeartBreakerSim) else BreakerRenderer(sim))
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


# ------------------------------------------------------------------ Neon-Herz-Level

NEON = dict(block=(168, 70, 255), block_d=(58, 16, 105), bg=(14, 8, 26), wall=(60, 240, 255), wall_d=(10, 120, 150),
            plus=(255, 205, 60), mult=(110, 255, 140))


class HeartBreakerSim(BreakerSim):
    """Gleiche Physik, aber die Wände bilden ein Symbol: ein großes Herz (Eingang an der Spitze) mit einer
    Raute im Inneren (Eingang oben). Hinter jedem Eingang ein kurzer Korridor mit Power-ups."""

    def __init__(self, seed, speed=1500.0, n_outer_plus=8, cap=260, **kw):
        kw["cap"] = cap
        super().__init__(seed, speed=speed, **kw)
        rng = random.Random(seed + 99)
        self.kind[:] = GREEN
        self.hp[:] = 1
        cx, cy = COLS / 2 - 0.5, 21.0
        inside = np.zeros((ROWS, COLS), bool)
        for r_ in range(ROWS):
            for c_ in range(COLS):
                x, y = (c_ - cx) / 13.0, -(r_ - cy) / 12.0
                inside[r_, c_] = (x * x + y * y - 1) ** 3 - x * x * y ** 3 <= 0
        outline = inside & ~(np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1))
        self.kind[outline] = WALL
        # Eingang an der Herzspitze
        tip_r = max(r_ for r_ in range(ROWS) if inside[r_].any())
        cmid = int(round(cx))
        for r_ in range(tip_r - 2, tip_r + 1):
            for c_ in range(cmid - 2, cmid + 3):
                if self.kind[r_, c_] == WALL:
                    self.kind[r_, c_] = GREEN
        for r_ in range(tip_r - 7, tip_r - 1):  # Korridor ins Herz
            for c_ in range(cmid - 1, cmid + 2):
                if inside[r_, c_] and self.kind[r_, c_] == GREEN:
                    self.kind[r_, c_] = EMPTY
        for dc, kd in ((-2, PLUS), (0, PLUS), (2, PLUS)):
            self.kind[tip_r - 8, cmid + dc] = kd
        # Raute im Herzen, Eingang oben
        dr, dcy = 5, 20
        for r_ in range(ROWS):
            for c_ in range(COLS):
                if abs(r_ - dcy) + abs(c_ - cx) in (dr - 0.5, dr + 0.5) or abs(abs(r_ - dcy) + abs(c_ - cx) - dr) < 0.6:
                    self.kind[r_, c_] = WALL
        for r_ in range(dcy - dr - 1, dcy - dr + 2):
            for c_ in range(cmid - 1, cmid + 2):
                if self.kind[r_, c_] == WALL:
                    self.kind[r_, c_] = GREEN
        self.kind[dcy, cmid] = MULT
        self.kind[dcy, cmid - 2] = PLUS
        self.kind[dcy, cmid + 2] = PLUS
        # Power-ups außerhalb des Herzens (unten zuerst) + ein paar im Herzen
        out_cells = [(r_, c_) for r_ in range(ROWS) for c_ in range(COLS) if self.kind[r_, c_] == GREEN and not inside[r_, c_]]
        rng.shuffle(out_cells)
        low = [rc for rc in out_cells if rc[0] >= ROWS - 8][:3]
        for rc in low + out_cells[:n_outer_plus]:
            self.kind[rc] = PLUS
        top = [rc for rc in out_cells if rc[0] < 9 and self.kind[rc] == GREEN]
        for rc in top[:2]:
            self.kind[rc] = MULT
        for rc in top[2:5]:
            self.kind[rc] = PLUS
        in_cells = [(r_, c_) for r_ in range(ROWS) for c_ in range(COLS) if self.kind[r_, c_] == GREEN and inside[r_, c_]
                    and abs(r_ - dcy) + abs(c_ - cx) > dr + 1]
        rng.shuffle(in_cells)
        for rc in in_cells[:4]:
            self.kind[rc] = PLUS


class NeonRenderer(BreakerRenderer):
    def __init__(self, sim):
        super().__init__(sim)
        s = CELL * SS
        self.tiles = {GREEN: _tile(NEON["block"], NEON["block_d"], s), PLUS: _tile(NEON["plus"], (140, 100, 10), s),
                      MULT: _tile(NEON["mult"], (10, 100, 130), s), WALL: _tile(NEON["wall"], NEON["wall_d"], s)}

    def frame(self, fi):
        global BG, GREEN_C, PLUS_C, MULT_C
        BG, GREEN_C, PLUS_C, MULT_C = NEON["bg"], NEON["block"], NEON["plus"], NEON["mult"]
        im = super().frame(fi)
        # Neon-Glow
        from PIL import ImageFilter
        arr = np.asarray(im, np.float32)
        g = im.resize((W1 // 5, H1 // 5), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
        glow = np.asarray(g.resize((W1, H1), Image.BILINEAR), np.float32)
        arr = arr + np.clip(glow - 30, 0, 255) * 0.55
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")



# ------------------------------------------------------------------ Glänzende Blöcke + Licht-Reflexion

def glossy_tile(color, size, bg, rim=(255, 255, 255), neon=False):
    """Bonbon-artiger Block: abgerundet, Verlauf, Fase, Glanz oben, Glanzpunkt. 4x gerechnet, dann verkleinert."""
    S = size * 4
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) + 0.5
    m = S * 0.07                      # Fuge
    rad = S * 0.22                    # Eckenradius
    x0, y0, x1, y1 = m, m, S - m, S - m
    cx = np.clip(xx, x0 + rad, x1 - rad)
    cy = np.clip(yy, y0 + rad, y1 - rad)
    dist = np.hypot(xx - cx, yy - cy) - rad          # <0 innen
    alpha = np.clip(-dist / 3.0 + 0.5, 0, 1)
    u = (xx - x0) / (x1 - x0)
    v = (yy - y0) / (y1 - y0)
    c = np.array(color, np.float32)
    base = c * (1.18 - 0.42 * v)[..., None]
    edge = np.clip(-dist / (S * 0.12), 0, 1)          # 0 am Rand -> 1 innen
    bevel = (1 - edge) * (0.55 - u - v)               # links/oben hell, rechts/unten dunkel
    col = base + bevel[..., None] * 140
    gloss = np.clip(1 - v / 0.48, 0, 1) ** 1.6 * (edge > 0.25) * 0.38    # Glanz auf der oberen Hälfte
    col = col + gloss[..., None] * (255 - col)
    spec = np.exp(-(((u - 0.27) / 0.1) ** 2 + ((v - 0.2) / 0.07) ** 2)) * 0.85  # Glanzpunkt
    col = col + spec[..., None] * (255 - col)
    rimline = np.exp(-((dist + S * 0.025) / (S * 0.02)) ** 2) * 0.35        # feine helle Kante
    col = col + rimline[..., None] * (np.array(rim, np.float32) - col)
    if neon:
        core = np.exp(-(((v - 0.5) / 0.22) ** 2)) * 0.35
        col = col + core[..., None] * (255 - col)
    bgc = np.array(bg, np.float32)
    out = bgc + (np.clip(col, 0, 255) - bgc) * alpha[..., None]
    im = Image.fromarray(out.astype(np.uint8), "RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(im)


class GlossyRenderer(BreakerRenderer):
    def __init__(self, sim):
        super().__init__(sim)
        s = CELL * SS
        bg = NEON["bg"]
        self.tiles = {GREEN: glossy_tile(NEON["block"], s, bg), PLUS: glossy_tile(NEON["plus"], s, bg),
                      MULT: glossy_tile(NEON["mult"], s, bg), WALL: glossy_tile(NEON["wall"], s, bg, neon=True)}
        self.solid = {k: np.any(np.abs(t.astype(int) - np.array(bg)) > 12, axis=2) for k, t in self.tiles.items()}
        fy, fx = np.mgrid[0:ROWS * s, 0:COLS * s].astype(np.float32)
        self.diag = (fx + fy * 0.55) / s  # für die wandernde Licht-Reflexion
        self.ball_spr = self._ball_sprite(int(sim.r * SS))

    def _ball_sprite(self, r):
        n = r * 6
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2
        d = np.hypot(xx, yy)
        glow = np.clip(1 - d / (r * 2.6), 0, 1) ** 2 * 0.55
        core = np.clip(r - d + 0.5, 0, 1)
        hl = np.exp(-(((xx + r * 0.35) / (r * 0.35)) ** 2 + ((yy + r * 0.35) / (r * 0.35)) ** 2))
        a = np.maximum(glow, core)
        rgb = np.zeros((n, n, 3), np.float32)
        rgb[:] = (235, 225, 255)
        rgb = rgb * (1 - 0.25 * core[..., None] * (yy / r).clip(0, 1)[..., None]) + hl[..., None] * 40
        return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA")

    def frame(self, fi):
        global BG, GREEN_C, PLUS_C, MULT_C
        BG, GREEN_C, PLUS_C, MULT_C = NEON["bg"], NEON["block"], NEON["plus"], NEON["mult"]
        sim = self.sim
        kind, hp, balls = sim.frames[fi]
        t = (fi + 1) / FPS
        k = SS
        s = CELL * k
        img = np.zeros((H1 * k, W1 * k, 3), np.uint8)
        img[:] = (8, 5, 16)
        img[FY0 * k:FY1 * k, FX0 * k:FX1 * k] = NEON["bg"]
        field = np.zeros((ROWS * s, COLS * s, 3), np.uint8)
        field[:] = NEON["bg"]
        solid = np.zeros((ROWS * s, COLS * s), bool)
        for kd, tile in self.tiles.items():
            mask = kind == kd
            if not mask.any():
                continue
            big = np.kron(mask, np.ones((s, s), bool))
            field[big] = np.tile(tile, (ROWS, COLS, 1))[big]
            solid |= big & np.tile(self.solid[kd], (ROWS, COLS))
        # Licht-Reflexion: diagonales Glanzband wandert alle 5 s über das Feld
        pos = ((t % 5.0) / 5.0) * (COLS + ROWS * 0.55 + 16) - 8
        band = np.exp(-((self.diag - pos) / 2.2) ** 2) * 0.32
        f = field.astype(np.float32)
        f += (band * solid)[..., None] * (255 - f)
        field = f.astype(np.uint8)
        img[BY0 * k:BY0 * k + ROWS * s, BX0 * k:BX0 * k + COLS * s] = field
        im = Image.fromarray(img, "RGB")
        d = ImageDraw.Draw(im)
        for rr, cc in np.argwhere((kind == PLUS) | (kind == MULT)):
            x0, y0 = (BX0 + cc * CELL) * k, (BY0 + rr * CELL) * k
            cx, cy = x0 + s / 2, y0 + s / 2 + s * 0.03
            w = int(s * 0.13)
            ic = (90, 50, 0) if kind[rr, cc] == PLUS else (0, 80, 40)
            if kind[rr, cc] == PLUS:
                d.line([cx - s * 0.22, cy, cx + s * 0.22, cy], fill=ic, width=w)
                d.line([cx, cy - s * 0.22, cx, cy + s * 0.22], fill=ic, width=w)
            else:
                d.line([cx - s * 0.19, cy - s * 0.19, cx + s * 0.19, cy + s * 0.19], fill=ic, width=w)
                d.line([cx - s * 0.19, cy + s * 0.19, cx + s * 0.19, cy - s * 0.19], fill=ic, width=w)
        import bisect
        lo = bisect.bisect_left(self.broken_t, t - 0.22)
        hi = bisect.bisect_right(self.broken_t, t)
        cols = {GREEN: NEON["block"], PLUS: NEON["plus"], MULT: NEON["mult"]}
        for t0, rr, cc, kd in sim.broken[lo:hi]:  # Block zerplatzt: hell aufleuchten, schrumpfen
            e = (t - t0) / 0.22
            x0, y0 = (BX0 + cc * CELL) * k, (BY0 + rr * CELL) * k
            m = s * 0.5 * e
            c = tuple(int(v + (255 - v) * (1 - e) * 0.7) for v in cols.get(kd, NEON["block"]))
            d.rounded_rectangle([x0 + m, y0 + m, x0 + s - m, y0 + s - m], radius=max(1, int(s * 0.2 * (1 - e))), fill=c)
        lo2 = bisect.bisect_left(self.broken_t, t - 0.6)
        for t0, rr, cc, kd in sim.broken[lo2:hi]:
            if kd in (PLUS, MULT):
                e = (t - t0) / 0.6
                cx, cy = (BX0 + (cc + 0.5) * CELL) * k, (BY0 + (rr + 0.5) * CELL) * k
                R = (10 + 60 * e) * k
                col = NEON["plus"] if kd == PLUS else NEON["mult"]
                d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=tuple(int(v * (1 - e)) for v in col), width=3 * k)
        rgba = im.convert("RGBA")
        sp = self.ball_spr
        for x, y in balls:
            rgba.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        out = rgba.convert("RGB").reduce(SS)
        from PIL import ImageFilter
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 5, H1 // 5), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
        glow = np.asarray(g.resize((W1, H1), Image.BILINEAR), np.float32)
        arr = arr + np.clip(glow - 40, 0, 255) * 0.4
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


class SlowHeartSim(HeartBreakerSim):
    """Wie das Herz-Level, aber: Start mit 1 Ball, langsam beginnend (gut zu verfolgen), Tempo steigt mit der
    Zeit, unten nur ein Power-up -> die Ballzahl baut sich nach und nach auf."""

    def __init__(self, seed, ramp=(620.0, 1450.0, 32.0), **kw):
        kw.setdefault("n_start", 1)
        kw.setdefault("n_low", 1)
        super().__init__(seed, **kw)
        self.ramp = ramp



# ------------------------------------------------------------------ "VI"-Level im Vice-City-Look (eigenes Design)

def _poly_inside(px, py, poly):
    ins = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > py) != (y1 > py) and px < x0 + (py - y0) * (x1 - x0) / (y1 - y0):
            ins = not ins
    return ins


class ViceBreakerSim(BreakerSim):
    """Wände bilden große, hohle Buchstaben "VI" (eigene Formen). Jeder Buchstabe hat unten einen schmalen
    Eingang mit kurzem Korridor und Power-ups dahinter. Langsamer Start, Ballzahl baut sich auf."""

    def __init__(self, seed, ramp=(700.0, 1800.0, 22.0), n_outer_plus=14, cap=320, **kw):
        kw.setdefault("n_start", 1)
        kw.setdefault("n_low", 1)
        super().__init__(seed, cap=cap, **kw)
        self.ramp = ramp
        rng = random.Random(seed + 7)
        self.kind[:] = GREEN
        self.hp[:] = 1
        ox, oy = (COLS - 30) // 2, 7          # Buchstaben-Bereich: 30 Spalten breit, Zeilen 7..31
        V = [(0, 0), (7, 0), (10.5, 15), (14, 0), (21, 0), (13, 24), (8, 24)]
        I = [(23, 0), (30, 0), (30, 4), (28.5, 4), (28.5, 20), (30, 20), (30, 24), (23, 24), (23, 20), (24.5, 20),
             (24.5, 4), (23, 4)]
        self.letters = []
        for poly, gap_c in ((V, 10.5), (I, 26.5)):
            pts = [(x + ox, y + oy) for x, y in poly]
            inside = np.zeros((ROWS, COLS), bool)
            for r_ in range(ROWS):
                for c_ in range(COLS):
                    inside[r_, c_] = _poly_inside(c_ + 0.5, r_ + 0.5, pts)
            nb = np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
            outline = inside & ~nb
            self.kind[outline] = WALL
            bottom = max(r_ for r_ in range(ROWS) if inside[r_].any())
            gc = int(gap_c + ox)
            for c_ in range(gc - 1, gc + 1):  # schmaler Eingang (2 Felder)
                self.kind[bottom, c_] = GREEN
            for r_ in range(bottom - 3, bottom):  # kurzer Korridor
                for c_ in range(gc - 1, gc + 1):
                    if inside[r_, c_] and self.kind[r_, c_] == GREEN:
                        self.kind[r_, c_] = EMPTY
            for c_ in (gc - 1, gc):
                if self.kind[bottom - 4, c_] != WALL:
                    self.kind[bottom - 4, c_] = PLUS
            cells = [(r_, c_) for r_ in range(ROWS) for c_ in range(COLS) if inside[r_, c_] and self.kind[r_, c_] == GREEN]
            rng.shuffle(cells)
            for rc in cells[:3]:
                self.kind[rc] = PLUS
            if cells[3:]:
                self.kind[cells[3]] = MULT
            self.letters.append(inside)
        letters = self.letters[0] | self.letters[1]
        out_cells = [(r_, c_) for r_ in range(ROWS) for c_ in range(COLS) if self.kind[r_, c_] == GREEN and not letters[r_, c_]]
        rng.shuffle(out_cells)
        for rc in [rc for rc in out_cells if rc[0] >= ROWS - 8][:3] + out_cells[:n_outer_plus]:
            self.kind[rc] = PLUS
        top = [rc for rc in out_cells if rc[0] < 7 and self.kind[rc] == GREEN]
        for rc in top[:2]:
            self.kind[rc] = MULT


VICE = dict(block=(150, 70, 255), block_d=(50, 16, 105), bg=(14, 8, 26), plus=(255, 225, 90), mult=(90, 245, 210))
SUNSET = [(255, 214, 90), (255, 160, 60), (255, 110, 80), (255, 70, 130)]


class ViceRenderer(GlossyRenderer):
    """Glanz-Look mit Sonnenuntergangs-Verlauf auf den Wänden + kleiner Ball-Zähler unten."""

    def __init__(self, sim):
        global NEON
        NEON = dict(NEON, **VICE)
        super().__init__(sim)
        s = CELL * SS
        rows = np.nonzero((sim.kind == WALL).any(axis=1))[0]
        r0, r1 = rows.min(), rows.max()
        self.wall_tiles = {}
        for r_ in range(ROWS):
            f = 0 if r1 == r0 else min(1.0, max(0.0, (r_ - r0) / (r1 - r0)))
            seg = f * (len(SUNSET) - 1)
            i = min(int(seg), len(SUNSET) - 2)
            c = tuple(int(a + (b - a) * (seg - i)) for a, b in zip(SUNSET[i], SUNSET[i + 1]))
            self.wall_tiles[r_] = glossy_tile(c, s, NEON["bg"], neon=True)
        from PIL import ImageFont
        self.font = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", 30 * SS)

    def frame(self, fi):
        # Wände je Zeile im Verlauf: dafür die Wand-Kachel pro Frame zeilenweise einsetzen
        sim = self.sim
        kind = sim.frames[fi][0]
        base_tiles = self.tiles
        self.tiles = {k: v for k, v in base_tiles.items() if k != WALL}
        im = GlossyRenderer.frame(self, fi)
        self.tiles = base_tiles
        k = SS
        s = CELL * k
        # Wände separat (in 1x) einzeichnen: Kacheln auf 1x verkleinert
        arr = np.array(im)
        s1 = CELL
        for r_, c_ in np.argwhere(kind == WALL):
            t = self.wall_tiles[r_]
            t1 = t[::SS, ::SS]
            y0, x0 = BY0 + r_ * s1, BX0 + c_ * s1
            arr[y0:y0 + s1, x0:x0 + s1] = t1[:s1, :s1]
        im = Image.fromarray(arr)
        # Neon-Glow für die Wände
        from PIL import ImageFilter
        wmask = np.zeros((H1, W1), np.float32)
        for r_, c_ in np.argwhere(kind == WALL):
            y0, x0 = BY0 + r_ * s1, BX0 + c_ * s1
            wmask[y0:y0 + s1, x0:x0 + s1] = 1
        wl = Image.fromarray((np.asarray(im, np.float32) * wmask[..., None]).astype(np.uint8))
        g = wl.resize((W1 // 4, H1 // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
        glow = np.asarray(g.resize((W1, H1), Image.BILINEAR), np.float32)
        out = np.clip(np.asarray(im, np.float32) + glow * 0.9, 0, 255).astype(np.uint8)
        im = Image.fromarray(out)
        draw_counter(im, len(sim.frames[fi][2]))
        return im


def draw_counter(im, n):
    """Kleiner Ball-Zähler unten im Feld (Randinfo)."""
    from PIL import ImageFont
    d = ImageDraw.Draw(im, "RGBA")
    txt = f"{n}"
    f = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", 30)
    tw = d.textlength(txt, font=f)
    cx, cy = (FX0 + FX1) / 2, FY1 - 34
    w = tw + 58
    d.rounded_rectangle([cx - w / 2, cy - 22, cx + w / 2, cy + 22], radius=22, fill=(0, 0, 0, 150),
                        outline=(255, 255, 255, 70), width=2)
    d.ellipse([cx - w / 2 + 14, cy - 8, cx - w / 2 + 30, cy + 8], fill=(240, 235, 255, 255))
    d.text((cx - w / 2 + 40, cy), txt, font=f, fill=(255, 255, 255, 235), anchor="lm")


# ------------------------------------------------------------------ Blitz-Level mit Effekten

class SymbolBreakerSim(BreakerSim):
    """Wände bilden ein hohles Symbol (Polygon in Zell-Koordinaten). An den Lücken-Punkten ist die Wand offen,
    dahinter ein kurzer Korridor mit Power-ups. Nach dem Lauf werden die Kettenreaktions-Wellen bestimmt."""

    SHAPE = [(17, 0), (30, 0), (22, 13), (29, 13), (11, 35), (15, 19), (8, 19)]   # Blitz
    OFFSET = (0, 4)
    GAPS = [(12.0, 22.5), (24.0, 4.5), (25.0, 17.5)]

    def __init__(self, seed, ramp=(700.0, 1800.0, 22.0), n_outer_plus=14, n_inner=(4, 1), n_low_plus=6, cap=320, **kw):
        kw.setdefault("n_start", 1)
        kw.setdefault("n_low", 1)
        super().__init__(seed, cap=cap, **kw)
        self.ramp = ramp
        rng = random.Random(seed + 7)
        self.kind[:] = GREEN
        self.hp[:] = 1
        ox, oy = self.OFFSET
        pts = [(x + ox, y + oy) for x, y in self.SHAPE]
        inside = np.array([[_poly_inside(c_ + 0.5, r_ + 0.5, pts) for c_ in range(COLS)] for r_ in range(ROWS)])
        nb = np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1)
        outline = inside & ~nb
        self.kind[outline] = WALL
        rr, cc = np.mgrid[0:ROWS, 0:COLS] + 0.5
        for gx, gy in self.GAPS:
            d = np.hypot(cc - gx, rr - gy)
            self.kind[outline & (d <= 1.1)] = GREEN          # Lücke in der Wand
            self.kind[inside & ~outline & (d <= 2.6)] = EMPTY  # kurzer Korridor dahinter
            ring = np.argwhere(inside & ~outline & (d > 2.6) & (d <= 3.8) & (self.kind == GREEN))
            for r_, c_ in ring[:2]:
                self.kind[r_, c_] = PLUS
        cells = [tuple(rc) for rc in np.argwhere(inside & (self.kind == GREEN))]
        rng.shuffle(cells)
        npl, nm = n_inner
        for rc in cells[:npl]:
            self.kind[rc] = PLUS
        for rc in cells[npl:npl + nm]:
            self.kind[rc] = MULT
        out_cells = [tuple(rc) for rc in np.argwhere(~inside & (self.kind == GREEN))]
        rng.shuffle(out_cells)
        for rc in [rc for rc in out_cells if rc[0] >= ROWS - 12][:n_low_plus] + out_cells[:n_outer_plus]:
            self.kind[rc] = PLUS
        for rc in [rc for rc in out_cells if rc[0] < 5 and self.kind[rc] == GREEN][:2]:
            self.kind[rc] = MULT
        # Zellen, die kein Ball erreichen kann (nur diagonal verbunden), gar nicht erst füllen
        reach = np.zeros((ROWS, COLS), bool)
        stack = [(ROWS - 1, c_) for c_ in range(COLS) if self.kind[ROWS - 1, c_] != WALL]
        while stack:
            r_, c_ = stack.pop()
            if reach[r_, c_]:
                continue
            reach[r_, c_] = True
            for r2, c2 in ((r_ + 1, c_), (r_ - 1, c_), (r_, c_ + 1), (r_, c_ - 1)):
                if 0 <= r2 < ROWS and 0 <= c2 < COLS and not reach[r2, c2] and self.kind[r2, c2] != WALL:
                    stack.append((r2, c2))
        self.kind[~reach & (self.kind != WALL)] = EMPTY
        self.inside = inside

    def run(self, max_t=75.0):
        super().run(max_t)
        self.waves = find_waves(self)
        return self


def find_waves(sim, win=0.35, need=14, pause=2.2):
    """Kettenreaktionen: MULT-Treffer und dichte Serien von Treffern lösen eine Licht-Welle aus.
    -> Liste (t, x, y, stark)"""
    waves = []
    bt = [b[0] for b in sim.broken]
    last = -9.0
    j = 0
    for i, (t0, r_, c_, kd) in enumerate(sim.broken):
        cx, cy = BX0 + (c_ + 0.5) * CELL, BY0 + (r_ + 0.5) * CELL
        if kd == MULT:
            waves.append((t0, cx, cy, 1.0))
            last = t0
            continue
        while bt[j] < t0 - win:
            j += 1
        if i - j + 1 >= need and t0 - last >= pause:
            grp = sim.broken[j:i + 1]
            x = sum(BX0 + (g[2] + 0.5) * CELL for g in grp) / len(grp)
            y = sum(BY0 + (g[1] + 0.5) * CELL for g in grp) / len(grp)
            waves.append((t0, x, y, 0.65))
            last = t0
    return waves


BOLT = [(250, 252, 255), (150, 235, 255), (70, 170, 255), (120, 110, 255)]
BOLT_PAL = dict(block=(160, 70, 255), bg=(12, 7, 24), plus=(255, 215, 80), mult=(80, 245, 200))


class FxRenderer(GlossyRenderer):
    """Glanz-Blöcke mit Schatten (3D), Verlaufs-Wände in voller Auflösung, weiche kurze Lichtspuren,
    Partikel + Lichtblitz beim Zerplatzen, Kettenreaktions-Wellen, Vignette, Ball-Zähler."""

    TRAIL_PX = 70      # Spurlänge in Pixeln (kurzer Kometenschweif)
    P_LIFE = 0.55      # Partikel-Lebensdauer
    W_LIFE = 1.6       # Wellen-Lebensdauer

    def __init__(self, sim, wall_cols=BOLT, pal=BOLT_PAL):
        global NEON
        NEON = dict(NEON, **pal)
        super().__init__(sim)
        k, s = SS, CELL * SS
        bg = np.array(NEON["bg"], np.float32)
        # Wand-Kacheln im Verlauf von oben nach unten
        rows = np.nonzero((sim.kind == WALL).any(axis=1))[0]
        r0, r1 = rows.min(), rows.max()
        self.wall_tiles = {}
        for r_ in range(ROWS):
            f = min(1.0, max(0.0, (r_ - r0) / max(1, r1 - r0)))
            seg = f * (len(wall_cols) - 1)
            i = min(int(seg), len(wall_cols) - 2)
            c = tuple(int(a + (b - a) * (seg - i)) for a, b in zip(wall_cols[i], wall_cols[i + 1]))
            self.wall_tiles[r_] = glossy_tile(c, s, NEON["bg"], neon=True)
        self.walls = np.argwhere(sim.kind == WALL)
        # Hintergrund: dunkle Vignette außen, im Feld sanfter Lichtschein von oben
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        vig = 0.45 + 0.55 * np.exp(-(((xx - W1 / 2) / W1) ** 2 + ((yy - H1 * 0.45) / H1) ** 2) * 3.0)
        base = np.array((8, 5, 16), np.float32) * vig[..., None]
        fy = (yy - FY0) / (FY1 - FY0)
        fx = (xx - (FX0 + FX1) / 2) / (FX1 - FX0)
        glow = np.exp(-(fx ** 2) / 0.18 - ((fy - 0.15) ** 2) / 0.3) * 0.55
        fld = bg * (0.75 + glow)[..., None]
        inf = (xx >= FX0) & (xx < FX1) & (yy >= FY0) & (yy < FY1)
        base[inf] = fld[inf]
        self.base = base.astype(np.uint8)
        fb = self.base[BY0 * k:BY0 * k + ROWS * s, BX0 * k:BX0 * k + COLS * s]
        self.field_bg = fb.copy()
        # Glow der Wände (statisch) einmal vorberechnen
        from PIL import ImageFilter
        wl = np.zeros((H1, W1, 3), np.float32)
        for r_, c_ in self.walls:
            t1 = np.asarray(Image.fromarray(self.wall_tiles[r_]).reduce(SS), np.float32)
            y0, x0 = BY0 + r_ * CELL, BX0 + c_ * CELL
            wl[y0:y0 + CELL, x0:x0 + CELL] = t1
        g = Image.fromarray(wl.astype(np.uint8)).resize((W1 // 4, H1 // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
        self.wall_glow = np.asarray(g.resize((W1, H1), Image.BILINEAR), np.float32) * 0.85
        # Partikel für jeden zerplatzenden Block vorab festlegen
        cols = {GREEN: NEON["block"], PLUS: NEON["plus"], MULT: NEON["mult"]}
        P = []
        for t0, r_, c_, kd in sim.broken:
            rng = random.Random(r_ * 1000 + c_ * 7 + int(t0 * 1000))
            cx, cy = BX0 + (c_ + 0.5) * CELL, BY0 + (r_ + 0.5) * CELL
            n = 9 if kd in (PLUS, MULT) else 6
            col = cols.get(kd, NEON["block"])
            for _ in range(n):
                a = rng.uniform(0, 2 * math.pi)
                v = rng.uniform(90, 330) * (1.4 if kd in (PLUS, MULT) else 1.0)
                P.append((t0, cx + rng.uniform(-6, 6), cy + rng.uniform(-6, 6), math.cos(a) * v, math.sin(a) * v,
                          rng.uniform(2.0, 4.2), *col))
        self.P = np.array(P, np.float32).reshape(-1, 9)
        self.waves = getattr(sim, "waves", [])
        yy, xx = np.mgrid[-24:24, -24:24].astype(np.float32)
        self.flash_spr = (np.exp(-(xx ** 2 + yy ** 2) / (2 * 9.0 ** 2)) * 0.9)[..., None] * np.array((255, 225, 255), np.float32)
        self.ccx = (BX0 + (np.arange(COLS) + 0.5) * CELL)[None, :]
        self.ccy = (BY0 + (np.arange(ROWS) + 0.5) * CELL)[:, None]

    def wave_field(self, t):
        """Helligkeit je Zelle durch laufende Wellen + Liste der sichtbaren Ringe."""
        I = np.zeros((ROWS, COLS), np.float32)
        rings = []
        for t0, x, y, st in self.waves:
            a = t - t0
            if 0 <= a < self.W_LIFE:
                R = 60 + 900 * a
                fade = (1 - a / self.W_LIFE) ** 1.5 * st
                d = np.hypot(self.ccx - x, self.ccy - y)
                I += np.exp(-((d - R) / 55) ** 2) * fade
                rings.append((x, y, R, fade))
        return np.clip(I, 0, 1), rings

    def frame(self, fi):
        global BG, GREEN_C, PLUS_C, MULT_C
        BG, GREEN_C, PLUS_C, MULT_C = NEON["bg"], NEON["block"], NEON["plus"], NEON["mult"]
        import bisect
        from PIL import ImageFilter
        sim = self.sim
        kind, hp, balls = sim.frames[fi]
        t = (fi + 1) / FPS
        k, s = SS, CELL * SS
        img = self.base.copy()
        field = self.field_bg.copy()
        solid = np.zeros((ROWS * s, COLS * s), bool)
        for kd, tile in self.tiles.items():
            if kd == WALL:
                continue
            mask = kind == kd
            if not mask.any():
                continue
            big = np.kron(mask, np.ones((s, s), bool))
            field[big] = np.tile(tile, (ROWS, COLS, 1))[big]
            solid |= big & np.tile(self.solid[kd], (ROWS, COLS))
        wsolid = self.solid[WALL]
        for r_, c_ in self.walls:
            sl = (slice(r_ * s, r_ * s + s), slice(c_ * s, c_ * s + s))
            tile = self.wall_tiles[r_]
            field[sl][wsolid] = tile[wsolid]
            solid[sl] |= wsolid
        # 3D: weicher Schlagschatten unten rechts neben den Blöcken
        sh = np.zeros_like(solid)
        sh[3 * k:, 2 * k:] = solid[:-3 * k, :-2 * k]
        sh &= ~solid
        field[sh] = (field[sh] * 0.45).astype(np.uint8)
        f = field.astype(np.float32)
        # Licht-Reflexion wandert alle 5 s über das Feld
        pos = ((t % 5.0) / 5.0) * (COLS + ROWS * 0.55 + 16) - 8
        band = np.exp(-((self.diag - pos) / 2.2) ** 2) * 0.3
        # Kettenreaktion: Welle lässt die Blöcke kurz aufleuchten und die Farbe pulsieren
        I, rings = self.wave_field(t)
        if rings:
            Ib = np.repeat(np.repeat(I, s, 0), s, 1)
            pulse = np.array((255, 170, 255), np.float32)
            f += (Ib * solid)[..., None] * (pulse - f) * 0.6
        f += (band * solid)[..., None] * (255 - f)
        img[BY0 * k:BY0 * k + ROWS * s, BX0 * k:BX0 * k + COLS * s] = np.clip(f, 0, 255).astype(np.uint8)
        im = Image.fromarray(img, "RGB")
        d = ImageDraw.Draw(im, "RGBA")
        for rr, cc in np.argwhere((kind == PLUS) | (kind == MULT)):
            x0, y0 = (BX0 + cc * CELL) * k, (BY0 + rr * CELL) * k
            cx, cy = x0 + s / 2, y0 + s / 2 + s * 0.03
            w = int(s * 0.13)
            ic = (90, 50, 0) if kind[rr, cc] == PLUS else (0, 80, 40)
            if kind[rr, cc] == PLUS:
                d.line([cx - s * 0.22, cy, cx + s * 0.22, cy], fill=ic, width=w)
                d.line([cx, cy - s * 0.22, cx, cy + s * 0.22], fill=ic, width=w)
            else:
                d.line([cx - s * 0.19, cy - s * 0.19, cx + s * 0.19, cy + s * 0.19], fill=ic, width=w)
                d.line([cx - s * 0.19, cy + s * 0.19, cx + s * 0.19, cy - s * 0.19], fill=ic, width=w)
        # Wellen-Ring (dezent) im Feld
        if rings:
            ov = Image.new("RGBA", ((FX1 - FX0) * k, (FY1 - FY0) * k))
            od = ImageDraw.Draw(ov)
            for x, y, R, fade in rings:
                X, Y, RR = (x - FX0) * k, (y - FY0) * k, R * k
                for w_, a_ in ((14 * k, 0.12), (5 * k, 0.3), (2 * k, 0.55)):
                    od.ellipse([X - RR, Y - RR, X + RR, Y + RR], outline=(255, 190, 255, int(255 * a_ * fade)), width=w_)
            im.paste(ov, (FX0 * k, FY0 * k), ov)
        cols = {GREEN: NEON["block"], PLUS: NEON["plus"], MULT: NEON["mult"]}
        lo = bisect.bisect_left(self.broken_t, t - 0.22)
        hi = bisect.bisect_right(self.broken_t, t)
        flashes = []
        for t0, rr, cc, kd in sim.broken[lo:hi]:  # zerplatzen: Lichtblitz + schrumpfender heller Block
            e = (t - t0) / 0.22
            x0, y0 = (BX0 + cc * CELL) * k, (BY0 + rr * CELL) * k
            if e < 0.5:
                flashes.append((x0 + s / 2, y0 + s / 2, 1 - e / 0.5))
            m = s * 0.5 * e
            c = tuple(int(v + (255 - v) * (1 - e) * 0.7) for v in cols.get(kd, NEON["block"]))
            d.rounded_rectangle([x0 + m, y0 + m, x0 + s - m, y0 + s - m], radius=max(1, int(s * 0.2 * (1 - e))), fill=c)
        lo2 = bisect.bisect_left(self.broken_t, t - 0.6)
        for t0, rr, cc, kd in sim.broken[lo2:hi]:
            if kd in (PLUS, MULT):
                e = (t - t0) / 0.6
                cx, cy = (BX0 + (cc + 0.5) * CELL) * k, (BY0 + (rr + 0.5) * CELL) * k
                R = (10 + 60 * e) * k
                col = NEON["plus"] if kd == PLUS else NEON["mult"]
                d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=col + (int(255 * (1 - e)),), width=3 * k)
        # Partikel: fliegen auseinander, werden langsamer, verblassen
        P = self.P
        if len(P):
            i0 = np.searchsorted(P[:, 0], t - self.P_LIFE)
            i1 = np.searchsorted(P[:, 0], t, side="right")
            Q = P[i0:i1]
            if len(Q) > 1800:
                Q = Q[np.linspace(0, len(Q) - 1, 1800).astype(int)]
            a = t - Q[:, 0]
            dr = 6.0
            px = Q[:, 1] + Q[:, 3] * (1 - np.exp(-a * dr)) / dr
            py = Q[:, 2] + Q[:, 4] * (1 - np.exp(-a * dr)) / dr + 60 * a * a
            life = 1 - a / self.P_LIFE
            sz = Q[:, 5] * (0.4 + 0.6 * life)
            ok = (px > FX0 + 3) & (px < FX1 - 3) & (py > FY0 + 3) & (py < FY1 - 3)
            px, py, sz, life, Q = px[ok], py[ok], sz[ok], life[ok], Q[ok]
            for x, y, z, l, r_, g_, b_ in zip(px * k, py * k, sz * k, life, Q[:, 6], Q[:, 7], Q[:, 8]):
                wb = 0.5 * l
                col = (int(r_ + (255 - r_) * wb), int(g_ + (255 - g_) * wb), int(b_ + (255 - b_) * wb), int(255 * min(1, l * 1.4)))
                d.rounded_rectangle([x - z, y - z, x + z, y + z], radius=z * 0.4, fill=col)
        # weiche, kurze Lichtspuren hinter den Bällen
        tr = Image.new("L", (W1 * k, H1 * k))
        td = ImageDraw.Draw(tr)
        nb = len(balls)
        amp = 0.55 * min(1.0, max(0.3, (50 / max(nb, 1)) ** 0.5))
        step = 1.0
        if fi > 0 and nb:
            m0 = min(nb, len(sim.frames[fi - 1][2]))
            if m0:
                step = float(np.median(np.hypot(*(balls[:m0] - sim.frames[fi - 1][2][:m0]).T))) + 1e-3
        n_tr = int(min(10, max(2, round(self.TRAIL_PX / step))))
        hist = [sim.frames[j][2] for j in range(max(0, fi - n_tr), fi + 1)]
        L = len(hist)
        for j in range(L - 1):
            p0, p1 = hist[j], hist[j + 1]
            m = min(len(p0), len(p1))
            if m == 0:
                continue
            q = (j + 1) / L
            val = int(255 * amp * q ** 1.6)
            wd = max(1, int(sim.r * k * 1.5 * (0.35 + 0.65 * q)))
            for (x0, y0), (x1, y1) in zip(p0[:m] * k, p1[:m] * k):
                if abs(x1 - x0) + abs(y1 - y0) < 80 * k:
                    td.line([x0, y0, x1, y1], fill=val, width=wd)
        rgba = im.convert("RGBA")
        sp = self.ball_spr
        for x, y in balls:
            rgba.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        out = rgba.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        fs = self.flash_spr   # Lichtblitz additiv (weicher Lichtpunkt statt Scheibe)
        h = fs.shape[0] // 2
        for x, y, a in flashes[-400:]:
            X, Y = int(x / k), int(y / k)
            y0, y1, x0, x1 = max(0, Y - h), min(H1, Y + h), max(0, X - h), min(W1, X + h)
            arr[y0:y1, x0:x1] += fs[y0 - Y + h:y1 - Y + h, x0 - X + h:x1 - X + h] * a
        trl = np.asarray(tr.reduce(SS).filter(ImageFilter.GaussianBlur(2.2)), np.float32) / 255
        arr = arr + trl[..., None] * np.array((200, 160, 255), np.float32) * (1 - arr / 320)
        arr = arr + self.wall_glow * (0.85 + 0.15 * math.sin(t * 2.2))
        g = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((W1 // 5, H1 // 5), Image.BILINEAR)
        g = g.filter(ImageFilter.GaussianBlur(4))
        glow = np.asarray(g.resize((W1, H1), Image.BILINEAR), np.float32)
        arr = arr + np.clip(glow - 40, 0, 255) * 0.45
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        draw_counter(im, nb)
        return im


SEED = 23
BREAKER = {"k02_block_breaker_waende": lambda: BreakerSim(SEED).run(),
           "k03_neon_herz": lambda: HeartBreakerSim(SEED_HEART).run(),
           "k04_neon_herz_glanz": lambda: SlowHeartSim(20, ramp=(700.0, 1800.0, 22.0), n_outer_plus=14, cap=320).run(),
           "k05_vice_vi": lambda: ViceBreakerSim(SEED_VICE).run(),
           "k06_blitz": lambda: SymbolBreakerSim(SEED_BOLT).run()}
SEED_BOLT = 17
SEED_VICE = 18
SEED_HEART = 10

if __name__ == "__main__":
    import sys
    preview = "--preview" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(BREAKER)
    for name in names:
        mk = BREAKER[name]
        sim = mk()
        if preview:
            ts = [0.0, 3.0, 8.0, 14.0, 20.0, 26.0, 32.0, 38.0, 44.0, sim.total - 3, sim.total - 0.1]
            render(sim, name, None, preview_dir="output/preview", preview_times=ts)
            print(name, "Vorschau", round(sim.total, 1), "clear", sim.clear_t)
        else:
            out = f"output/{name}_vertical.mp4"
            render(sim, name, out)
            print(out, round(sim.total, 1))

