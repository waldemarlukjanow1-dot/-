"""Deterministische 2D-Physik für Satisfying-Simulationen.

Welt-Koordinaten: Arena-Mittelpunkt (0, 0), y nach unten, Arena-Radius 400.
Jede Simulation läuft vorab komplett durch und speichert pro Videoframe einen
Snapshot + eine Ereignisliste (für Töne und Effekte). Der Renderer ist dadurch zustandslos.
"""
import colorsys
import math
import random

import numpy as np

FPS = 30
SUB = 8  # Physik-Substeps pro Frame
DT = 1 / (FPS * SUB)
ARENA = 400.0

COLORS = {
    "ROT": (255, 60, 70), "BLAU": (50, 140, 255), "GRÜN": (40, 230, 110), "GELB": (255, 215, 30),
    "LILA": (190, 80, 255), "ORANGE": (255, 140, 30), "PINK": (255, 90, 200), "TÜRKIS": (40, 230, 230), "WEISS": (240, 240, 255),
}


def hue_rgb(h, s=0.85, v=1.0):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def angdiff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class Ball:
    def __init__(self, x, y, vx, vy, r, name=None, color=(255, 255, 255)):
        self.x, self.y, self.vx, self.vy, self.r = x, y, vx, vy, r
        self.name, self.color, self.alive = name, color, True
        self.last_note = -1.0

    def snap(self):
        return (self.x, self.y, self.r, self.alive, self.color)


def clamp_speed(b, vmin, vmax):
    sp = math.hypot(b.vx, b.vy) or 1e-6
    k = min(max(sp, vmin), vmax) / sp
    b.vx *= k
    b.vy *= k


def collide_pairs(balls):
    hits = []
    for i in range(len(balls)):
        a = balls[i]
        for j in range(i + 1, len(balls)):
            b = balls[j]
            dx, dy = b.x - a.x, b.y - a.y
            d = math.hypot(dx, dy)
            if 0 < d < a.r + b.r:
                nx, ny = dx / d, dy / d
                rel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny
                if rel > 0:  # nähern sich -> elastischer Stoß (gleiche Masse)
                    a.vx -= rel * nx
                    a.vy -= rel * ny
                    b.vx += rel * nx
                    b.vy += rel * ny
                    hits.append((i, j))
                ov = (a.r + b.r - d) / 2
                a.x -= nx * ov
                a.y -= ny * ov
                b.x += nx * ov
                b.y += ny * ov
    return hits


def contain(b, R, gap_center=None, gap_half=0.0):
    """Hält den Ball im Kreis R. Rückgabe: 'bounce', 'escape' oder None."""
    d = math.hypot(b.x, b.y) or 1e-6
    if gap_center is not None and angdiff(math.atan2(b.y, b.x), gap_center) < gap_half:
        return "escape" if d > R else None
    if d > R and gap_center is not None:
        return "escape"  # Lücke gerade verlassen, Mittelpunkt schon draußen
    if d + b.r > R:
        nx, ny = b.x / d, b.y / d
        vn = b.vx * nx + b.vy * ny
        b.x, b.y = nx * (R - b.r), ny * (R - b.r)
        if vn > 0:
            b.vx -= 2 * vn * nx
            b.vy -= 2 * vn * ny
            # flache Treffer anheben, damit nichts an der Wand entlangrollt
            sp = math.hypot(b.vx, b.vy)
            if vn < 0.35 * sp:
                b.vx -= (0.35 * sp - vn) * nx
                b.vy -= (0.35 * sp - vn) * ny
            return "bounce"
    return None


class Sim:
    """Basisklasse: run() füllt self.frames (Snapshots) und self.events [(t, typ, daten)]."""
    gravity = 1100.0
    outro = 6.0  # Sekunden nach dem Finale (Gewinner + Teaser)

    def __init__(self, seed):
        self.seed = seed
        self.rng = random.Random(seed)
        self.t = 0.0
        self.events = []
        self.frames = []
        self.finish_t = None

    def ev(self, kind, **data):
        self.events.append((self.t, kind, data))

    def note(self, i, b):
        if self.t - b.last_note > 0.045:
            b.last_note = self.t
            self.ev("note", ball=i)

    def run(self, max_t=75.0):
        while True:
            for _ in range(SUB):
                self.step(DT)
                self.t += DT
            self.frames.append(self.snapshot())
            if self.finish_t is not None and self.t >= self.finish_t + self.outro:
                break
            if self.t > max_t:
                break
        return self

    @property
    def total(self):
        return len(self.frames) / FPS


# ------------------------------------------------------------------ Ring-Escape

class RingEscape(Sim):
    """Farbkugeln in vielen rotierenden Ringen. Wer durch eine Lücke fliegt, zerschlägt den Ring."""

    def __init__(self, seed, names=("ROT", "BLAU", "GRÜN", "GELB"), n_rings=22, r_in=105, gap_deg=34):
        super().__init__(seed)
        rng = self.rng
        self.rings = []
        for k in range(n_rings):
            R = r_in + (ARENA - r_in) * k / (n_rings - 1)
            om = (0.9 + 0.7 * rng.random()) * (1 if k % 2 == 0 else -1)
            self.rings.append(dict(R=R, th0=rng.uniform(0, 2 * math.pi), om=om,
                                   half=math.radians(gap_deg) / 2, alive=True, color=hue_rgb(k / n_rings * 0.85)))
        self.balls = []
        for i, n in enumerate(names):
            a = 2 * math.pi * i / len(names)
            sp = rng.uniform(450, 600)
            va = rng.uniform(0, 2 * math.pi)
            self.balls.append(Ball(math.cos(a) * 45, math.sin(a) * 45, math.cos(va) * sp, math.sin(va) * sp, 15, n, COLORS[n]))
        self.score = [0] * len(names)
        self.k = 0

    def gap(self, ring, t):
        return ring["th0"] + ring["om"] * t

    def step(self, dt):
        for b in self.balls:
            b.vy += self.gravity * dt
            b.x += b.vx * dt
            b.y += b.vy * dt
        collide_pairs(self.balls)
        for i, b in enumerate(self.balls):
            if self.k >= len(self.rings):
                break
            ring = self.rings[self.k]
            res = contain(b, ring["R"], self.gap(ring, self.t), ring["half"])
            if res == "bounce":
                clamp_speed(b, 520, 900)
                self.note(i, b)
            elif res == "escape":
                ring["alive"] = False
                self.score[i] += 1
                self.ev("break", ring=self.k, ball=i, gap=self.gap(ring, self.t))
                self.k += 1
                if len(self.balls) == 1 and self.k < len(self.rings):
                    b.color = self.rings[self.k]["color"]  # Solo: Ball nimmt Farbe des nächsten Rings an
                if self.k == len(self.rings):
                    self.finish_t = self.t
                    self.ev("finish", winner=self.winner())
        if self.k >= len(self.rings):
            for b in self.balls:  # nach dem Finale: Kugeln im äußeren Rand halten
                if contain(b, ARENA + 30) == "bounce":
                    clamp_speed(b, 300, 900)

    def winner(self):
        m = max(self.score)
        return self.score.index(m)

    def snapshot(self):
        return dict(t=self.t, balls=[b.snap() for b in self.balls], k=self.k, score=list(self.score))

    def quality(self):
        """Spannung: Führungswechsel, knappes Ende, Dauer im Zielbereich."""
        if self.finish_t is None:
            return None
        score, leader, changes, early = [0] * len(self.balls), None, 0, None
        for t, kind, d in self.events:
            if kind != "break":
                continue
            score[d["ball"]] += 1
            m = max(score)
            lead = [i for i, s in enumerate(score) if s == m]
            if len(lead) == 1 and lead[0] != leader:
                if leader is not None:
                    changes += 1
                leader = lead[0]
                if early is None and t > 8:
                    early = leader
        srt = sorted(score)
        margin = srt[-1] - srt[-2]
        return dict(finish=self.finish_t, changes=changes, margin=margin, comeback=early != self.winner())


# --------------------------------------------------------------- Wachsender Ball

class GrowingBall(Sim):
    """Ein Ball wächst bei jedem Aufprall, bis er den Kreis ausfüllt."""

    outro = 5.5

    def __init__(self, seed, r0=18.0, growth=2.3):
        super().__init__(seed)
        rng = self.rng
        a = rng.uniform(0, 2 * math.pi)
        self.ball = Ball(rng.uniform(-80, 80), rng.uniform(-150, 0), math.cos(a) * 520, math.sin(a) * 520, r0)
        self.growth = growth
        self.bounces = 0
        self.hue = rng.random()
        self.r0 = r0
        self.done = False

    def step(self, dt):
        b = self.ball
        if self.done:
            return
        b.vy += self.gravity * dt
        b.x += b.vx * dt
        b.y += b.vy * dt
        if contain(b, ARENA) == "bounce":
            self.bounces += 1
            b.r += self.growth
            self.hue += 0.013
            ang = self.rng.uniform(-0.06, 0.06)  # kleine Störung gegen Endlosschleifen
            c, s = math.cos(ang), math.sin(ang)
            b.vx, b.vy = b.vx * c - b.vy * s, b.vx * s + b.vy * c
            clamp_speed(b, 560, 1000)
            self.note(0, b)
            if b.r >= ARENA * 0.965:
                b.r = ARENA * 0.965
                self.done = True
                self.finish_t = self.t
                self.ev("finish", bounces=self.bounces)

    def snapshot(self):
        b = self.ball
        return dict(t=self.t, x=b.x, y=b.y, r=b.r, hue=self.hue, bounces=self.bounces,
                    pct=(b.r - self.r0) / (ARENA * 0.965 - self.r0))


# ---------------------------------------------------------------- Last One Standing

class Elimination(Sim):
    """Kugeln in einer Arena mit rotierender Lücke. Wer rausfällt, ist raus. Der Letzte gewinnt."""

    def __init__(self, seed, names=("ROT", "BLAU", "GRÜN", "GELB", "LILA", "ORANGE"), gap_deg=30, omega=1.1):
        super().__init__(seed)
        rng = self.rng
        self.th0 = rng.uniform(0, 2 * math.pi)
        self.om = omega * rng.choice((1, -1))
        self.half = math.radians(gap_deg) / 2
        self.balls = []
        for i, n in enumerate(names):
            a = 2 * math.pi * i / len(names)
            sp = rng.uniform(450, 650)
            va = rng.uniform(0, 2 * math.pi)
            self.balls.append(Ball(math.cos(a) * 160, math.sin(a) * 160, math.cos(va) * sp, math.sin(va) * sp, 26, n, COLORS[n]))
        self.out_order = []

    def gap(self, t):
        return self.th0 + self.om * t

    def step(self, dt):
        alive = [b for b in self.balls if b.alive]
        for b in self.balls:
            b.vy += self.gravity * dt
            b.x += b.vx * dt
            b.y += b.vy * dt
        if self.finish_t is not None:
            for b in alive:  # Gewinner hüpft weiter in der Arena
                if contain(b, ARENA) == "bounce":
                    clamp_speed(b, 500, 900)
            return
        for (i, j) in collide_pairs(alive):
            pass
        for i, b in enumerate(self.balls):
            if not b.alive:
                continue
            res = contain(b, ARENA, self.gap(self.t), self.half)
            if res == "bounce":
                clamp_speed(b, 520, 950)
                self.note(i, b)
            elif res == "escape":
                b.alive = False
                self.out_order.append(i)
                left = [k for k, x in enumerate(self.balls) if x.alive]
                self.ev("elim", ball=i, left=len(left))
                if len(left) == 1:
                    self.finish_t = self.t
                    self.ev("finish", winner=left[0])

    def snapshot(self):
        return dict(t=self.t, balls=[b.snap() for b in self.balls], out=list(self.out_order))

    def quality(self):
        if self.finish_t is None:
            return None
        ts = [t for t, k, _ in self.events if k == "elim"]
        gaps = [b - a for a, b in zip([0.0] + ts, ts)]
        return dict(finish=self.finish_t, first=ts[0], min_gap=min(gaps[1:]), max_gap=max(gaps))


# ------------------------------------------------------------- Ball-Explosion

class Multiply(Sim):
    """Jeder Ball, der durch die rotierende Lücke entkommt, wird durch 2 neue ersetzt."""

    def __init__(self, seed, gap_deg=24, omega=1.0, target=250, r=10.0):
        super().__init__(seed)
        rng = self.rng
        self.target, self.r = target, r
        self.th0 = rng.uniform(0, 2 * math.pi)
        self.om = omega * rng.choice((1, -1))
        self.half = math.radians(gap_deg) / 2
        a = rng.uniform(0, 2 * math.pi)
        self.x, self.y = np.zeros(1), np.zeros(1)
        self.vx, self.vy = np.array([math.cos(a) * 550]), np.array([math.sin(a) * 550])
        self.hue = np.array([rng.random()])
        self.fallen = []  # (t, x, y, vx, vy, hue)
        self.last_note = -1.0
        self.last_pop = -1.0
        self.np_rng = np.random.default_rng(seed)

    def gap(self, t):
        return self.th0 + self.om * t

    @property
    def count(self):
        return len(self.x)

    def step(self, dt):
        self.vy += self.gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        d = np.hypot(self.x, self.y) + 1e-6
        R = ARENA
        open_ = self.finish_t is None
        if open_:
            ang = np.arctan2(self.y, self.x)
            ingap = np.abs((ang - self.gap(self.t) + np.pi) % (2 * np.pi) - np.pi) < self.half
            esc = d > R
        else:
            ingap = np.zeros(len(d), bool)
            esc = np.zeros(len(d), bool)
        hit = ~ingap & ~esc & (d + self.r > R)
        if hit.any():
            nx, ny = self.x[hit] / d[hit], self.y[hit] / d[hit]
            vx, vy = self.vx[hit], self.vy[hit]
            vn = vx * nx + vy * ny
            self.x[hit], self.y[hit] = nx * (R - self.r), ny * (R - self.r)
            out = vn > 0
            vx = np.where(out, vx - 2 * vn * nx, vx)
            vy = np.where(out, vy - 2 * vn * ny, vy)
            sp = np.hypot(vx, vy)
            vn2 = -(vx * nx + vy * ny)
            boost = np.where(out & (vn2 < 0.35 * sp), 0.35 * sp - vn2, 0)
            vx, vy = vx - boost * nx, vy - boost * ny
            sp = np.hypot(vx, vy) + 1e-6
            k = np.clip(sp, 480, 900) / sp
            self.vx[hit], self.vy[hit] = vx * k, vy * k
            if out.any() and self.t - self.last_note > 0.07:
                self.last_note = self.t
                self.ev("note", ball=0)
        if esc.any():
            idx = np.nonzero(esc)[0]
            for i in idx:
                self.fallen.append((self.t, self.x[i], self.y[i], self.vx[i], self.vy[i], self.hue[i]))
            keep = ~esc
            n_new = 2 * len(idx)
            ang = self.np_rng.uniform(0, 2 * np.pi, n_new)
            sp = self.np_rng.uniform(450, 620, n_new)
            self.x = np.concatenate([self.x[keep], self.np_rng.uniform(-20, 20, n_new)])
            self.y = np.concatenate([self.y[keep], self.np_rng.uniform(-20, 20, n_new)])
            self.vx = np.concatenate([self.vx[keep], np.cos(ang) * sp])
            self.vy = np.concatenate([self.vy[keep], np.sin(ang) * sp])
            self.hue = np.concatenate([self.hue[keep], (np.repeat(self.hue[esc], 2) + 0.07) % 1])
            if self.t - self.last_pop > 0.05:
                self.last_pop = self.t
                self.ev("pop", count=self.count)
            if self.count >= self.target and self.finish_t is None:
                self.finish_t = self.t
                self.ev("finish", count=self.count)

    def snapshot(self):
        return dict(t=self.t, x=self.x.copy(), y=self.y.copy(), hue=self.hue.copy(), n=self.count,
                    n_fallen=len(self.fallen))


# ------------------------------------------------------------------ String Art

class StringArt(Sim):
    """Jeder Aufprall setzt einen Punkt am Rand. Von allen Punkten führt eine Linie zum Ball."""

    outro = 5.5

    def __init__(self, seed, target=100, speed=(520, 820)):
        super().__init__(seed)
        rng = self.rng
        a = rng.uniform(0, 2 * math.pi)
        self.ball = Ball(rng.uniform(-60, 60), rng.uniform(-120, 0), math.cos(a) * 600, math.sin(a) * 600, 14)
        self.target, self.speed = target, speed
        self.points = []  # (x, y, hue)
        self.hue = rng.random()

    def step(self, dt):
        b = self.ball
        if self.finish_t is not None:
            return
        b.vy += self.gravity * dt
        b.x += b.vx * dt
        b.y += b.vy * dt
        if contain(b, ARENA) == "bounce":
            d = math.hypot(b.x, b.y) or 1
            self.hue += 0.011
            self.points.append((b.x / d * ARENA, b.y / d * ARENA, self.hue))
            clamp_speed(b, *self.speed)
            self.note(0, b)
            if len(self.points) >= self.target:
                self.finish_t = self.t
                self.ev("finish", lines=len(self.points))

    def snapshot(self):
        b = self.ball
        return dict(t=self.t, x=b.x, y=b.y, n=len(self.points), hue=self.hue)
