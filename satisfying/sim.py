"""Deterministische 2D-Physik für Satisfying-Simulationen.

Welt-Koordinaten: Arena-Mittelpunkt (0, 0), y nach unten, Arena-Radius 400.
Jede Simulation läuft vorab komplett durch und speichert pro Videoframe einen
Snapshot + eine Ereignisliste (für Töne und Effekte). Der Renderer ist dadurch zustandslos.
"""
import colorsys
import math
import random

FPS = 30
SUB = 8  # Physik-Substeps pro Frame
DT = 1 / (FPS * SUB)
ARENA = 400.0

COLORS = {
    "ROT": (255, 60, 70), "BLAU": (50, 140, 255), "GRÜN": (40, 230, 110), "GELB": (255, 215, 30),
    "LILA": (190, 80, 255), "ORANGE": (255, 140, 30), "PINK": (255, 90, 200), "TÜRKIS": (40, 230, 230),
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
