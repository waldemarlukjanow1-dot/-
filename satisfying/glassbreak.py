"""Glass Escape: Bälle arbeiten zusammen und brechen sich Etage für Etage durch Glasscheiben zum Ausgang.

  * Jede Etage besteht aus mehreren Glasscheiben zwischen festen Pfeilern (verschiedene Wege).
  * Die Bälle springen unter Schwerkraft; jeder Aufprall beschädigt die Scheibe (Risse wachsen sichtbar).
  * Ist eine Scheibe kaputt, zerspringt sie in Scherben; ein darin eingeschlossener Ball wird befreit und hilft mit.
  * Ganz unten liegt die dicke Glastür vor dem Ausgang; ein Countdown läuft.
  * Die Festigkeit jeder Scheibe wird vorab per Simulation so bestimmt, dass die Etagen zu den geplanten Zeiten
    brechen (die Physik selbst bleibt unverändert).
"""
import math
import os
import random
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb

FPS = 60
W1, H1, SS = 1080, 1920, 2
SR = 44100
F_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"

FX0, FX1, FY0, FY1 = 110, 930, 330, 1440
EXIT_Y = 1372          # unterhalb davon ist der Ausgang
G = 1500.0
R_BALL = 22.0
PANE_T = 32.0          # Glasdicke
BALL_COLS = [(255, 70, 150), (40, 210, 255), (255, 214, 60), (60, 235, 160), (170, 110, 255), (255, 140, 70)]


class Pane:
    def __init__(self, x0, x1, y, level, trapped):
        self.x0, self.x1, self.y = x0, x1, y
        self.level, self.trapped = level, trapped
        self.hp = float("inf")
        self.dmg = 0.0
        self.hits = []            # (t, x, stärke)
        self.broken_t = None


def make_level(y, level, rng, n_min=2, n_max=3, door=False):
    """Eine Etage: Scheiben zwischen Pfeilern über die ganze Breite."""
    if door:
        return [Pane(FX0, FX1, y, level, False)], []
    n = rng.randint(n_min, n_max)
    while True:   # Pfeiler mit Mindestabstand (jede Scheibe ausreichend breit)
        cuts = sorted(rng.uniform(0.2, 0.8) for _ in range(n - 1))
        if all(b_ - a_ > 0.2 for a_, b_ in zip([0.0] + cuts, cuts + [1.0])):
            break
    xs = [FX0] + [FX0 + c * (FX1 - FX0) for c in cuts] + [FX1]
    panes, pillars = [], []
    for k in range(n):
        a, b = xs[k], xs[k + 1]
        if k > 0:
            a += 9
            pillars.append((xs[k] - 9, xs[k] + 9, y - 22, y + PANE_T + 22))
        if k < n - 1:
            b -= 9
        panes.append(Pane(a, b, y, level, rng.random() < 0.7))
    return panes, pillars


class GlassSim:
    def __init__(self, seed, level_ys=(500, 670, 840, 1010, 1180, 1330), n_start=4, timer=45.0, lead=2.0):
        self.rng = random.Random(seed)
        self.seed = seed
        self.level_ys = level_ys
        self.timer, self.lead = timer, lead
        self.panes, self.pillars = [], []
        for k, y in enumerate(level_ys):
            p, q = make_level(y, k, self.rng, door=(k == len(level_ys) - 1))
            self.panes += p
            self.pillars += q
        # runde Stifte zwischen den Etagen (Abwechslung beim Abprallen)
        self.pegs = []
        for k in range(len(level_ys) - 1):
            ya, yb = level_ys[k], level_ys[k + 1]
            for _ in range(2):
                self.pegs.append((self.rng.uniform(FX0 + 80, FX1 - 80), self.rng.uniform(ya + 55, yb - 45), 12.0))
        self.n_start = n_start

    # ---------------------------------------------------------------- Simulation
    def run(self, until=None):
        rng = random.Random(self.seed * 7 + 1)
        for p in self.panes:
            p.dmg, p.hits, p.broken_t = 0.0, [], None
        balls = []
        for k in range(self.n_start):
            balls.append([FX0 + (FX1 - FX0) * (k + 1) / (self.n_start + 1), FY0 + 40 + rng.uniform(0, 30),
                          rng.uniform(-120, 120), 0.0, k % len(BALL_COLS), None, 0.0, -9.0])
            # x y vx vy farbe entkommen geboren letzter-Treffer
        self.balls_hist = []
        self.events = []
        self.shards = []
        self.escaped_t = []
        total = (until or (self.lead + self.timer + 7.0))
        dt = 1 / (FPS * 4)
        t = 0.0
        self.dmg_hist = []
        frame = 0
        self.door_t = None
        while t < total:
            for _ in range(4):
                self._step(balls, t, dt, rng)
                t += dt
            frame += 1
            self.balls_hist.append(np.array([(b[0], b[1], b[4], -1 if b[5] is None else b[5], b[6], b[7])
                                             for b in balls], np.float32))
            self.dmg_hist.append([p.dmg for p in self.panes])
        self.total = total
        return self

    def _step(self, balls, t, dt, rng):
        if t < self.lead * 0.5:
            return
        live = [b for b in balls if b[5] is None]
        for b in live:
            b[3] += G * dt
            b[0] += b[2] * dt
            b[1] += b[3] * dt
            # Rand
            if b[0] < FX0 + R_BALL:
                b[0], b[2] = FX0 + R_BALL, abs(b[2])
            if b[0] > FX1 - R_BALL:
                b[0], b[2] = FX1 - R_BALL, -abs(b[2])
            if b[1] < FY0 + R_BALL:
                b[1], b[3] = FY0 + R_BALL, abs(b[3])
            # Glasscheiben
            for p in self.panes:
                if p.broken_t is not None:
                    continue
                if p.x0 - R_BALL < b[0] < p.x1 + R_BALL and p.y - R_BALL < b[1] < p.y + PANE_T + R_BALL:
                    cx = min(max(b[0], p.x0), p.x1)
                    cy = min(max(b[1], p.y), p.y + PANE_T)
                    dx, dy = b[0] - cx, b[1] - cy
                    d = math.hypot(dx, dy)
                    if d < R_BALL:
                        if d < 1e-6:
                            dx, dy, d = 0.0, -1.0, 1.0
                        nx, ny = dx / d, dy / d
                        b[0] += nx * (R_BALL - d)
                        b[1] += ny * (R_BALL - d)
                        vn = b[2] * nx + b[3] * ny
                        if vn < 0:
                            imp = -vn
                            b[2] -= 2 * vn * nx
                            b[3] -= 2 * vn * ny
                            if ny < -0.5:   # von oben: Schwung erhalten (Bälle springen immer wieder)
                                b[3] = -min(max(-b[3], 660.0 + rng.uniform(0, 140)), 900.0)
                                b[2] += rng.uniform(-60, 60)
                            b[7] = t
                            dmg = 1.0 if imp > 150 else 0.0    # jeder kräftige Treffer zählt genau 1
                            p.dmg += dmg
                            if dmg:
                                p.hits.append((t, cx, dmg, ny < 0))
                            if dmg:
                                self.events.append((t, "hit", {"frac": min(1.0, p.dmg / p.hp) if p.hp < 1e9 else 0.3,
                                                               "level": p.level,
                                                               "left": int(p.hp - p.dmg) if p.hp < 1e9 else 0}))
                            if p.dmg >= p.hp:
                                self._shatter(p, t, balls, rng)
            # Pfeiler
            for x0, x1, y0, y1 in self.pillars:
                cx = min(max(b[0], x0), x1)
                cy = min(max(b[1], y0), y1)
                dx, dy = b[0] - cx, b[1] - cy
                d = math.hypot(dx, dy)
                if d < R_BALL:
                    if d < 1e-6:
                        dx, dy, d = 0.0, -1.0, 1.0
                    nx, ny = dx / d, dy / d
                    b[0] += nx * (R_BALL - d)
                    b[1] += ny * (R_BALL - d)
                    vn = b[2] * nx + b[3] * ny
                    if vn < 0:
                        b[2] -= 2 * vn * nx
                        b[3] -= 2 * vn * ny
            # Stifte
            for px, py, pr in self.pegs:
                dx, dy = b[0] - px, b[1] - py
                d = math.hypot(dx, dy)
                if d < R_BALL + pr:
                    nx, ny = dx / d, dy / d
                    b[0], b[1] = px + nx * (R_BALL + pr), py + ny * (R_BALL + pr)
                    vn = b[2] * nx + b[3] * ny
                    if vn < 0:
                        b[2] -= 1.9 * vn * nx
                        b[3] -= 1.9 * vn * ny
                        self.events.append((t, "peg", {}))
            # Tempo begrenzen
            sp = math.hypot(b[2], b[3])
            if sp > 1100:
                b[2] *= 1100 / sp
                b[3] *= 1100 / sp
            # Ausgang
            if b[1] > EXIT_Y + 30:
                b[5] = t
                self.escaped_t.append(t)
                self.events.append((t, "escape", {"n": len(self.escaped_t)}))
        # Bälle untereinander
        for i in range(len(live)):
            a = live[i]
            for j in range(i + 1, len(live)):
                c = live[j]
                dx, dy = c[0] - a[0], c[1] - a[1]
                d2 = dx * dx + dy * dy
                if d2 < (2 * R_BALL) ** 2 and d2 > 1e-9:
                    d = math.sqrt(d2)
                    nx, ny = dx / d, dy / d
                    ov = 2 * R_BALL - d
                    a[0] -= nx * ov / 2
                    a[1] -= ny * ov / 2
                    c[0] += nx * ov / 2
                    c[1] += ny * ov / 2
                    rv = (c[2] - a[2]) * nx + (c[3] - a[3]) * ny
                    if rv < 0:
                        a[2] += rv * nx
                        a[3] += rv * ny
                        c[2] -= rv * nx
                        c[3] -= rv * ny

    def _shatter(self, p, t, balls, rng):
        p.broken_t = t
        self.events.append((t, "shatter", {"level": p.level, "w": p.x1 - p.x0}))
        if p.level == len(self.level_ys) - 1:
            self.door_t = t
            self.events.append((t, "door", {}))
        # Scherben
        n = int((p.x1 - p.x0) / 16)
        hx = [h[1] for h in p.hits[-30:]] or [(p.x0 + p.x1) / 2]
        cx = hx[-1]
        for k in range(n):
            x = rng.uniform(p.x0, p.x1)
            y = p.y + rng.uniform(0, PANE_T)
            vx = (x - cx) * rng.uniform(0.6, 1.6) + rng.uniform(-60, 60)
            vy = rng.uniform(-220, -20)
            self.shards.append((t, x, y, vx, vy, rng.uniform(9, 24), rng.uniform(0, 6.28), rng.uniform(-6, 6),
                                rng.random()))
        if p.trapped:
            balls.append([(p.x0 + p.x1) / 2, p.y + PANE_T / 2, rng.uniform(-150, 150), -300.0,
                          (len(balls) + 2) % len(BALL_COLS), None, t, -9.0])
            self.events.append((t, "free", {}))


def calibrate(seed, targets, **kw):
    """Festigkeit der Scheiben so wählen, dass die Etagen zu den Zielzeiten brechen."""
    sim = GlassSim(seed, **kw)
    for lvl, T in enumerate(targets):
        sim.run(until=T + 0.5)
        ids = [i for i, p in enumerate(sim.panes) if p.level == lvl]
        fi = min(len(sim.dmg_hist) - 1, int(T * FPS))
        dm = {i: sim.dmg_hist[fi][i] for i in ids}
        best = max(ids, key=lambda i: dm[i])
        if dm[best] <= 0.05:
            return None
        for i in ids:
            sim.panes[i].hp = dm[best] if i == best else math.ceil(max(dm[i] + 3.0, dm[best] * 1.25))
    return sim.run()


# ------------------------------------------------------------------ Darstellung

def _font(path, size, cache={}):
    k = (path, size)
    if k not in cache:
        cache[k] = ImageFont.truetype(path, size)
    return cache[k]


def ball_sprite(r, col):
    """Glänzende Bonbon-Murmel: Verlauf, Randlicht, großer weicher und kleiner harter Glanzpunkt, weiches Leuchten."""
    n = int(r * 5)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2 + 0.5
    d = np.hypot(xx, yy)
    core = np.clip(r - d + 0.7, 0, 1)
    glow = np.clip(1 - d / (r * 2.4), 0, 1) ** 2.2 * 0.55
    u = np.clip(d / r, 0, 1)
    c = np.array(col, np.float32)
    light = np.clip(1.0 - ((xx + r * 0.35) ** 2 + (yy + r * 0.45) ** 2) / (2.2 * r * r), 0, 1)
    rgb = c * (0.55 + 0.6 * light[..., None]) + (255 - c) * (0.25 * light[..., None] ** 3)
    rgb = rgb * (1 - 0.25 * u[..., None] ** 4)                                   # dunklerer Rand
    rim = np.exp(-((u - 0.92) / 0.06) ** 2) * np.clip((yy + xx * 0.2) / r, 0, 1)  # Randlicht unten
    rgb = rgb + rim[..., None] * (255 - rgb) * 0.45
    soft = np.exp(-(((xx + r * 0.32) / (r * 0.42)) ** 2 + ((yy + r * 0.42) / (r * 0.28)) ** 2))
    hard = np.exp(-(((xx + r * 0.38) / (r * 0.13)) ** 2 + ((yy + r * 0.5) / (r * 0.1)) ** 2))
    rgb = rgb + (soft * 0.45 + hard * 0.95)[..., None] * (255 - rgb)
    gl = np.zeros_like(rgb)
    gl[:] = c
    rgb = rgb * core[..., None] + gl * (1 - core[..., None])
    a = np.maximum(glow, core)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA")


class GlassRenderer:
    def __init__(self, sim):
        self.sim = sim
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        top, bot = np.array((34, 16, 62), np.float32), np.array((8, 22, 56), np.float32)
        f = (yy / H1)[..., None]
        base = top * (1 - f) + bot * f
        rng = np.random.default_rng(3)
        for _ in range(14):   # weiche Bokeh-Lichter im Hintergrund
            bx, by, br = rng.uniform(0, W1), rng.uniform(150, H1 - 150), rng.uniform(40, 120)
            colb = np.array(BALL_COLS[rng.integers(len(BALL_COLS))], np.float32)
            m = np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * br * br)) * 0.10
            base += m[..., None] * colb
        inside = (xx > FX0) & (xx < FX1) & (yy > FY0) & (yy < FY1)
        base[inside] = base[inside] * 0.55 + np.array((6, 8, 20))
        ex = inside & (yy > EXIT_Y)
        gl = np.clip((yy - EXIT_Y) / (FY1 - EXIT_Y), 0, 1)
        base[ex] = base[ex] + (gl[ex] ** 0.7)[:, None] * np.array((20, 110, 70))
        self.base = Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(self.base, "RGBA")
        d.rounded_rectangle([(FX0 - 10) * k, (FY0 - 10) * k, (FX1 + 10) * k, (FY1 + 10) * k], radius=30 * k,
                            outline=(160, 140, 255, 150), width=4 * k)
        for x0, x1, y0, y1 in sim.pillars:
            d.rounded_rectangle([x0 * k, y0 * k, x1 * k, y1 * k], radius=8 * k, fill=(70, 66, 120))
            d.rounded_rectangle([x0 * k + 3 * k, y0 * k + 4 * k, x0 * k + 7 * k, y1 * k - 4 * k], radius=2 * k,
                                fill=(150, 145, 220))
        for px, py, pr in sim.pegs:
            spr = ball_sprite(pr * k, (190, 185, 230))
            self.base.alpha_composite(spr, (int(px * k - spr.width / 2), int(py * k - spr.height / 2)))
        d.text((520 * k, (EXIT_Y + 38) * k), "EXIT", font=_font(F_BOLD, 42 * k), fill=(110, 255, 180), anchor="mm")
        self.sprites = [ball_sprite(R_BALL * k, c) for c in BALL_COLS]
        self.small = [ball_sprite(R_BALL * 0.8 * k, c) for c in BALL_COLS]

    @staticmethod
    def cracks(i, p, t, frac):
        """Spinnennetz-Risse ausgehend von den Trefferstellen."""
        lines = []
        spots = []
        for (ht, hx, dm, top) in p.hits:
            if ht > t:
                break
            if not spots or min(abs(hx - s_) for s_ in spots) > 34:
                spots.append(hx)
        cy = p.y + PANE_T * 0.5
        for hx in spots[-12:]:
            rng = random.Random(i * 1000 + int(hx))
            n_arm = rng.randint(5, 8)
            L = (10 + 110 * frac) * rng.uniform(0.7, 1.2)
            arms = []
            for a in range(n_arm):
                ang = 2 * math.pi * (a + rng.uniform(-0.3, 0.3)) / n_arm
                x, y = hx, cy
                pts = [(x, y)]
                for _ in range(4):
                    ang += rng.uniform(-0.4, 0.4)
                    x += math.cos(ang) * L / 4
                    y += math.sin(ang) * L / 4 * 0.45
                    y = min(max(y, p.y + 3), p.y + PANE_T - 3)
                    x = min(max(x, p.x0 + 3), p.x1 - 3)
                    pts.append((x, y))
                arms.append(pts)
                lines.append(pts)
            if frac > 0.45:   # Ringe zwischen den Armen
                for ring in (0.4, 0.75):
                    q = [arm[min(len(arm) - 1, int(ring * 4) + 1)] for arm in arms]
                    lines.append(q + q[:1])
        return lines

    def frame(self, fi):
        sim = self.sim
        k = SS
        t = (fi + 1) / FPS
        fi_ = min(fi, len(sim.dmg_hist) - 1)
        balls = sim.balls_hist[fi_]
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        # eingeschlossene Bälle hinter dem Glas
        for i, p in enumerate(sim.panes):
            if p.broken_t is not None and p.broken_t <= t:
                continue
            if p.trapped:
                spr = self.small[(i + 2) % len(BALL_COLS)]
                cx, cy = (p.x0 + p.x1) / 2 + 60, p.y + PANE_T / 2
                im.alpha_composite(spr, (int(cx * k - spr.width / 2), int(cy * k - spr.height / 2)))
        d = ImageDraw.Draw(im, "RGBA")
        for i, p in enumerate(sim.panes):
            if p.broken_t is not None and p.broken_t <= t:
                continue
            dmg = sim.dmg_hist[fi_][i]
            frac = min(1.0, dmg / p.hp) if p.hp < 1e9 else 0.0
            door = p.level == len(sim.level_ys) - 1
            jit = math.sin(t * 80 + i) * (frac - 0.8) * 7 if frac > 0.8 else 0.0
            x0, y0, x1, y1 = p.x0 * k, (p.y + jit) * k, p.x1 * k, (p.y + PANE_T + jit) * k
            tint = (185, 230, 255) if not door else (185, 255, 225)
            # Kristallglas: klar, Verlauf, Fase oben hell, unten dunkel
            for q in range(4):
                yy0 = y0 + (y1 - y0) * q / 4
                yy1 = y0 + (y1 - y0) * (q + 1) / 4
                d.rectangle([x0, yy0, x1, yy1], fill=tint + (int(42 - 7 * q + 55 * frac),))
            d.rounded_rectangle([x0, y0, x1, y1], radius=6 * k, outline=(225, 245, 255, 200), width=2 * k)
            d.line([x0 + 8 * k, y0 + 4 * k, x1 - 8 * k, y0 + 4 * k], fill=(255, 255, 255, 170), width=2 * k)
            d.line([x0 + 8 * k, y1 - 3 * k, x1 - 8 * k, y1 - 3 * k], fill=(40, 90, 140, 120), width=2 * k)
            for sx in np.arange(p.x0 + 30, p.x1 - 40, 170):
                d.polygon([(sx * k, y1 - 3 * k), ((sx + 14) * k, y0 + 3 * k), ((sx + 30) * k, y0 + 3 * k),
                           ((sx + 16) * k, y1 - 3 * k)], fill=(255, 255, 255, 40))
            for pts in self.cracks(i, p, t, frac):
                q = [(x * k, (y + jit) * k) for x, y in pts]
                d.line(q, fill=(255, 255, 255, 225), width=max(1, int(1.5 * k)), joint="curve")
            # verbleibende Treffer
            if p.hp < 1e9:
                left = int(p.hp - dmg)
                last = p.hits[-1][0] if p.hits else -9
                lh = [h[0] for h in p.hits if h[0] <= t]
                pop = max(0.0, 1 - (t - lh[-1]) / 0.15) if lh else 0.0
                fs = int(((40 if door else 30) * (1 + 0.25 * pop)) * k)
                d.text(((p.x0 + p.x1) / 2 * k, (p.y + PANE_T / 2 + jit) * k), str(max(left, 0)),
                       font=_font(F_BOLD, fs), fill=(255, 255, 255, 255), anchor="mm",
                       stroke_width=4 * k, stroke_fill=(20, 40, 90, 230))
        # Treffer-Ringe
        for p in sim.panes:
            for (ht, hx, dm, top) in p.hits[-12:]:
                e = (t - ht) / 0.3
                if 0 <= e < 1:
                    R = (8 + 30 * e) * k
                    yy = (p.y if top else p.y + PANE_T) * k
                    d.ellipse([hx * k - R, yy - R * 0.45, hx * k + R, yy + R * 0.45],
                              outline=(255, 255, 255, int(200 * (1 - e))), width=2 * k)
        # Scherben: groß, schillernd, langsam fallend
        for (t0, x, y, vx, vy, sz, a0, w, sh) in sim.shards:
            e = t - t0
            if e < 0 or e > 2.2:
                continue
            X = x + vx * e * 0.8
            Y = y + vy * e + 0.5 * G * 0.55 * e * e
            if Y > FY1 - 6 or X < FX0 + 4 or X > FX1 - 4:
                continue
            a = a0 + w * e
            al = int(220 * min(1.0, (2.2 - e) / 0.8))
            pts = [(X + math.cos(a + q) * sz * (0.6 + 0.4 * ((q * 7) % 1)), Y + math.sin(a + q) * sz * 0.7)
                   for q in (0, 2.2, 4.3)]
            hue = (sh + e * 0.3) % 1.0
            ir = [(150, 230, 255), (255, 190, 240), (200, 255, 230), (255, 240, 180)][int(hue * 4) % 4]
            d.polygon([(px * k, py * k) for px, py in pts], fill=ir + (int(al * 0.55),), outline=(255, 255, 255, al))
            if (int((t - t0) * 20 + sh * 10) % 7) == 0:   # Glitzern
                gx, gy = pts[0]
                d.line([(gx - 6) * k, gy * k, (gx + 6) * k, gy * k], fill=(255, 255, 255, al), width=2 * k)
                d.line([gx * k, (gy - 6) * k, gx * k, (gy + 6) * k], fill=(255, 255, 255, al), width=2 * k)
        # weicher Lichtblitz + "+1" beim Befreien
        for i, p in enumerate(sim.panes):
            if p.broken_t is None:
                continue
            e = t - p.broken_t
            if 0 <= e < 0.18:
                d.rounded_rectangle([p.x0 * k, (p.y - 18 * (1 - e / 0.18)) * k, p.x1 * k,
                                     (p.y + PANE_T + 18 * (1 - e / 0.18)) * k], radius=10 * k,
                                    fill=(255, 255, 255, int(130 * (1 - e / 0.18))))
            if p.trapped and 0 <= e < 1.0:
                cx = (p.x0 + p.x1) / 2
                col = BALL_COLS[(i + 2) % len(BALL_COLS)]
                d.text((cx * k, (p.y - 30 - 60 * e) * k), "+1", font=_font(F_BOLD, 46 * k),
                       fill=col + (int(255 * (1 - e)),), anchor="mm", stroke_width=4 * k,
                       stroke_fill=(0, 0, 0, int(160 * (1 - e))))
        # Bälle (mit Stauchen beim Aufprall)
        for x, y, col, esc, born, lh in balls:
            if esc >= 0:
                e = (t - esc) / 0.5
                if e < 1:
                    R = (R_BALL + 44 * e) * k
                    d.ellipse([x * k - R, y * k - R, x * k + R, y * k + R], outline=(110, 255, 180, int(220 * (1 - e))),
                              width=3 * k)
                continue
            spr = self.sprites[int(col)]
            e = (t - lh) / 0.12
            if 0 <= e < 1:
                sx_, sy_ = 1 + 0.13 * (1 - e), 1 - 0.11 * (1 - e)
                spr = spr.resize((int(spr.width * sx_), int(spr.height * sy_)), Image.BILINEAR)
            im.alpha_composite(spr, (int(x * k - spr.width / 2), int(y * k - spr.height / 2)))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 5, H1 // 5), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(g, np.float32) - 40, 0, 255) * 0.4
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        self.hud(out, t, balls)
        return out

    def hud(self, im, t, balls):
        sim = self.sim
        d = ImageDraw.Draw(im, "RGBA")
        left = sim.timer - max(0.0, t - sim.lead)
        if sim.door_t is not None and t >= sim.door_t:
            left = sim.timer - (sim.door_t - sim.lead)
        left = max(0.0, left)
        red = left < 10 and (sim.door_t is None or t < sim.door_t)
        pulse = 1.0 + (0.08 * max(0.0, 1 - (left % 1.0) / 0.25) if red else 0)
        col = (255, 90, 110) if red else (255, 255, 255)
        d.rounded_rectangle([520 - 130, 228, 520 + 130, 312], radius=26, fill=(10, 6, 30, 200),
                            outline=col + (170,), width=3)
        frozen = sim.door_t is not None and t >= sim.door_t
        txt = f"{left:.1f}" if (left < 10 or frozen) else f"0:{int(math.ceil(left - 1e-6)):02d}"
        if frozen:
            col = (110, 255, 180)
        d.text((520, 270), txt, font=_font(F_BOLD, int(64 * pulse)), fill=col + (255,), anchor="mm")
        alive = sum(1 for b in balls if b[3] < 0)
        esc = sum(1 for b in balls if b[3] >= 0)
        d.text((FX0 + 6, 256), "TEAM", font=_font(F_SEMI, 20), fill=(200, 200, 255, 220), anchor="lm")
        d.text((FX0 + 6, 284), f"{alive + esc}", font=_font(F_BOLD, 34), fill=(255, 255, 255, 240), anchor="lm")
        d.text((FX1 - 6, 256), "ESCAPED", font=_font(F_SEMI, 20), fill=(140, 255, 190, 220), anchor="rm")
        d.text((FX1 - 6, 284), f"{esc}", font=_font(F_BOLD, 34), fill=(140, 255, 190, 240), anchor="rm")
        if t < sim.lead + 1.4:
            a = min(1.0, t / 0.3) * min(1.0, (sim.lead + 1.4 - t) / 0.5)
            d.rounded_rectangle([520 - 410, 380, 520 + 410, 470], radius=28, fill=(10, 6, 30, int(215 * a)),
                                outline=(255, 255, 255, int(70 * a)), width=2)
            d.text((520, 425), "Can they break out in time?", font=_font(F_BOLD, 50),
                   fill=(255, 255, 255, int(255 * a)), anchor="mm")
        if sim.door_t is not None and t > sim.door_t + 0.3:
            a = min(1.0, (t - sim.door_t - 0.3) / 0.4)
            sc = 1 + 0.15 * max(0.0, 1 - (t - sim.door_t - 0.3) / 0.3)
            d.rounded_rectangle([520 - 360, 760, 520 + 360, 900], radius=30, fill=(10, 6, 30, int(215 * a)),
                                outline=(110, 255, 180, int(200 * a)), width=3)
            d.text((520, 812), "ESCAPED!", font=_font(F_BOLD, int(76 * sc)), fill=(110, 255, 180, int(255 * a)),
                   anchor="mm")
            left_s = sim.timer - (sim.door_t - sim.lead)
            d.text((520, 868), f"with {left_s:.1f} seconds left", font=_font(F_SEMI, 34),
                   fill=(255, 255, 255, int(235 * a)), anchor="mm")


# ------------------------------------------------------------------ Klang: weich und melodisch

PENTA = [0, 2, 4, 7, 9]


def glass_note(f, dur=0.9):
    """Weicher Glas-Ton (wie ein angeschlagenes Weinglas)."""
    t = np.arange(int(SR * dur)) / SR
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 5.5) + 0.35 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t * 9)
         + 0.12 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t * 14))
    return s * np.minimum(t / 0.004, 1) * 0.28


def tink(f, dur=0.35):
    return glass_note(f, dur)


def shatter_sound(seed, big=False):
    """Angenehmes Zerspringen: weiches Rauschen + absteigende Glas-Glöckchen + sanfter Stoß."""
    rng = np.random.default_rng(seed)
    dur = 1.8 if big else 1.2
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    soft = np.convolve(np.diff(noise, prepend=0), np.ones(6) / 6, mode="same")
    s = soft * np.exp(-t * (6 if big else 9)) * 0.22
    base = 1046.5 if big else 1318.5
    for j in range(9 if big else 6):          # glitzernde Arpeggio-Kaskade
        semi = 12 * (j // 5) + PENTA[j % 5]
        f = base * 2 ** (-(semi) / 12) * 2
        k = glass_note(f, 0.8) * (0.5 - 0.03 * j)
        i = int((0.03 + j * 0.045) * SR)
        e = min(n, i + len(k))
        s[i:e] += k[: e - i]
    thump = np.sin(2 * np.pi * (80 + 40 * np.exp(-t * 25)) * t) * np.exp(-t * 14)
    return s + thump * (0.35 if big else 0.18)


def build_audio(sim, path):
    from .viral import soft_tone
    total = sim.total
    n = int(SR * total)
    out = np.zeros(n + SR * 3)
    last = -1.0

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    for t, kind, d in sim.events:
        if kind == "hit":
            if t - last < 0.045:
                continue
            last = t
            left = d.get("left", 0)
            semi = PENTA[left % 5] + 12 * ((left // 5) % 2)          # Treffer spielen eine Tonleiter
            put(t, glass_note(1046.5 * 2 ** (semi / 12)), 0.4)
        elif kind == "shatter":
            put(t, shatter_sound(int(t * 100), big=d["level"] == len(sim.level_ys) - 1), 0.85)
        elif kind == "free":
            for j, f in enumerate((783.99, 1046.5, 1318.5)):
                put(t + 0.12 + j * 0.06, soft_tone(f), 0.3)
        elif kind == "escape":
            put(t, soft_tone(523.25 * 2 ** (min(d["n"], 14) / 12)), 0.3)
        elif kind == "door":
            for j, f in enumerate((523.25, 659.25, 783.99, 1046.5, 1318.5)):
                put(t + 0.4 + j * 0.09, soft_tone(f), 0.55)
    end = sim.door_t or (sim.lead + sim.timer)
    for s_ in range(10, 0, -1):
        tt = sim.lead + sim.timer - s_
        if tt < end:
            put(tt, glass_note(700, 0.15), 0.5)
    out = out[:n]
    tt = np.arange(n) / SR
    pad = sum(np.sin(2 * np.pi * f * tt + j) / (j + 2) for j, f in enumerate((130.81, 196.0, 261.63, 329.63)))
    mix_ = out + reverb(out, 1.4) * 0.25 + pad * 0.02 * np.minimum(1, tt / 2)
    mix_ *= np.clip((total - tt) / 1.2, 0, 1)
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


def render(sim, out_path, jobs=4):
    global _R
    _R = GlassRenderer(sim)
    n = len(sim.balls_hist)
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


SEED_GLASS = 3
TARGETS = [8, 15, 22, 29, 36, 45]

if __name__ == "__main__":
    sim = calibrate(SEED_GLASS, TARGETS)
    print("Brüche", sorted(round(p.broken_t, 1) for p in sim.panes if p.broken_t), "Team", len(sim.balls_hist[-1]),
          "Tür", round(sim.door_t, 2), "Dauer", sim.total)
    out = "output/g02_glass_escape_v2_vertical.mp4"
    render(sim, out)
    print(out)
