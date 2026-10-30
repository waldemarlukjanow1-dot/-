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
R_BALL = 18.0
PANE_T = 24.0          # Glasdicke
BALL_COLS = [(255, 90, 120), (90, 200, 255), (255, 210, 80), (140, 255, 150), (200, 130, 255), (255, 150, 70)]


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
                          rng.uniform(-120, 120), 0.0, k % len(BALL_COLS), None, 0.0])  # x y vx vy farbe entkommen geboren
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
            self.balls_hist.append(np.array([(b[0], b[1], b[4], -1 if b[5] is None else b[5], b[6]) for b in balls],
                                            np.float32))
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
                            dmg = imp / 600.0
                            p.dmg += dmg
                            p.hits.append((t, cx, dmg, ny < 0))
                            self.events.append((t, "hit", {"frac": min(1.0, p.dmg / p.hp) if p.hp < 1e9 else 0.3,
                                                           "level": p.level}))
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
        n = int((p.x1 - p.x0) / 9)
        for k in range(n):
            x = rng.uniform(p.x0, p.x1)
            y = p.y + rng.uniform(0, PANE_T)
            hx = [h[1] for h in p.hits[-30:]] or [x]
            cx = hx[-1]
            vx = (x - cx) * rng.uniform(1.0, 3.0) + rng.uniform(-80, 80)
            vy = rng.uniform(-260, 60)
            self.shards.append((t, x, y, vx, vy, rng.uniform(5, 14), rng.uniform(0, 6.28), rng.uniform(-14, 14),
                                rng.random()))
        if p.trapped:
            balls.append([(p.x0 + p.x1) / 2, p.y + PANE_T / 2, rng.uniform(-150, 150), -300.0,
                          len(balls) % len(BALL_COLS), None, t])
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
            sim.panes[i].hp = dm[best] if i == best else max(dm[i] * 1.0 + 3.0, dm[best] * 1.25)
    return sim.run()


# ------------------------------------------------------------------ Darstellung

def _font(path, size, cache={}):
    k = (path, size)
    if k not in cache:
        cache[k] = ImageFont.truetype(path, size)
    return cache[k]


def ball_sprite(r, col):
    n = int(r * 6)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2
    d = np.hypot(xx, yy)
    glow = np.clip(1 - d / (r * 2.8), 0, 1) ** 2 * 0.6
    core = np.clip(r - d + 0.5, 0, 1)
    hl = np.exp(-(((xx + r * 0.35) / (r * 0.33)) ** 2 + ((yy + r * 0.38) / (r * 0.33)) ** 2))
    shade = 1 - 0.35 * np.clip((xx * 0.4 + yy) / r, 0, 1)
    rgb = np.zeros((n, n, 3), np.float32)
    rgb[:] = col
    rgb = rgb * shade[..., None] + hl[..., None] * (255 - rgb) * 0.85
    a = np.maximum(glow, core)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA")


class GlassRenderer:
    def __init__(self, sim):
        self.sim = sim
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        vig = 0.5 + 0.5 * np.exp(-(((xx - 520) / W1) ** 2 + ((yy - 880) / H1) ** 2) * 3.0)
        base = np.array((9, 12, 24), np.float32) * vig[..., None]
        inside = (xx > FX0) & (xx < FX1) & (yy > FY0) & (yy < FY1)
        base[inside] = (base[inside] * 1.25 + np.array((4, 6, 12))).clip(0, 255)
        # Ausgang: grünes Leuchten unten
        ex = inside & (yy > EXIT_Y)
        glow = np.clip((yy - EXIT_Y) / (FY1 - EXIT_Y), 0, 1)
        base[ex] = base[ex] + (glow[ex] ** 0.7)[:, None] * np.array((20, 90, 55))
        self.base = Image.fromarray(base.clip(0, 255).astype(np.uint8))
        d = ImageDraw.Draw(self.base)
        d.rounded_rectangle([(FX0 - 8) * k, (FY0 - 8) * k, (FX1 + 8) * k, (FY1 + 8) * k], radius=24 * k,
                            outline=(70, 80, 110), width=4 * k)
        for x0, x1, y0, y1 in sim.pillars:
            d.rounded_rectangle([x0 * k, y0 * k, x1 * k, y1 * k], radius=5 * k, fill=(60, 66, 86))
            d.line([(x0 + 3) * k, (y0 + 4) * k, (x0 + 3) * k, (y1 - 4) * k], fill=(120, 130, 160), width=2 * k)
        for px, py, pr in sim.pegs:
            d.ellipse([(px - pr) * k, (py - pr) * k, (px + pr) * k, (py + pr) * k], fill=(70, 78, 100))
            d.ellipse([(px - pr * 0.5) * k, (py - pr * 0.6) * k, (px + pr * 0.1) * k, (py - pr * 0.1) * k],
                      fill=(150, 160, 190))
        d.text((520 * k, (EXIT_Y + 40) * k), "EXIT", font=_font(F_BOLD, 40 * k), fill=(90, 255, 160), anchor="mm")
        self.sprites = [ball_sprite(R_BALL * k, c) for c in BALL_COLS]
        self.crack_cache = {}

    def cracks(self, i, p, t):
        """Risslinien für Scheibe i bis Zeit t (deterministisch je Treffer)."""
        lines = []
        frac = min(1.0, p.dmg_at / p.hp) if p.hp < 1e9 else min(1.0, p.dmg_at / 40.0)
        spots = []
        for (ht, hx, dm, top) in p.hits:
            if ht > t:
                break
            if not spots or min(abs(hx - s_) for s_ in spots) > 26:
                spots.append(hx)
        for si, hx in enumerate(spots[-14:]):
            rng = random.Random(i * 1000 + int(hx))
            for arm in range(rng.randint(3, 5)):
                ang = rng.choice((0, math.pi)) + rng.uniform(-0.5, 0.5)
                L = (14 + 120 * frac) * rng.uniform(0.5, 1.2)
                x, y = hx, p.y + PANE_T * rng.uniform(0.3, 0.7)
                pts = [(x, y)]
                for _ in range(5):
                    ang += rng.uniform(-0.6, 0.6)
                    x += math.cos(ang) * L / 5
                    y += math.sin(ang) * L / 5
                    y = min(max(y, p.y + 2), p.y + PANE_T - 2)
                    x = min(max(x, p.x0 + 2), p.x1 - 2)
                    pts.append((x, y))
                lines.append(pts)
        return lines, frac

    def frame(self, fi):
        sim = self.sim
        k = SS
        t = (fi + 1) / FPS
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        fi_ = min(fi, len(sim.dmg_hist) - 1)
        balls = sim.balls_hist[fi_]
        # eingeschlossene Bälle (noch nicht befreit) im Glas
        for i, p in enumerate(sim.panes):
            p.dmg_at = sim.dmg_hist[fi_][i]
            if p.broken_t is not None and p.broken_t <= t:
                continue
            if p.trapped:
                cx, cy = (p.x0 + p.x1) / 2, p.y + PANE_T / 2
                spr = self.sprites[(i * 3) % len(BALL_COLS)]
                im.alpha_composite(spr, (int(cx * k - spr.width / 2), int(cy * k - spr.height / 2))) \
                    if im.mode == "RGBA" else None
        im = im.convert("RGBA")
        for i, p in enumerate(sim.panes):   # eingeschlossene Bälle (hinter dem Glas)
            if p.broken_t is not None and p.broken_t <= t:
                continue
            if p.trapped:
                cx, cy = (p.x0 + p.x1) / 2, p.y + PANE_T / 2
                spr = self.sprites[(i * 3) % len(BALL_COLS)]
                im.alpha_composite(spr, (int(cx * k - spr.width / 2), int(cy * k - spr.height / 2)))
        d = ImageDraw.Draw(im, "RGBA")
        for i, p in enumerate(sim.panes):
            if p.broken_t is not None and p.broken_t <= t:
                continue
            lines, frac = self.cracks(i, p, t)
            door = p.level == len(sim.level_ys) - 1
            jit = 0.0
            if frac > 0.75:   # kurz vor dem Bruch: Glas zittert
                jit = math.sin(t * 90 + i) * (frac - 0.75) * 6
            x0, y0, x1, y1 = p.x0 * k, (p.y + jit) * k, p.x1 * k, (p.y + PANE_T + jit) * k
            tint = (175, 225, 255) if not door else (175, 255, 215)
            d.rectangle([x0, y0, x1, y1], fill=tint + (int(38 + 60 * frac),), outline=tint + (150,), width=k)
            d.line([x0, y0 + 2 * k, x1, y0 + 2 * k], fill=(255, 255, 255, 190), width=2 * k)
            # Spiegelung: schräge helle Streifen
            for sx in np.arange(p.x0 + 20, p.x1 - 30, 140):
                d.polygon([((sx) * k, y1 - 2 * k), ((sx + 10) * k, y0 + 3 * k), ((sx + 22) * k, y0 + 3 * k),
                           ((sx + 12) * k, y1 - 2 * k)], fill=(255, 255, 255, 45))
            for pts in lines:
                q = [(x * k, (y + jit) * k) for x, y in pts]
                d.line(q, fill=(20, 40, 60, 120), width=max(1, int(3.2 * k)))
                d.line(q, fill=(255, 255, 255, 235), width=max(1, int(1.8 * k)))
        # Scherben
        for (t0, x, y, vx, vy, sz, a0, w, sh) in sim.shards:
            e = t - t0
            if e < 0 or e > 1.6:
                continue
            X = x + vx * e
            Y = y + vy * e + 0.5 * G * e * e
            if Y > FY1 - 4:
                continue
            a = a0 + w * e
            al = int(230 * (1 - e / 1.6))
            pts = [(X + math.cos(a + q) * sz * (0.6 + 0.4 * ((q * 7) % 1)), Y + math.sin(a + q) * sz * 0.8)
                   for q in (0, 2.3, 4.2)]
            c = (200 + int(55 * sh), 240, 255)
            d.polygon([(px * k, py * k) for px, py in pts], fill=c + (al,), outline=(255, 255, 255, al))
        # Lichtblitz beim Zerspringen
        for p in sim.panes:
            if p.broken_t is not None and 0 <= t - p.broken_t < 0.12:
                e = (t - p.broken_t) / 0.12
                d.rectangle([p.x0 * k, (p.y - 30 * (1 - e)) * k, p.x1 * k, (p.y + PANE_T + 30 * (1 - e)) * k],
                            fill=(255, 255, 255, int(160 * (1 - e))))
        # Bälle
        for x, y, col, esc, born in balls:
            if esc >= 0:
                e = (t - esc) / 0.5
                if e < 1:
                    R = (R_BALL + 40 * e) * k
                    d.ellipse([x * k - R, y * k - R, x * k + R, y * k + R], outline=(110, 255, 170, int(220 * (1 - e))),
                              width=3 * k)
                continue
            spr = self.sprites[int(col)]
            im.alpha_composite(spr, (int(x * k - spr.width / 2), int(y * k - spr.height / 2)))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 5, H1 // 5), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(g, np.float32) - 45, 0, 255) * 0.35
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
        col = (255, 90, 90) if red else (255, 255, 255)
        f = _font(F_BOLD, int(64 * pulse))
        d.rounded_rectangle([520 - 130, 228, 520 + 130, 312], radius=24, fill=(0, 0, 0, 170),
                            outline=col + (160,), width=3)
        d.text((520, 270), f"0:{int(math.ceil(left - 1e-6)):02d}", font=f, fill=col + (255,), anchor="mm")
        alive = sum(1 for b in balls if b[3] < 0)
        esc = sum(1 for b in balls if b[3] >= 0)
        d.text((FX0 + 6, 270), f"TEAM {alive + esc}", font=_font(F_BOLD, 30), fill=(200, 220, 255, 230), anchor="lm")
        d.text((FX1 - 6, 270), f"OUT {esc}", font=_font(F_BOLD, 30), fill=(110, 255, 170, 230), anchor="rm")
        # Hook
        if t < sim.lead + 1.2:
            a = min(1.0, t / 0.3) * min(1.0, (sim.lead + 1.2 - t) / 0.5)
            d.rounded_rectangle([520 - 400, 395, 520 + 400, 485], radius=26, fill=(0, 0, 0, int(200 * a)))
            d.text((520, 440), "Can they break out in time?", font=_font(F_BOLD, 50),
                   fill=(255, 255, 255, int(255 * a)), anchor="mm")
        if sim.door_t is not None and t > sim.door_t + 0.3:
            a = min(1.0, (t - sim.door_t - 0.3) / 0.4)
            sc = 1 + 0.15 * max(0.0, 1 - (t - sim.door_t - 0.3) / 0.3)
            d.rounded_rectangle([520 - 360, 760, 520 + 360, 900], radius=30, fill=(0, 0, 0, int(200 * a)),
                                outline=(110, 255, 170, int(200 * a)), width=3)
            d.text((520, 812), "ESCAPED!", font=_font(F_BOLD, int(76 * sc)), fill=(110, 255, 170, int(255 * a)),
                   anchor="mm")
            left_s = sim.timer - (sim.door_t - sim.lead)
            d.text((520, 868), f"with {left_s:.1f} seconds left", font=_font(F_SEMI, 34),
                   fill=(255, 255, 255, int(235 * a)), anchor="mm")


# ------------------------------------------------------------------ Klang

def tink(f, dur=0.35):
    t = np.arange(int(SR * dur)) / SR
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 22) + 0.5 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 35)
         + 0.25 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 55))
    return s * np.minimum(t / 0.001, 1) * 0.3


def shatter_sound(seed, big=False):
    rng = np.random.default_rng(seed)
    dur = 1.4 if big else 0.9
    n = int(SR * dur)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    hp = np.diff(noise, prepend=0)                     # hell, glasig
    s = hp * np.exp(-t * (5 if big else 8)) * 0.5
    for _ in range(60 if big else 30):                 # herabregnende Scherben
        t0 = rng.uniform(0.02, dur * 0.8) ** 1.3
        f = rng.uniform(2500, 6500)
        k = tink(f, 0.15) * rng.uniform(0.2, 0.6)
        i = int(t0 * SR)
        e = min(n, i + len(k))
        s[i:e] += k[: e - i]
    thump = np.sin(2 * np.pi * (70 + 50 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
    return s + thump * (0.6 if big else 0.3)


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

    k = 0
    for t, kind, d in sim.events:
        if kind == "hit":
            if t - last < 0.03:
                continue
            last = t
            f = 2200 + 1800 * d["frac"] + 150 * (k % 5)
            put(t, tink(f), 0.35 + 0.3 * d["frac"])
            k += 1
        elif kind == "shatter":
            put(t, shatter_sound(int(t * 100), big=d["level"] == len(sim.level_ys) - 1), 0.9)
        elif kind == "free":
            for j, f in enumerate((1046.5, 1318.5)):
                put(t + 0.1 + j * 0.07, soft_tone(f), 0.35)
        elif kind == "escape":
            put(t, soft_tone(523.25 * 2 ** (min(d["n"], 14) / 12)), 0.35)
        elif kind == "door":
            for j, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
                put(t + 0.35 + j * 0.1, soft_tone(f), 0.6)
    # Countdown-Ticken in den letzten 10 s
    end = sim.door_t or (sim.lead + sim.timer)
    for s_ in range(10, 0, -1):
        tt = sim.lead + sim.timer - s_
        if tt < end:
            put(tt, tink(1500, 0.08) * 0.6, 0.5)
    out = out[:n]
    tt = np.arange(n) / SR
    pulse = np.sin(2 * np.pi * 55 * tt) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 1.0 * tt))) * 0.02
    mix_ = out + reverb(out, 1.0) * 0.18 + pulse * np.clip((tt - sim.lead) / 2, 0, 1) * (tt < end)
    mix_ *= np.clip((total - tt) / 1.2, 0, 1)
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
    out = "output/g01_glass_escape_vertical.mp4"
    render(sim, out)
    print(out)
