"""Pong Wars – Block für Block, im Glanz-Look.

Regeln (bewusst einfach und nachvollziehbar):
  * Lila (oben) gegen Türkis (unten). Ein Ball fliegt nur durch Felder seiner Farbe.
  * Berührt ein Ball ein gegnerisches Feld, wird genau dieses eine Feld umgefärbt und der Ball prallt ab.
  * Schwung: die Ballzahl eines Teams richtet sich nach seiner Fläche (überproportional, mit der Zeit stärker). Wer Gebiet gewinnt,
    bekommt einen Ball dazu (spaltet sich von einem eigenen Ball ab); wer Gebiet verliert, verliert einen Ball
    (der am stärksten eingekesselte ploppt). So setzt sich irgendwann eine Farbe durch.
  * Hat ein Team keine Bälle mehr, füllen die Sieger die letzten Felder, danach wird alles ruhig.
"""
import bisect
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
from .breaker import glossy_tile, plop
from .marble_render import reverb

FPS = 60
W1, H1, SS = 1080, 1920, 2
SR = 44100
CELL = 30
COLS, ROWS = 28, 38
FX0, FY0 = 100, 290
FX1, FY1 = FX0 + COLS * CELL, FY0 + ROWS * CELL   # 940, 1430
BG = (12, 7, 22)
TEAM_C = [(170, 75, 255), (40, 215, 235)]           # Lila, Türkis
BALL_C = [(120, 245, 255), (240, 150, 255)]      # Ball in der Gegenfarbe, gut sichtbar


class PongFx:
    def __init__(self, seed, power=(1.0, 0.3), n_max=40, n_ramp=34.0, ramp=(560.0, 1250.0, 26.0), r=9.0, max_t=75.0, ko=0.04,
                 split="tb", team_c=None, ball_c=None, names=None, konter=None, lag=0.0):
        self.rng = random.Random(seed)
        self.state = np.zeros((ROWS, COLS), np.uint8)
        self.split = split
        if split == "lr":          # links gegen rechts
            self.state[:, COLS // 2:] = 1
        else:                      # oben gegen unten
            self.state[ROWS // 2:] = 1
        self.team_c, self.ball_c, self.names = team_c or TEAM_C, ball_c or BALL_C, names
        # Konter: (Schwelle, Bonus, Abklingzeit) – einmal pro Runde, wenn ein Team unter die Schwelle fällt
        self.konter, self.konter_t, self.konter_team = konter, None, None
        self.lag, self.share_hist = lag, []   # Verstärkung reagiert verzögert auf die Lage
        self.ko = ko
        self.power, self.n_max, self.n_ramp, self.ramp, self.r, self.max_t = power, n_max, n_ramp, ramp, r, max_t
        self.balls = []      # [x, y, vx, vy, team, alive]
        self.flips = []      # (t, row, col, neues Team, altes Team)
        self.events = []     # (t, art, daten)
        self.births, self.deaths = [], []   # (t, ball-index)
        self.frames = []
        self.t = 0.0
        self.winner = self.win_t = self.clear_t = None
        for team in (0, 1):
            if split == "lr":
                x = FX0 + (COLS * 0.25 if team == 0 else COLS * 0.75) * CELL
                y = (FY0 + FY1) / 2 + self.rng.uniform(-150, 150)
                a = math.radians((30 if team == 0 else 210) + self.rng.uniform(-15, 15))
            else:
                x = (FX0 + FX1) / 2 + self.rng.uniform(-80, 80)
                y = FY0 + (ROWS * 0.25 if team == 0 else ROWS * 0.75) * CELL
                a = math.radians((60 if team == 0 else -120) + self.rng.uniform(-15, 15))
            self.balls.append([x, y, math.cos(a), math.sin(a), team, True])
        self.last_change = [-9.0, -9.0]

    # ---------------------------------------------------------------- Physik
    def step_ball(self, b, dist):
        r = self.r
        b[0] += b[2] * dist
        b[1] += b[3] * dist
        if b[0] < FX0 + r and b[2] < 0 or b[0] > FX1 - r and b[2] > 0:
            b[2] = -b[2]
            b[0] = min(max(b[0], FX0 + r), FX1 - r)
        if b[1] < FY0 + r and b[3] < 0 or b[1] > FY1 - r and b[3] > 0:
            b[3] = -b[3]
            b[1] = min(max(b[1], FY0 + r), FY1 - r)
        c0, c1 = int((b[0] - r - FX0) // CELL), int((b[0] + r - FX0) // CELL)
        r0, r1 = int((b[1] - r - FY0) // CELL), int((b[1] + r - FY0) // CELL)
        best, nx_s, ny_s = None, 0.0, 0.0
        for rr in range(max(0, r0), min(ROWS - 1, r1) + 1):
            for cc in range(max(0, c0), min(COLS - 1, c1) + 1):
                if self.state[rr, cc] == b[4]:
                    continue
                x0, y0 = FX0 + cc * CELL, FY0 + rr * CELL
                px, py = min(max(b[0], x0), x0 + CELL), min(max(b[1], y0), y0 + CELL)
                dx, dy = b[0] - px, b[1] - py
                d2 = dx * dx + dy * dy
                if d2 >= r * r:
                    continue
                if d2 > 1e-9:
                    d = math.sqrt(d2)
                    nx, ny, pen = dx / d, dy / d, r - d
                else:
                    opts = [(b[0] - x0, -1, 0), (x0 + CELL - b[0], 1, 0), (b[1] - y0, 0, -1), (y0 + CELL - b[1], 0, 1)]
                    p0, nx, ny = min(opts)
                    pen = p0 + r
                nx_s += nx
                ny_s += ny
                if best is None or pen > best[0]:
                    best = (pen, rr, cc, nx, ny)
        if best is None:
            return
        pen, rr, cc, nx, ny = best
        b[0] += nx * pen
        b[1] += ny * pen
        L = math.hypot(nx_s, ny_s)
        if L > 1e-9:
            nx, ny = nx_s / L, ny_s / L
        vn = b[2] * nx + b[3] * ny
        if vn < 0:
            b[2] -= 2 * vn * nx
            b[3] -= 2 * vn * ny
        a = self.rng.uniform(-0.05, 0.05)   # winzige Streuung gegen Endlosschleifen
        vx, vy = b[2] * math.cos(a) - b[3] * math.sin(a), b[2] * math.sin(a) + b[3] * math.cos(a)
        for lim, i in ((0.15, 1), (0.15, 0)):  # keine ganz flachen Winkel
            v = [vx, vy]
            if abs(v[i]) < lim:
                v[i] = math.copysign(lim, v[i] if v[i] else 1.0)
            vx, vy = v
        L = math.hypot(vx, vy)
        b[2], b[3] = vx / L, vy / L
        # genau EIN Feld pro Berührung
        old = int(self.state[rr, cc])
        self.state[rr, cc] = b[4]
        self.flips.append((self.t, rr, cc, b[4], old))
        self.events.append((self.t, "flip", {"team": b[4]}))

    # ---------------------------------------------------------------- Schwung-Regel
    def targets(self):
        share = float((self.state == 0).mean())
        self.share_hist.append(share)
        if self.lag:
            share = self.share_hist[max(0, len(self.share_hist) - 1 - int(self.lag * FPS))]
        N = 2 + (self.n_max - 2) * min(1.0, self.t / self.n_ramp)
        p = self.power[0] + self.power[1] * self.t   # Schwung wird mit der Zeit stärker
        sa, sb = share, 1 - share
        if self.konter_t is not None:   # Konter-Bonus klingt langsam ab
            bonus = self.konter[1] * math.exp(-(self.t - self.konter_t) / self.konter[2])
            if self.konter_team == 0:
                sa += bonus
            else:
                sb += bonus
        wa, wb = sa ** p, sb ** p
        tgt = [N * wa / (wa + wb), N * wb / (wa + wb)]
        for team, sh in ((0, share), (1, 1 - share)):  # solange ein Team noch Gebiet hat, behält es einen Ball
            tgt[team] = max(tgt[team], 1.0) if sh >= self.ko else 0.0
        return tgt, share

    def alive(self, team):
        return [i for i, b in enumerate(self.balls) if b[5] and b[4] == team]

    def manage(self):
        # Konter: (Schwelle, Bonus, Abklingzeit[, bis Zeitpunkt]) – mit Ende-Zeit auch mehrfach möglich
        until = self.konter[3] if self.konter and len(self.konter) > 3 else None
        ready = self.konter_t is None if until is None else self.t < until
        if self.konter and ready and self.t > 6:
            share = float((self.state == 0).mean())
            for team, sh in ((0, share), (1, 1 - share)):
                if sh < self.konter[0] and not (self.konter_team == team and self.t - self.konter_t < 6):
                    self.konter_t, self.konter_team = self.t, team
                    self.events.append((self.t, "konter", {"team": team}))
                    for i in self.alive(team):   # jeder eigene Ball spaltet sich
                        src = self.balls[i]
                        a = math.atan2(src[3], src[2]) + math.pi * 0.5
                        self.balls.append([src[0], src[1], math.cos(a), math.sin(a), team, True])
                        self.births.append((self.t, len(self.balls) - 1))
                    self.last_change[team] = self.t
                    break
        tgt, share = self.targets()
        for team in (0, 1):
            ids = self.alive(team)
            if self.t - self.last_change[team] < 0.3 or not ids:
                continue
            if tgt[team] >= len(ids) + 0.6:  # Ball dazu: spaltet sich von einem eigenen Ball ab
                src = self.balls[self.rng.choice(ids)]
                a = math.atan2(src[3], src[2]) + self.rng.choice((-1, 1)) * self.rng.uniform(0.6, 1.4)
                self.balls.append([src[0], src[1], math.cos(a), math.sin(a), team, True])
                self.births.append((self.t, len(self.balls) - 1))
                self.events.append((self.t, "split", {"team": team}))
                self.last_change[team] = self.t
            elif tgt[team] <= len(ids) - 0.6:
                # der am stärksten eingekesselte Ball scheidet aus
                def own(i):
                    b = self.balls[i]
                    cc, rr = int((b[0] - FX0) // CELL), int((b[1] - FY0) // CELL)
                    return int((self.state[max(0, rr - 2):rr + 3, max(0, cc - 2):cc + 3] == team).sum())
                i = min(ids, key=own)
                self.balls[i][5] = False
                self.deaths.append((self.t, i))
                self.events.append((self.t, "pop", {"team": team}))
                self.last_change[team] = self.t
                if len(ids) == 1:
                    self.winner, self.win_t = 1 - team, self.t
                    self.events.append((self.t, "win", {}))

    def run(self):
        dt = 1 / FPS
        v0, v1, tr = self.ramp
        while True:
            p_ = min(1.0, self.t / tr)
            sp = v0 + (v1 - v0) * p_ * p_ * (3 - 2 * p_)
            if self.clear_t is not None:
                sp *= max(0.0, 1 - (self.t - self.clear_t) / 2.5) ** 1.3
            dist = sp * dt
            n_sub = max(1, int(math.ceil(dist / (self.r * 0.7))))
            for _ in range(n_sub):
                for b in self.balls:
                    if b[5]:
                        self.step_ball(b, dist / n_sub)
                self.t += dt / n_sub
            if self.winner is None:
                self.manage()
            elif self.clear_t is None and (self.state == self.winner).all():
                self.clear_t = self.t
                self.events.append((self.t, "clear", {}))
            self.frames.append((self.state.copy(), np.array([(b[0], b[1], b[4], b[5]) for b in self.balls], np.float32)))
            if self.clear_t is not None and self.t >= self.clear_t + 5.0:
                break
            if self.t > self.max_t:
                break
        self.total = len(self.frames) / FPS
        return self

    def lead_changes(self):
        lead, ch = None, 0
        for st, _ in self.frames[FPS * 3::6]:
            a = float((st == 0).mean())
            l = 0 if a > 0.55 else 1 if a < 0.45 else lead
            if l is not None and lead is not None and l != lead:
                ch += 1
            lead = l if l is not None else lead
        return ch


# ------------------------------------------------------------------ Darstellung

class PongFxRenderer:
    def __init__(self, sim):
        self.sim = sim
        self.tc, self.bc = sim.team_c, sim.ball_c
        k, s = SS, CELL * SS
        self.nt = nt = len(self.tc)
        self.tiles = [glossy_tile(c, s, BG) for c in self.tc]
        self.solid = [np.any(np.abs(t.astype(int) - np.array(BG)) > 12, axis=2) for t in self.tiles]
        self.full = [np.tile(self.tiles[i], (ROWS, COLS, 1)) for i in range(nt)]
        self.fsolid = [np.tile(self.solid[i], (ROWS, COLS)) for i in range(nt)]
        fy, fx = np.mgrid[0:ROWS * s, 0:COLS * s].astype(np.float32)
        self.diag = (fx + fy * 0.55) / s
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        vig = 0.45 + 0.55 * np.exp(-(((xx - W1 / 2) / W1) ** 2 + ((yy - H1 * 0.45) / H1) ** 2) * 3.0)
        self.base = (np.array((8, 5, 16), np.float32) * vig[..., None]).astype(np.uint8)
        self.flip_t = [f[0] for f in sim.flips]
        glow = getattr(sim, "ball_glow", None) or [self.tc[1 - i] for i in range(nt)]
        self.ball_spr = [self._ball_sprite(int(sim.r * k), self.bc[i], glow[i]) for i in range(nt)]
        yy, xx = np.mgrid[-24:24, -24:24].astype(np.float32)
        self.flash = np.exp(-(xx ** 2 + yy ** 2) / (2 * 8.0 ** 2))[..., None] * 0.8
        self.font = ImageFont.truetype("/usr/share/fonts/opentype/inter/Inter-Bold.otf", 26)
        # Partikel je umgefärbtem Feld
        P = []
        for t0, rr, cc, team, old in sim.flips:
            rng = random.Random(rr * 977 + cc * 31 + int(t0 * 1000))
            cx, cy = FX0 + (cc + 0.5) * CELL, FY0 + (rr + 0.5) * CELL
            col = self.tc[old]               # Splitter in der alten Farbe
            for _ in range(4):
                a = rng.uniform(0, 2 * math.pi)
                v = rng.uniform(80, 260)
                P.append((t0, cx, cy, math.cos(a) * v, math.sin(a) * v, rng.uniform(2.0, 3.6), *col))
        self.P = np.array(P, np.float32).reshape(-1, 9)

    @staticmethod
    def _ball_sprite(r, col, glow_c):
        n = r * 6
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2
        d = np.hypot(xx, yy)
        glow = np.clip(1 - d / (r * 2.6), 0, 1) ** 2 * 0.6
        core = np.clip(r - d + 0.5, 0, 1)
        hl = np.exp(-(((xx + r * 0.35) / (r * 0.35)) ** 2 + ((yy + r * 0.35) / (r * 0.35)) ** 2))
        rgb = np.zeros((n, n, 3), np.float32)
        rgb[:] = col
        rgb = rgb * (1 - core[..., None] * 0) + (np.array(glow_c, np.float32) - rgb) * ((1 - core) * 0.6)[..., None]
        rgb = rgb * (1 - 0.22 * core[..., None] * (yy / r).clip(0, 1)[..., None]) + hl[..., None] * 50
        a = np.maximum(glow, core)
        return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA")

    def frame(self, fi):
        sim = self.sim
        state, balls = sim.frames[fi]
        t = (fi + 1) / FPS
        k, s = SS, CELL * SS
        img = self.base.copy()
        field = np.zeros((ROWS * s, COLS * s, 3), np.uint8)
        field[:] = BG
        solid = np.zeros((ROWS * s, COLS * s), bool)
        for team in range(self.nt):
            big = np.repeat(np.repeat(state == team, s, 0), s, 1)
            field[big] = self.full[team][big]
            solid |= big & self.fsolid[team]
        sh = np.zeros_like(solid)                      # 3D: weicher Schatten
        sh[3 * k:, 2 * k:] = solid[:-3 * k, :-2 * k]
        sh &= ~solid
        field[sh] = (field[sh] * 0.5).astype(np.uint8)
        f = field.astype(np.float32)
        pos = ((t % 6.0) / 6.0) * (COLS + ROWS * 0.55 + 16) - 8
        band = np.exp(-((self.diag - pos) / 2.2) ** 2) * 0.28
        f += (band * solid)[..., None] * (255 - f)
        img[FY0 * k:FY1 * k, FX0 * k:FX1 * k] = np.clip(f, 0, 255).astype(np.uint8)
        im = Image.fromarray(img, "RGB")
        d = ImageDraw.Draw(im, "RGBA")
        # frisch eroberte Felder: wachsen aus der Mitte, leuchten kurz auf
        lo = bisect.bisect_left(self.flip_t, t - 0.3)
        hi = bisect.bisect_right(self.flip_t, t)
        flashes = []
        for t0, rr, cc, team, old in sim.flips[lo:hi]:
            e = (t - t0) / 0.3
            x0, y0 = (FX0 + cc * CELL) * k, (FY0 + rr * CELL) * k
            d.rectangle([x0, y0, x0 + s, y0 + s], fill=BG)
            m = s * 0.43 * (1 - e) ** 2
            c = tuple(int(v + (255 - v) * (1 - e) * 0.75) for v in self.tc[team])
            d.rounded_rectangle([x0 + m + s * 0.07, y0 + m + s * 0.07, x0 + s - m - s * 0.07, y0 + s - m - s * 0.07],
                                radius=max(1, int(s * 0.2)), fill=c)
            if e < 0.5:
                flashes.append((x0 / k + CELL / 2, y0 / k + CELL / 2, 1 - e / 0.5))
        # Partikel (alte Farbe splittert ab)
        P = self.P
        if len(P):
            i0, i1 = np.searchsorted(P[:, 0], t - 0.5), np.searchsorted(P[:, 0], t, side="right")
            Q = P[i0:i1]
            if len(Q) > 1600:
                Q = Q[np.linspace(0, len(Q) - 1, 1600).astype(int)]
            a = t - Q[:, 0]
            px = Q[:, 1] + Q[:, 3] * (1 - np.exp(-a * 6)) / 6
            py = Q[:, 2] + Q[:, 4] * (1 - np.exp(-a * 6)) / 6 + 50 * a * a
            life = 1 - a / 0.5
            ok = (px > FX0 + 3) & (px < FX1 - 3) & (py > FY0 + 3) & (py < FY1 - 3)
            for x, y, z, l, r_, g_, b_ in zip(px[ok] * k, py[ok] * k, Q[ok, 5] * k * (0.4 + 0.6 * life[ok]), life[ok],
                                              Q[ok, 6], Q[ok, 7], Q[ok, 8]):
                wb = 0.45 * l
                d.rounded_rectangle([x - z, y - z, x + z, y + z], radius=z * 0.4,
                                    fill=(int(r_ + (255 - r_) * wb), int(g_ + (255 - g_) * wb), int(b_ + (255 - b_) * wb),
                                          int(255 * min(1, l * 1.4))))
        # Geburt (Abspaltung) und Ausscheiden: Ring
        for lst, col_i in ((sim.births, None), (sim.deaths, None)):
            for t0, i in lst:
                e = (t - t0) / 0.6
                if 0 <= e < 1:
                    x, y, team = balls[i][0], balls[i][1], int(balls[i][2])
                    R = (sim.r + 26 * e) * k
                    col = self.bc[team] if lst is sim.births else (255, 255, 255)
                    d.ellipse([x * k - R, y * k - R, x * k + R, y * k + R], outline=col + (int(230 * (1 - e)),), width=3 * k)
        rgba = im.convert("RGBA")
        for i, (x, y, team, alive) in enumerate(balls):
            sp = self.ball_spr[int(team)]
            if not alive:
                dt_ = next((t - t0 for t0, j in sim.deaths if j == i), 9)
                e = dt_ / 0.35
                if e >= 1:
                    continue
                sc = 1 + 0.9 * e
                sp = sp.resize((max(2, int(sp.width * sc)), max(2, int(sp.height * sc))), Image.BILINEAR)
                a = np.asarray(sp).copy()
                a[..., 3] = (a[..., 3] * (1 - e)).astype(np.uint8)
                sp = Image.fromarray(a, "RGBA")
            rgba.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        out = rgba.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32).copy()
        h = self.flash.shape[0] // 2
        for x, y, a in flashes[-300:]:
            X, Y = int(x), int(y)
            y0, y1, x0, x1 = max(0, Y - h), min(H1, Y + h), max(0, X - h), min(W1, X + h)
            arr[y0:y1, x0:x1] += self.flash[y0 - Y + h:y1 - Y + h, x0 - X + h:x1 - X + h] * a * 255
        g = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize((W1 // 5, H1 // 5), Image.BILINEAR)
        glow = np.asarray(g.filter(ImageFilter.GaussianBlur(4)).resize((W1, H1), Image.BILINEAR), np.float32)
        arr = arr + np.clip(glow - 40, 0, 255) * 0.4
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        self.hud(im, state, balls)
        if sim.names:
            self.story(im, state, t)
        return im

    def story(self, im, state, t):
        """Prozent-Anzeige oben, Text-Hook am Anfang, Konter-Einblendung, Sieger am Ende."""
        sim = self.sim
        d = ImageDraw.Draw(im, "RGBA")
        F = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
        share = float((state == 0).mean())
        pc = [round(share * 100), 100 - round(share * 100)]
        f = ImageFont.truetype(F, 38)
        y = FY0 - 36
        d.text((FX0 + 4, y), f"{sim.names[0]} {pc[0]}%", font=f, fill=self.tc[0] + (255,), anchor="lm",
               stroke_width=3, stroke_fill=(0, 0, 0, 200))
        d.text((FX1 - 4, y), f"{pc[1]}% {sim.names[1]}", font=f, fill=self.tc[1] + (255,), anchor="rm",
               stroke_width=3, stroke_fill=(0, 0, 0, 200))

        def banner(lines, alpha, cy, scale=1.0):
            if alpha <= 0.01:
                return
            hs = [int(sz * scale) for _, sz, _ in lines]
            fonts = [ImageFont.truetype(F, h) for h in hs]
            ws = [d.textlength(txt, font=fo) for (txt, _, _), fo in zip(lines, fonts)]
            H = sum(hs) * 1.25 + 40
            W = max(ws) + 70
            cx = (FX0 + FX1) / 2
            d.rounded_rectangle([cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2], radius=28,
                                fill=(6, 4, 14, int(235 * alpha)), outline=(255, 255, 255, int(60 * alpha)), width=2)
            yy = cy - H / 2 + 20
            for (txt, _, col), fo, h in zip(lines, fonts, hs):
                if isinstance(col, list):   # mehrfarbig: [(teil, farbe), ...]
                    x = cx - d.textlength(txt, font=fo) / 2
                    for part, c in col:
                        c = tuple(int(v + (255 - v) * 0.2) for v in c)
                        d.text((x, yy), part, font=fo, fill=c + (int(255 * alpha),), anchor="la")
                        x += d.textlength(part, font=fo)
                else:
                    col = tuple(int(v + (255 - v) * 0.2) for v in col)
                    d.text((cx, yy), txt, font=fo, fill=col + (int(255 * alpha),), anchor="ma",
                           stroke_width=2, stroke_fill=(255, 255, 255, int(90 * alpha)))
                yy += h * 1.25

        # Hook in den ersten Sekunden
        a = min(1.0, t / 0.25) * min(1.0, max(0.0, (3.4 - t) / 0.5))
        n0, n1 = sim.names
        banner([("Welche Farbe gewinnt?", 64, (255, 255, 255)),
                (f"{n0} vs {n1}", 50, [(n0, self.tc[0]), (" vs ", (255, 255, 255)), (n1, self.tc[1])])],
               a, (FY0 + FY1) / 2 - 40)
        # Sieger
        if sim.clear_t is not None and t > sim.clear_t + 0.4:
            a = min(1.0, (t - sim.clear_t - 0.4) / 0.6)
            w = sim.winner
            banner([(f"{sim.names[w]} GEWINNT!", 66, self.tc[w])], a, (FY0 + FY1) / 2 - 40)

    def hud(self, im, state, balls):
        """Schmaler Kräfte-Balken unter dem Feld + kleine Ball-Zähler (Randinfo)."""
        d = ImageDraw.Draw(im, "RGBA")
        share = float((state == 0).mean())
        x0, x1, y = FX0 + 70, FX1 - 70, FY1 + 26
        w = x1 - x0
        d.rounded_rectangle([x0, y - 6, x1, y + 6], radius=6, fill=self.tc[1])
        if share > 0.02:
            d.rounded_rectangle([x0, y - 6, x0 + w * share, y + 6], radius=6, fill=self.tc[0])
        d.line([x0 + w / 2, y - 11, x0 + w / 2, y + 11], fill=(255, 255, 255, 160), width=2)
        for team, cx in ((0, FX0 + 30), (1, FX1 - 30)):
            n = int(sum(1 for b in balls if int(b[2]) == team and b[3]))
            d.ellipse([cx - 22, y - 7 - 0, cx - 8, y + 7], fill=self.tc[team] + (255,))
            d.text((cx - 3, y), f"{n}", font=self.font, fill=(255, 255, 255, 230), anchor="lm")


# ------------------------------------------------------------------ Battle Royale: 4 Farben, eine gewinnt

class PongRoyale(PongFx):
    """Vier Farben starten in den Ecken. Gleiche Regeln (ein Block pro Berührung, Schwung über die Fläche).
    Fällt eine Farbe unter `split_at`, spalten sich ihre Bälle einmal (bis `split_until`) -> Aufholjagden.
    Wer unter `ko` Fläche fällt, verliert seine Bälle und ist raus. Die letzte Farbe gewinnt."""

    def __init__(self, seed, team_c, names, n_max=48, n_ramp=30.0, power=(1.0, 0.35), ramp=(560.0, 1700.0, 24.0),
                 ko=0.03, split_at=0.12, split_until=34.0, bonus=(0.25, 6.0), grid=(2, 2),
                 final=None, **kw):
        nt = len(team_c)
        super().__init__(seed, power=power, n_max=n_max, n_ramp=n_ramp, ramp=ramp, ko=ko, team_c=team_c,
                         ball_c=[(255, 255, 255)] * nt, names=names, **kw)
        self.ball_glow = [tuple(int(v * 0.55) for v in c) for c in team_c]
        self.nt = nt
        gr, gc = grid    # Startgebiete: gr Reihen x gc Spalten, zeilenweise nummeriert
        rb = [round(ROWS * i / gr) for i in range(gr + 1)]
        cb = [round(COLS * j / gc) for j in range(gc + 1)]
        self.balls = []
        cx, cy = (FX0 + FX1) / 2, (FY0 + FY1) / 2
        for team in range(nt):
            i, j = divmod(team, gc)
            self.state[rb[i]:rb[i + 1], cb[j]:cb[j + 1]] = team
            x = FX0 + (cb[j] + cb[j + 1]) / 2 * CELL
            y = FY0 + (rb[i] + rb[i + 1]) / 2 * CELL
            a = math.atan2(cy - y, cx - x) + self.rng.uniform(-0.5, 0.5)
            self.balls.append([x, y, math.cos(a), math.sin(a), team, True])
        self.last_change = [-9.0] * nt
        self.out = {}                 # Team -> Zeitpunkt des Ausscheidens
        self.split_at, self.split_until, self.bonus = split_at, split_until, bonus
        self.split_t = {}             # Team -> letzter Aufhol-Split
        # Finale (2 Farben übrig): Schwung startet neu, Schwächere darf einmal spalten. (Schwelle, Bonus, Abkling)
        self.final, self.final_t, self.final_split = final, None, set()

    def shares(self):
        return [float((self.state == k).mean()) for k in range(self.nt)]

    def manage(self):
        sh = self.shares()
        alive_teams = [k for k in range(self.nt) if k not in self.out]
        if self.final and self.final_t is None and len(alive_teams) == 2:
            self.final_t = self.t
            self.events.append((self.t, "final", {}))
        if self.final_t is not None:
            for k in alive_teams:
                if sh[k] < self.final[0] and k not in self.final_split:
                    self.final_split.add(k)
                    self.split_t[k] = self.t
                    self.events.append((self.t, "konter", {"team": k}))
                    for i in self.alive(k):
                        src = self.balls[i]
                        a = math.atan2(src[3], src[2]) + math.pi * 0.5
                        self.balls.append([src[0], src[1], math.cos(a), math.sin(a), k, True])
                        self.births.append((self.t, len(self.balls) - 1))
                    self.last_change[k] = self.t
        # Aufholjagd: kleine Farbe spaltet einmal alle Bälle
        if self.t < self.split_until and self.t > 6:
            for k in alive_teams:
                if sh[k] < self.split_at and k not in self.split_t and len(alive_teams) > 2:
                    self.split_t[k] = self.t
                    self.events.append((self.t, "konter", {"team": k}))
                    for i in self.alive(k):
                        src = self.balls[i]
                        a = math.atan2(src[3], src[2]) + math.pi * 0.5
                        self.balls.append([src[0], src[1], math.cos(a), math.sin(a), k, True])
                        self.births.append((self.t, len(self.balls) - 1))
                    self.last_change[k] = self.t
        N = self.nt + (self.n_max - self.nt) * min(1.0, self.t / self.n_ramp)
        if self.final_t is None:
            p = self.power[0] + self.power[1] * self.t
        else:   # Finale: Schwung startet bei 1 und steigt (eigene Steigung)
            p = 1.0 + (self.final[3] if len(self.final) > 3 else self.power[1]) * (self.t - self.final_t)
        eff = []
        for k in range(self.nt):
            e = sh[k]
            if k in self.split_t:
                b0, b1 = (self.final[1], self.final[2]) if k in self.final_split else self.bonus
                e += b0 * math.exp(-(self.t - self.split_t[k]) / b1)
            eff.append(e if k in alive_teams else 0.0)
        wts = [e ** p if e > 0 else 0.0 for e in eff]
        W = sum(wts) or 1.0
        tgt = [N * w_ / W for w_ in wts]
        for k in range(self.nt):
            tgt[k] = max(tgt[k], 1.0) if (k in alive_teams and sh[k] >= self.ko) else 0.0
        for k in alive_teams:
            ids = self.alive(k)
            if self.t - self.last_change[k] < 0.25 or not ids:
                continue
            if tgt[k] >= len(ids) + 0.6:
                src = self.balls[self.rng.choice(ids)]
                a = math.atan2(src[3], src[2]) + self.rng.choice((-1, 1)) * self.rng.uniform(0.6, 1.4)
                self.balls.append([src[0], src[1], math.cos(a), math.sin(a), k, True])
                self.births.append((self.t, len(self.balls) - 1))
                self.events.append((self.t, "split", {"team": k}))
                self.last_change[k] = self.t
            elif tgt[k] <= len(ids) - 0.6:
                def own(i):
                    b = self.balls[i]
                    cc, rr = int((b[0] - FX0) // CELL), int((b[1] - FY0) // CELL)
                    return int((self.state[max(0, rr - 2):rr + 3, max(0, cc - 2):cc + 3] == k).sum())
                i = min(ids, key=own)
                self.balls[i][5] = False
                self.deaths.append((self.t, i))
                self.events.append((self.t, "pop", {"team": k}))
                self.last_change[k] = self.t
                if len(ids) == 1:
                    self.out[k] = self.t
                    self.events.append((self.t, "out", {"team": k}))
                    rest = [j for j in range(self.nt) if j not in self.out]
                    if len(rest) == 1:
                        self.winner, self.win_t = rest[0], self.t
                        self.events.append((self.t, "win", {}))


class RoyaleRenderer(PongFxRenderer):
    def hud(self, im, state, balls):
        """Gestapelter Balken unter dem Feld: Anteil jeder Farbe."""
        d = ImageDraw.Draw(im, "RGBA")
        x0, x1, y = FX0 + 10, FX1 - 10, FY1 + 26
        w = x1 - x0
        x = x0
        for k in range(self.nt):
            sh = float((state == k).mean())
            if sh > 0.004:
                d.rectangle([x, y - 6, x + w * sh, y + 6], fill=self.tc[k])
            x += w * sh

    def story(self, im, state, t):
        sim = self.sim
        d = ImageDraw.Draw(im, "RGBA")
        F = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
        nt = self.nt
        gap = 12 if nt <= 4 else 6
        fn, fp = (ImageFont.truetype(F, 21), ImageFont.truetype(F, 34)) if nt <= 4 else \
            (ImageFont.truetype(F, 15), ImageFont.truetype(F, 27))
        # Anzeigen oben: Name + Prozent, ausgeschieden = abgedunkelt mit X
        pw = (FX1 - FX0 - (nt - 1) * gap) / nt
        for k in range(nt):
            x0 = FX0 + k * (pw + gap)
            y0, y1 = FY0 - 78, FY0 - 8
            out = k in sim.out and t >= sim.out[k]
            c = self.tc[k]
            d.rounded_rectangle([x0, y0, x0 + pw, y1], radius=14 if nt <= 4 else 10, fill=(0, 0, 0, 170),
                                outline=c + (90 if out else 230,), width=3)
            pc = round(float((state == k).mean()) * 100)
            a = 90 if out else 255
            d.text((x0 + pw / 2, y0 + 15), sim.names[k], font=fn, fill=c + (a,), anchor="mm")
            d.text((x0 + pw / 2, y0 + 48), "OUT" if out else f"{pc}%", font=fp,
                   fill=(255, 255, 255, a) if not out else (200, 200, 200, 120), anchor="mm")
            if out:
                e = min(1.0, (t - sim.out[k]) / 0.3)
                m_ = 14 if nt <= 4 else 8
                d.line([x0 + m_, y0 + 10, x0 + m_ + (pw - 2 * m_) * e, y1 - 10], fill=c + (220,), width=4)

        def banner(lines, alpha, cy):
            if alpha <= 0.01:
                return
            fonts = [ImageFont.truetype(F, sz) for _, sz, _ in lines]
            ws = [d.textlength(txt, font=fo) for (txt, _, _), fo in zip(lines, fonts)]
            H = sum(sz for _, sz, _ in lines) * 1.25 + 40
            W = max(ws) + 70
            cx = (FX0 + FX1) / 2
            d.rounded_rectangle([cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2], radius=28,
                                fill=(6, 4, 14, int(235 * alpha)), outline=(255, 255, 255, int(60 * alpha)), width=2)
            yy = cy - H / 2 + 20
            for (txt, sz, col), fo in zip(lines, fonts):
                if isinstance(col, list):
                    x = cx - d.textlength(txt, font=fo) / 2
                    for part, c in col:
                        d.text((x, yy), part, font=fo, fill=c + (int(255 * alpha),), anchor="la")
                        x += d.textlength(part, font=fo)
                else:
                    d.text((cx, yy), txt, font=fo, fill=col + (int(255 * alpha),), anchor="ma")
                yy += sz * 1.25

        a = min(1.0, t / 0.25) * min(1.0, max(0.0, (3.6 - t) / 0.5))
        if nt <= 4:
            parts = []
            for k in range(nt):
                parts.append((sim.names[k], self.tc[k]))
                if k < nt - 1:
                    parts.append(("  ", (255, 255, 255)))
            banner([("Wähl deine Farbe!", 66, (255, 255, 255)), ("".join(p_ for p_, _ in parts), 36, parts),
                    ("Nur eine überlebt.", 34, (200, 200, 215))], a, (FY0 + FY1) / 2)
        else:
            dots = []
            for k in range(nt):
                dots.append(("●", self.tc[k]))
            banner([("Wähl deine Farbe!", 66, (255, 255, 255)), ("".join(p_ for p_, _ in dots), 44, dots),
                    (f"{nt} Farben. Nur eine überlebt.", 34, (200, 200, 215))], a, (FY0 + FY1) / 2)
        if sim.clear_t is not None and t > sim.clear_t + 0.4:
            a = min(1.0, (t - sim.clear_t - 0.4) / 0.6)
            w = sim.winner
            c = tuple(int(v + (255 - v) * 0.15) for v in self.tc[w])
            banner([(f"{sim.names[w]} GEWINNT!", 66, c)], a, (FY0 + FY1) / 2)


PongRoyale.renderer = RoyaleRenderer


# ------------------------------------------------------------------ Klang: nur weiches Ploppen

def build_audio(sim, path):
    total = sim.total
    n = int(SR * total)
    out = np.zeros(n + SR * 2)
    rng = random.Random(3)
    last = [-1.0] * 8
    pitches = [(600, 640, 680, 720), (430, 460, 490, 520), (520, 555, 590, 625), (370, 395, 420, 445)]

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    for t, kind, d in sim.events:
        if kind == "flip":
            tm = d["team"]
            if t - last[tm] < 0.045:
                continue
            last[tm] = t
            f = rng.choice(pitches[tm % 4])
            put(t, plop(f), 0.5)
        elif kind == "split":
            put(t, plop(330, big=True), 0.7)
        elif kind == "pop":
            put(t, plop(220, big=True), 0.9)
        elif kind == "out":   # Farbe scheidet aus: tiefes Doppel-Plopp
            put(t, plop(160, big=True), 1.0)
            put(t + 0.12, plop(120, big=True), 0.9)
        elif kind == "clear":
            for j, f in enumerate((300, 360, 450)):
                put(t + j * 0.12, plop(f, big=True), 0.6)
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
    _R = getattr(sim, "renderer", PongFxRenderer)(sim)
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


SEED_PONG = 7
ORANGE_BLAU = dict(team_c=[(255, 135, 30), (45, 120, 255)], ball_c=[(150, 205, 255), (255, 200, 120)],
                   names=["ORANGE", "BLAU"])
PINK_GRUEN = dict(team_c=[(255, 55, 160), (50, 225, 105)], ball_c=[(175, 255, 195), (255, 185, 225)],
                  names=["PINK", "GRÜN"])
ROT_GELB = dict(team_c=[(240, 40, 60), (255, 205, 30)], ball_c=[(255, 240, 160), (190, 20, 50)],
                names=["ROT", "GELB"])
ROYALE = dict(team_c=[(255, 105, 90), (60, 235, 180), (110, 95, 255), (255, 195, 50)],
              names=["KORALLE", "MINT", "INDIGO", "GOLD"])
REGENBOGEN = dict(team_c=[(255, 60, 70), (60, 120, 255), (255, 220, 40), (255, 80, 190), (50, 230, 170),
                          (255, 140, 30), (160, 80, 255), (150, 235, 40)],
                  names=["ROT", "BLAU", "GELB", "PINK", "MINT", "ORANGE", "LILA", "LIME"])
NEON6 = dict(team_c=[(0, 215, 255), (255, 40, 200), (255, 225, 40), (140, 80, 255), (60, 230, 90), (255, 120, 20)],
             names=["CYAN", "MAGENTA", "GELB", "LILA", "GRÜN", "ORANGE"])
PONG = {"p01_pong_block_battle": lambda: PongFx(SEED_PONG).run(),
        "p02_orange_vs_blau": lambda: PongFx(12, power=(1.0, 0.36), split="lr", konter=(0.25, 0.95, 9.0),
                                             ramp=(560.0, 1450.0, 24.0), **ORANGE_BLAU).run(),
        # Hin und Her: wer unter 37 % fällt, spaltet alle Bälle (bis 30 s) -> Führungswechsel, dann entscheidet der Schwung
        "p03_hin_und_her": lambda: PongFx(27, power=(1.0, 0.6), split="lr", konter=(0.37, 0.95, 5.5, 30.0),
                                          ramp=(560.0, 1700.0, 24.0), **ORANGE_BLAU).run(),
        "p04_pink_vs_gruen": lambda: PongFx(20, power=(1.0, 0.6), split="lr", konter=(0.37, 0.95, 5.5, 30.0),
                                            ramp=(560.0, 1700.0, 24.0), **PINK_GRUEN).run(),
        "p05_rot_vs_gelb": lambda: PongFx(4, power=(1.0, 0.6), split="lr", konter=(0.37, 0.95, 5.5, 30.0),
                                          ramp=(560.0, 1700.0, 24.0), **ROT_GELB).run(),
        "p06_battle_royale": lambda: PongRoyale(SEED_ROYALE, power=(1.0, 0.22), n_ramp=40.0, split_until=42.0,
                                                split_at=0.15, bonus=(0.45, 9.0), **ROYALE).run()}
SEED_ROYALE = 13
PONG["p07_regenbogen_royale"] = lambda: PongRoyale(SEED_RB, grid=(4, 2), **RB_KW, **REGENBOGEN).run()
RB_KW = dict(n_max=64, n_ramp=40.0, power=(1.0, 0.22), split_at=0.06, split_until=40.0, bonus=(0.25, 9.0), ko=0.015)
SEED_RB = 0
N6_KW = dict(n_max=56, n_ramp=40.0, power=(1.0, 0.2), split_at=0.08, split_until=40.0, bonus=(0.3, 9.0), ko=0.02)
PONG["p08_neon6_royale"] = lambda: PongRoyale(SEED_N6, grid=(3, 2), **N6_KW, **NEON6).run()
SEED_N6 = 41

if __name__ == "__main__":
    import sys
    preview = "--preview" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(PONG)
    for name in names:
        sim = PONG[name]()
        print(name, "Dauer", round(sim.total, 1), "Sieger", sim.winner, "bei", sim.win_t, "fertig", sim.clear_t)
        if preview:
            ts = [1.5, 4.0, 10.0, 16.0, 22.0, 28.0, 34.0, 40.0, 46.0, sim.total - 4, sim.total - 0.1]
            render(sim, name, None, preview_dir="output/preview", preview_times=ts)
        else:
            out = f"output/{name}_vertical.mp4"
            render(sim, name, out)
            print(out)
