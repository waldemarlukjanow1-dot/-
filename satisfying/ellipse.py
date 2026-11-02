"""Ellipsen-Brennpunkt-Loop: alle Bälle starten in einem Brennpunkt, prallen einmal an der Ellipse ab und treffen sich
exakt gleichzeitig im anderen Brennpunkt (jeder Weg Brennpunkt -> Wand -> Brennpunkt ist genau 2a lang).
Bei jedem Treffen vermehren sich die Bälle und fächern als Regenbogen wieder auf. Immer weiter, hin und her.

Die Bahnen werden exakt berechnet (keine numerische Physik): Richtung theta -> Wandpunkt P -> anderer Brennpunkt.
Die Spur zeigt den Weg seit dem letzten Brennpunkt (Brennpunkt -> P -> Ball).
"""
import colorsys
import math
import os
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

CX, CY = 520.0, 880.0
A, B = 470.0, 400.0                 # Halbachsen (vertikal, horizontal) – fast rund
C = math.sqrt(A * A - B * B)
FOCI = [(CX, CY - C), (CX, CY + C)]  # oben, unten


def hit_point(f, th):
    """Schnitt des Strahls vom Brennpunkt f in Richtung th mit der Ellipse."""
    dx, dy = math.cos(th), math.sin(th)
    ox, oy = f[0] - CX, f[1] - CY
    a_ = (dx / B) ** 2 + (dy / A) ** 2
    b_ = 2 * (ox * dx / B ** 2 + oy * dy / A ** 2)
    c_ = (ox / B) ** 2 + (oy / A) ** 2 - 1
    t = (-b_ + math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
    return f[0] + dx * t, f[1] + dy * t, t


NEON = [(255, 30, 150), (190, 40, 255), (40, 120, 255), (0, 240, 255), (40, 255, 120), (230, 255, 30),
        (255, 150, 0)]


def neon(v):
    """Neonfarbe im Kreis über die Palette (v in 0..1)."""
    v = (v % 1.0) * len(NEON)
    i = int(v)
    f = v - i
    a, b = NEON[i % len(NEON)], NEON[(i + 1) % len(NEON)]
    return tuple(int(x + (y - x) * f) for x, y in zip(a, b))


def hue(v, s=0.75, l=1.0):
    r, g, b = colorsys.hsv_to_rgb(v % 1.0, s, l)
    return int(r * 255), int(g * 255), int(b * 255)


class EllipseLoop:
    def __init__(self, counts=(2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024),
                 t_first=3.1, t_min=2.25, shrink=0.97, lead=0.9, outro=4.0):
        self.trips = []      # (t_start, dauer, von_brennpunkt, winkel[], farbe[])
        t = lead
        T = t_first
        for k, n in enumerate(counts):
            src = k % 2
            off = 0.5 * math.pi / n + 0.37 * k          # leicht gedreht je Runde
            if k == 0:
                ths = [math.radians(-35), math.radians(140)]   # wie im Original: zwei Richtungen
            else:
                ths = [off + 2 * math.pi * j / n for j in range(n)]
            cols = [neon((th % (2 * math.pi)) / (2 * math.pi) + 0.02 * k) for th in ths]
            hits = [hit_point(FOCI[src], th) for th in ths]
            self.trips.append((t, T, src, ths, cols, hits))
            t += T
            T = max(t_min, T * shrink)
        self.end_t = t
        self.lead, self.outro = lead, outro
        self.total = t + outro
        self.n_frames = int(self.total * FPS)
        self.meets = [tr[0] + tr[1] for tr in self.trips]       # Ankunftszeiten
        # Wandtreffer (für Klang)
        self.events = []
        for k, (t0, T, src, ths, cols, hits) in enumerate(self.trips):
            for th, (px, py, d1) in zip(ths, hits):
                self.events.append((t0 + T * d1 / (2 * A), "wall", {"th": th, "k": k}))
            self.events.append((t0 + T, "meet", {"k": k, "n": len(ths)}))
        self.events.sort(key=lambda e: e[0])

    def state(self, t):
        """Aktive Runde und Fortschritt (0..1)."""
        if t < self.lead:
            return 0, 0.0
        for k, (t0, T, *_rest) in enumerate(self.trips):
            if t < t0 + T:
                return k, (t - t0) / T
        return len(self.trips) - 1, 1.0


# ------------------------------------------------------------------ Darstellung

def _font(path, size, cache={}):
    key = (path, size)
    if key not in cache:
        cache[key] = ImageFont.truetype(path, size)
    return cache[key]


def dot_sprite(r, col):
    n = int(r * 6)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2 + 0.5
    d = np.hypot(xx, yy)
    core = np.clip(r - d + 0.7, 0, 1)
    glow = np.clip(1 - d / (r * 3.0), 0, 1) ** 2 * 0.6
    c = np.array(col, np.float32)
    hl = np.exp(-(((xx + r * 0.3) / (r * 0.45)) ** 2 + ((yy + r * 0.35) / (r * 0.45)) ** 2))
    rgb = c[None, None] * np.ones((n, n, 1)) + hl[..., None] * (255 - c) * 0.7
    a = np.maximum(glow, core)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA")


class LoopRenderer:
    def __init__(self, sim, bg=None, ring_w=5, ball_scale=1.0, intro=None):
        self.sim = sim
        self.intro = intro          # (Text, Dauer): Einblendung vor dem Start, ohne Bälle
        self.bg, self.ball_scale = bg, ball_scale
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        vig = 0.35 + 0.65 * np.exp(-(((xx - CX) / 700) ** 2 + ((yy - CY) / 1000) ** 2))
        base = np.array((14, 10, 18), np.float32) * vig[..., None]
        self.base = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
        if bg is not None:   # animierter Hintergrund: hier nur das Innere abdunkeln (Kontrast für die Spuren)
            m = Image.new("L", (W1 * k, H1 * k), 0)
            ImageDraw.Draw(m).ellipse([(CX - B) * k, (CY - A) * k, (CX + B) * k, (CY + A) * k], fill=205)
            m = m.filter(ImageFilter.GaussianBlur(6 * k))
            self.base = Image.new("RGBA", (W1 * k, H1 * k), (10, 6, 22, 0))
            self.base.putalpha(m)
        # Ellipse mit weichem Glow
        ring = Image.new("L", (W1 * k, H1 * k), 0)
        ImageDraw.Draw(ring).ellipse([(CX - B) * k, (CY - A) * k, (CX + B) * k, (CY + A) * k], outline=255,
                                     width=ring_w * k)
        gl = ring.filter(ImageFilter.GaussianBlur(12 * k))
        arr = np.asarray(self.base, np.float32).copy()
        # Neon-Rand: Farbe läuft im Kreis um die Ellipse
        ang = (np.arctan2(yy - CY, xx - CX) / (2 * np.pi)) % 1.0
        lut = np.array([neon(i / 256) for i in range(256)], np.float32)
        colr = lut[(ang * 255).astype(np.int32)]
        rg, rr_ = np.asarray(gl, np.float32) / 255, np.asarray(ring, np.float32) / 255
        arr[..., :3] += colr * (rg * 1.6)[..., None] + (colr * 0.55 + 255 * 0.45) * rr_[..., None]
        if bg is not None:
            arr[..., 3] = np.maximum(arr[..., 3], np.clip(rg * 1.6 + rr_, 0, 1) * 255)
        self.base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        self.sprites = {}

    def sprite(self, r, col):
        key = (round(r * 2) / 2, col)
        if key not in self.sprites:
            self.sprites[key] = dot_sprite(key[0] * SS, col)
        return self.sprites[key]

    def intro_frame(self, t):
        """Intro: nur Hintergrund + großer Text mit Neon-Herz."""
        text, dur = self.intro
        im = Image.fromarray(self.bg.frame(t)).convert("RGBA")
        dark = Image.new("RGBA", im.size, (8, 4, 20, 90))
        im.alpha_composite(dark)
        a = min(1.0, t / 0.25)
        pop = 1 + 0.18 * max(0.0, 1 - t / 0.35)
        beat = 1 + 0.06 * max(0.0, math.sin(t * 2 * math.pi * 1.2)) ** 6
        lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        cx, cy = CX, CY - 60
        r = 80 * pop * beat
        heart = []
        for q in range(80):
            u = q / 80 * 2 * math.pi
            hx = 16 * math.sin(u) ** 3
            hy = 13 * math.cos(u) - 5 * math.cos(2 * u) - 2 * math.cos(3 * u) - math.cos(4 * u)
            heart.append((cx + hx * r / 16, cy - 40 - hy * r / 16))
        d.polygon(heart, fill=(255, 40, 140, int(255 * a)))
        f = _font(F_BOLD, int(118 * pop))
        d.text((cx, cy + 120), text.split(" ")[0], font=f, fill=(255, 255, 255, int(255 * a)), anchor="mm",
               stroke_width=6, stroke_fill=(60, 0, 60, int(220 * a)))
        d.text((cx, cy + 225), " ".join(text.split(" ")[1:]).upper(), font=_font(F_BOLD, int(74 * pop)),
               fill=(255, 90, 190, int(255 * a)), anchor="mm", stroke_width=5, stroke_fill=(60, 0, 60, int(220 * a)))
        glow = lay.filter(ImageFilter.GaussianBlur(18))
        im.alpha_composite(glow)
        im.alpha_composite(glow)
        im.alpha_composite(lay)
        return im.convert("RGB")

    def frame(self, fi):
        t = (fi + 1) / FPS
        if self.intro is not None:
            dur = self.intro[1]
            if t < dur:
                return self.intro_frame(t)
            main = self._frame(fi)
            e = (t - dur) / 0.45
            if e < 1:   # weicher Übergang
                return Image.blend(self.intro_frame(t), main, e)
            return main
        return self._frame(fi)

    def _frame(self, fi):
        sim = self.sim
        k = SS
        t = (fi + 1) / FPS
        if self.bg is not None:
            im = Image.fromarray(self.bg.frame(t)).resize((W1 * k, H1 * k), Image.BILINEAR).convert("RGBA")
            im.alpha_composite(self.base)
        else:
            im = self.base.copy()
        lines = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(lines, "RGBA")
        d = ImageDraw.Draw(im, "RGBA")
        kk, u = sim.state(t)
        t0, T, src, ths, cols, hits = sim.trips[kk]
        fade = 1.0
        if t > sim.end_t:              # Ausklang: alles im Ziel, Spuren verblassen
            fade = max(0.0, 1 - (t - sim.end_t) / 2.5)
        n = len(ths)
        F0, F1 = FOCI[src], FOCI[1 - src]
        s = u * 2 * A
        lw = max(1, int((3.0 if n < 40 else 2.2 if n < 200 else 1.4) * k))
        la = int((230 if n < 40 else 170 if n < 200 else 110 if n < 600 else 80) * fade)
        rr = (9.0 if n < 40 else 7.0 if n < 200 else 5.0 if n < 600 else 4.0) * self.ball_scale
        pos = []
        for th, col, (px, py, d1) in zip(ths, cols, hits):
            if s <= d1:
                bx, by = F0[0] + math.cos(th) * s, F0[1] + math.sin(th) * s
                pts = [F0, (bx, by)]
            else:
                rest = s - d1
                vx, vy = F1[0] - px, F1[1] - py
                L = math.hypot(vx, vy)
                bx, by = px + vx / L * rest, py + vy / L * rest
                pts = [F0, (px, py), (bx, by)]
            ld.line([(x * k, y * k) for x, y in pts], fill=col + (la,), width=lw, joint="curve")
            pos.append((bx, by, col))
        # Spuren weich leuchten lassen
        g = lines.resize((W1 // 2, H1 // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3)).resize(im.size)
        im.alpha_composite(g)
        im.alpha_composite(lines)
        d = ImageDraw.Draw(im, "RGBA")
        # Brennpunkte
        for fx, fy in FOCI:
            R = 13 * k
            d.ellipse([fx * k - R, fy * k - R, fx * k + R, fy * k + R], outline=(230, 250, 255, 255), width=3 * k,
                      fill=(14, 10, 18, 255))
        # Treffen: nur ein kurzes, weiches Aufleuchten des Brennpunkts
        for j, mt in enumerate(sim.meets):
            e = t - mt
            if 0 <= e < 0.35:
                fx, fy = FOCI[1 - (j % 2)]
                R = (16 + 10 * e) * k
                d.ellipse([fx * k - R, fy * k - R, fx * k + R, fy * k + R],
                          fill=(255, 255, 255, int(120 * (1 - e / 0.35))))
        # Bälle
        if fade > 0:
            for bx, by, col in pos:
                spr = self.sprite(rr, col)
                im.alpha_composite(spr, (int(bx * k - spr.width / 2), int(by * k - spr.height / 2)))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        gg = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(gg, np.float32) - 30, 0, 255) * 0.45
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        # kleiner Zähler unter der Ellipse
        dd = ImageDraw.Draw(out, "RGBA")
        cnt = n if t <= sim.end_t else len(sim.trips[-1][3])
        dd.text((CX, CY - A - 34), f"{cnt:,}", font=_font(F_BOLD, 34), fill=(255, 255, 255, 200), anchor="mm")
        return out


# ------------------------------------------------------------------ Animierter Hintergrund im Vice-City-Sunset-Look

class ViceBackground:
    """Sonnenuntergang über dem Meer: Verlauf Lila -> Pink -> Orange, Retro-Sonne mit wandernden Streifen,
    Skyline mit Neon-Fenstern, glitzerndes Wasser und Palmen, die sich leicht im Wind wiegen."""

    HZ = 1290          # Horizont

    def __init__(self, seed=3):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:H1, 0:W1].astype(np.float32)
        stops = [(0, (22, 6, 52)), (420, (70, 16, 110)), (820, (170, 40, 135)), (1100, (245, 90, 110)),
                 (self.HZ, (255, 160, 80))]
        sky = np.zeros((H1, W1, 3), np.float32)
        for (y0, c0), (y1, c1) in zip(stops, stops[1:]):
            m = (yy >= y0) & (yy < y1)
            f = ((yy - y0) / (y1 - y0))[..., None]
            sky = np.where(m[..., None], np.array(c0) * (1 - f) + np.array(c1) * f, sky)
        sea = (yy >= self.HZ)
        fs = ((yy - self.HZ) / (H1 - self.HZ)).clip(0, 1)[..., None]
        sky = np.where(sea[..., None], np.array((60, 20, 90)) * (1 - fs) + np.array((12, 6, 34)) * fs, sky)
        self.sky = sky
        self.xx, self.yy = xx, yy
        # Wolkenbänder (weich, ziehen langsam)
        self.cloud = np.zeros((H1, W1 * 2), np.float32)
        for _ in range(9):
            cy, cx = rng.uniform(250, 950), rng.uniform(0, W1 * 2)
            w, h = rng.uniform(300, 700), rng.uniform(18, 45)
            Y, X = np.mgrid[0:H1, 0:W1 * 2].astype(np.float32)
            self.cloud += np.exp(-(((X - cx) / w) ** 2 + ((Y - cy) / h) ** 2)) * rng.uniform(0.25, 0.5)
        # Skyline (Silhouette + Fenster)
        sk = Image.new("RGBA", (W1, H1), (0, 0, 0, 0))
        d = ImageDraw.Draw(sk)
        x = -20
        self.windows = []
        while x < W1 + 20:
            w = rng.uniform(40, 95)
            h = rng.uniform(50, 230) * (0.6 if 300 < x < 780 else 1.0)
            d.rectangle([x, self.HZ - h, x + w, self.HZ], fill=(34, 10, 54, 255))
            if rng.random() < 0.3:
                d.rectangle([x + w * 0.4, self.HZ - h - 30, x + w * 0.5, self.HZ - h], fill=(34, 10, 54, 255))
            for wy in np.arange(self.HZ - h + 10, self.HZ - 8, 14):
                for wx in np.arange(x + 6, x + w - 6, 12):
                    if rng.random() < 0.28:
                        self.windows.append((wx, wy, rng.choice([0, 1, 2]), rng.uniform(0, 6.28)))
            x += w + rng.uniform(-6, 10)
        self.skyline = sk
        self.palms = [(40, 1600, 760, -0.10, 0.0), (140, 1700, 560, 0.06, 1.3), (1010, 1620, 720, 0.12, 2.1),
                      (930, 1730, 520, -0.05, 0.7)]   # (x, Fuß-y, Höhe, Neigung, Phase)

    def palm(self, d, x, base_y, h, lean, ph, t):
        sway = 0.035 * math.sin(t * 0.9 + ph) + 0.015 * math.sin(t * 2.1 + ph * 2)
        pts = []
        for i in range(13):
            u = i / 12
            px = x + (lean + sway) * h * u ** 1.6 + 18 * math.sin(u * 2.5)
            py = base_y - h * u
            pts.append((px, py))
        col = (16, 4, 26)
        for i in range(12):
            wdt = 26 - 12 * i / 12
            d.line([pts[i], pts[i + 1]], fill=col, width=int(wdt))
        tx, ty = pts[-1]
        for j in range(9):
            a = -math.pi / 2 + (j - 4) * 0.42 + sway * 4 + 0.08 * math.sin(t * 1.3 + j + ph)
            L = h * (0.42 + 0.06 * ((j * 7) % 3))
            leaf = []
            for q in range(10):
                u = q / 9
                ang = a + (1.3 * u * u) * (1 if math.cos(a) >= 0 else -1)
                lx = tx + math.cos(ang) * L * u
                ly = ty + math.sin(ang) * L * u + 0.35 * L * u * u
                leaf.append((lx, ly))
            top = [(lx, ly - 26 * math.sin(q / 9 * math.pi) * (1 - 0.4 * q / 9)) for q, (lx, ly) in enumerate(leaf)]
            bot = [(lx, ly + 16 * math.sin(q / 9 * math.pi) * (1 - 0.4 * q / 9)) for q, (lx, ly) in enumerate(leaf)]
            d.polygon(top + bot[::-1], fill=col)
        d.ellipse([tx - 16, ty - 12, tx + 16, ty + 14], fill=col)

    def frame(self, t):
        HZ = self.HZ
        img = self.sky.copy()
        # Wolken ziehen
        off = int(t * 12) % W1
        c = self.cloud[:, off:off + W1]
        img = img + c[..., None] * np.array((255, 150, 190)) * 0.35 * (self.yy < HZ)[..., None]
        # Retro-Sonne mit nach unten wandernden Streifen
        sx, sy, R = 540.0, HZ - 120 + 8 * math.sin(t * 0.35), 300.0
        dist = np.hypot(self.xx - sx, self.yy - sy)
        f = ((self.yy - (sy - R)) / (2 * R)).clip(0, 1)
        sun_c = np.array((255, 235, 120)) * (1 - f[..., None]) + np.array((255, 70, 150)) * f[..., None]
        stripe = np.ones_like(f)
        band = (self.yy - sy + t * 14) % 46
        lower = self.yy > sy - R * 0.15
        thick = 4 + 18 * ((self.yy - (sy - R * 0.15)) / (R * 1.15)).clip(0, 1)
        stripe[lower & (band < thick)] = 0
        inside = (dist < R) & (self.yy < HZ)
        glow = np.exp(-((dist - R).clip(0) / 120) ** 2) * 0.55 * (self.yy < HZ)
        img = img + glow[..., None] * np.array((255, 120, 140)) * 0.6
        img = np.where((inside & (stripe > 0))[..., None], sun_c, img)
        # Wasser: Spiegelung der Sonne als glitzernde Streifen
        sea = self.yy >= HZ
        ry = self.yy - HZ
        wob = np.sin(ry * 0.35 - t * 3.0 + np.sin(self.xx * 0.02 + t) * 2)
        spread = 140 + ry * 0.9
        refl = np.exp(-((self.xx - sx) / spread) ** 2) * (wob > 0.55) * np.exp(-ry / 420)
        img = img + (sea * refl)[..., None] * np.array((255, 170, 140)) * 0.8
        sparkle = (np.sin(self.xx * 0.9 + ry * 1.7 + t * 6) > 0.985) & sea
        img = img + sparkle[..., None] * 120
        im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")
        im.alpha_composite(self.skyline)
        d = ImageDraw.Draw(im, "RGBA")
        wcol = [(255, 80, 200), (80, 230, 255), (255, 200, 90)]
        for wx, wy, ci, ph in self.windows:
            if math.sin(t * 0.7 + ph) > -0.6:
                d.rectangle([wx, wy, wx + 5, wy + 6], fill=wcol[ci] + (200,))
        d.line([(0, HZ), (W1, HZ)], fill=(255, 150, 200, 160), width=2)
        for x, by, h, lean, ph in self.palms:
            self.palm(d, x, by, h, lean, ph, t)
        return np.asarray(im.convert("RGB"))


# ------------------------------------------------------------------ Klang

PENTA = [0, 2, 4, 7, 9]


def soft(f, dur=1.4, attack=0.012, decay=3.0):
    """Sehr weicher, runder Ton (fast reiner Sinus, sanfter Anschlag) – wie eine leise Kalimba."""
    t = np.arange(int(SR * dur)) / SR
    s = np.sin(2 * np.pi * f * t) + 0.18 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 8)
    return s * np.exp(-t * decay) * np.minimum(t / attack, 1)


def swell(freqs, dur=2.4):
    """Leise, weiche Akkord-Welle beim Treffen."""
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(t / 0.25, 1) * np.exp(-t * 1.6)
    return sum(np.sin(2 * np.pi * f * t) / (j + 1.5) for j, f in enumerate(freqs)) * env


def build_audio(sim, path):
    total = sim.total
    n = int(SR * total)
    out = np.zeros(n + SR * 3)
    last = -1.0
    scale = [0, 3, 5, 7, 10]            # Moll-Pentatonik: ruhig, nichts klingt schief

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    for t, kind, d in sim.events:
        if kind == "wall":
            if t - last < 0.11:
                continue
            last = t
            v = (d["th"] % (2 * math.pi)) / (2 * math.pi)
            j = int(v * 10)
            f = 293.66 * 2 ** ((scale[j % 5] + 12 * (j // 5)) / 12)
            put(t, soft(f, 1.1, 0.01, 4.5), 0.16)
        else:
            k = d["k"]
            root = 146.83 * 2 ** ([0, 3, 5, 7, 10, 12][k % 6] / 12)
            put(t, swell((root, root * 1.5, root * 2, root * 3)), 0.22)
            put(t, soft(root * 4, 1.8, 0.02, 2.2), 0.12)
    out = out[:n]
    tt = np.arange(n) / SR
    pad = sum(np.sin(2 * np.pi * f * tt + j) / (j + 2) for j, f in enumerate((73.42, 110.0, 146.83)))
    mix_ = out + reverb(out, 2.2) * 0.35 + pad * 0.02 * np.minimum(1, tt / 3)
    mix_ *= np.clip((total - tt) / 2.0, 0, 1) * np.minimum(1, tt / 0.5)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = mix_ / peak * 0.6
    st = np.stack([mix_, mix_], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def build_plop_audio(sim, path):
    """Nur ein angenehmes, weiches Plopp: leise an der Wand, etwas voller beim Treffen. Keine Melodie."""
    from .breaker import plop
    total = sim.total
    n = int(SR * total)
    out = np.zeros(n + SR * 2)
    rng = np.random.default_rng(2)
    last = -1.0

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    for t, kind, d in sim.events:
        if kind == "wall":
            if t - last < 0.09:
                continue
            last = t
            put(t, plop(float(rng.choice((560, 600, 640)))), 0.35)
        else:
            put(t, plop(380, big=True), 0.8)
    out = out[:n]
    tt = np.arange(n) / SR
    mix_ = out + reverb(out, 0.6) * 0.08
    mix_ *= np.clip((total - tt) / 1.5, 0, 1)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = mix_ / peak * 0.6
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
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "14M", "-bufsize", "28M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_R.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(sim, out_path, jobs=4, audio=None, **rkw):
    global _R
    _R = LoopRenderer(sim, **rkw)
    n = sim.n_frames
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
        (audio or build_audio)(sim, wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-19:TP=-3:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


VICE_KW = dict(ring_w=10, ball_scale=1.45)


def vice_sim(lead=0.9):
    """Etwas schneller, mehr Runden: Bälle fliegen ~20 % schneller."""
    return EllipseLoop(counts=(2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                               1024), t_first=2.5, t_min=1.85, shrink=0.97, lead=lead)


INTRO = ("1.000.000 Likes", 2.6)


if __name__ == "__main__" and "--plop" in __import__("sys").argv:
    import sys
    sim = vice_sim()
    print("Dauer", round(sim.total, 1))
    if "--preview" in sys.argv:
        R = LoopRenderer(sim, **VICE_KW)
        ims = [R.frame(int(t * FPS)) for t in (2.0, 20.0, 40.0)]
        out = Image.new("RGB", (360 * 3, 640))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), (i * 360, 0))
        out.save("output/preview/plop_check.png")
    else:
        render(sim, "output/e05_neon_loop_plop_vertical.mp4", audio=build_plop_audio, **VICE_KW)
        print("output/e05_neon_loop_plop_vertical.mp4")
    sys.exit(0)

if __name__ == "__main__" and "--vice" in __import__("sys").argv:
    import sys
    likes = "--likes" in sys.argv
    sim = vice_sim(lead=INTRO[1] + 0.9 if likes else 0.9)
    if likes:
        VICE_KW = dict(VICE_KW, intro=INTRO)
    print("Dauer", round(sim.total, 1), "Treffen", [round(m, 1) for m in sim.meets])
    bg = ViceBackground()
    if "--preview" in sys.argv:
        R = LoopRenderer(sim, bg=bg, **VICE_KW)
        ts = ([0.15, 1.0, 2.3, 2.8, 3.6, 5.0, 30.0, sim.total - 1] if likes else
              [0.5, 2.0, 6.0, 15.0, 30.0, 40.0, sim.end_t - 0.5, sim.total - 1])
        ims = [R.frame(int(t * FPS)) for t in ts]
        out = Image.new("RGB", (360 * 4, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 4) * 360, (i // 4) * 640))
        out.save("output/preview/vice_check.png")
        tiktok_ui_mock(ims[4], "vertical").save("output/preview/vice_ui.png")
    else:
        out = "output/e04_vice_loop_likes_intro_vertical.mp4" if likes else "output/e03_vice_circle_loop_vertical.mp4"
        render(sim, out, bg=bg, **VICE_KW)
        print(out)
    sys.exit(0)

if __name__ == "__main__":
    import sys
    sim = EllipseLoop()
    print("Dauer", round(sim.total, 1), "Treffen", [round(m, 1) for m in sim.meets])
    if "--preview" in sys.argv:
        R = LoopRenderer(sim)
        ts = [0.5, 1.6, 2.6, 3.65, 5.0, 12.0, 25.0, 40.0, sim.end_t - 0.6, sim.total - 1]
        ims = [R.frame(int(t * FPS)) for t in ts]
        out = Image.new("RGB", (360 * 5, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 5) * 360, (i // 5) * 640))
        out.save("output/preview/ellipse_check.png")
        tiktok_ui_mock(ims[7], "vertical").save("output/preview/ellipse_ui.png")
    else:
        render(sim, "output/e02_neon_circle_loop_vertical.mp4")
        print("output/e02_neon_circle_loop_vertical.mp4")
