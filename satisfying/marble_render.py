"""Hochwertiges Rendering der Murmelbahn: 60 fps, 2x Supersampling, weiche Schatten, glänzende Kugeln,
Abendhimmel mit Bokeh-Lichtern, sanfter Glow, Filmkorn, ruhige Kamerafahrt mit Rückzoom am Ende.
"""
import math
import os
import random
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tiktok.render import tiktok_ui_mock
from . import marble as M

W1, H1 = 1080, 1920          # Ausgabe
SS = 2                       # Supersampling der Leinwand
ZOOM = 1.3                   # Grund-Zoom der Kamera
ST = SS * ZOOM               # Auflösung der vorgerenderten Bahn (Pixel je Welteinheit)
W2, H2 = W1 * SS, H1 * SS
FOCUS_SX, FOCUS_SY = 490, 860  # Bildschirmpunkt, auf den die Kamera zielt (Mitte der Safe-Zone)
X0, Y0, X1, Y1 = -470.0, -420.0, 470.0, 3440.0  # Weltgrenzen der vorgerenderten Bahn

PEARL = (236, 230, 255)
PEARL_DARK = (150, 140, 200)
MINT = (130, 240, 215)
BAND = (90, 220, 255)
PEACH = (255, 184, 140)
LAVENDER = (190, 165, 255)
GOLD = (255, 205, 115)


def wp(x, y):
    """Welt -> Pixel der vorgerenderten Bahn."""
    return ((x - X0) * ST, (y - Y0) * ST)


def mixc(c, f):
    return tuple(int(min(255, v * f)) for v in c)


# ------------------------------------------------------------------ Sprites

def ball_sprite(color, r):
    """Glänzende Kugel (4x gerechnet, auf 2x verkleinert)."""
    S4 = 4
    R = r * S4
    n = int(R * 2 + 8)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    dx, dy = (xx - n / 2) / R, (yy - n / 2) / R
    d2 = dx * dx + dy * dy
    inside = d2 <= 1
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    L = np.array([-0.45, -0.6, 0.66])
    L /= np.linalg.norm(L)
    diff = np.clip(dx * L[0] + dy * L[1] + z * L[2], 0, 1)
    H = L + np.array([0, 0, 1.0])
    H /= np.linalg.norm(H)
    spec = np.clip(dx * H[0] + dy * H[1] + z * H[2], 0, 1) ** 45
    rim = (1 - z) ** 2.5
    base = np.array(color, np.float32)
    rgb = base[None, None, :] * (0.32 + 0.78 * diff[..., None]) + 255 * 0.95 * spec[..., None] \
        + np.array([180, 170, 255], np.float32) * 0.35 * rim[..., None]
    alpha = np.clip((1 - np.sqrt(d2)) * R * 1.2 + 0.5, 0, 1) * 255
    img = np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8)
    im = Image.fromarray(img, "RGBA")
    return im.resize((n * SS // S4, n * SS // S4), Image.LANCZOS)


def soft_sprite(radius, color, alpha, blur):
    n = int((radius + blur * 3) * 2)
    im = Image.new("RGBA", (n, n), color + (0,))
    d = ImageDraw.Draw(im)
    c = n / 2
    d.ellipse([c - radius, c - radius, c + radius, c + radius], fill=color + (alpha,))
    return im.filter(ImageFilter.GaussianBlur(blur))


# ------------------------------------------------------------------ Statische Bahn

def offset_line(path):
    off = M.R_BALL + 6
    return [wp(x + nx * off, y + ny * off) for (x, y), (nx, ny) in zip(path.pts, path.nrm)]


def draw_tube(d, pts, color, width, highlight=True):
    d.line(pts, fill=color, width=int(width * ST), joint="curve")
    r = width * ST / 2
    for p in (pts[0], pts[-1]):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)
    if highlight:
        hp = [(x - 1.2 * ST, y - 1.6 * ST) for x, y in pts]
        d.line(hp, fill=tuple(min(255, c + 45) for c in color[:3]) + (170,), width=int(width * 0.35 * ST), joint="curve")


def band_ranges():
    out = []
    for name in ("D", "F"):
        p = M.PATHS[name]
        for s0, s1, _ in p.boosts:
            out.append((p, s0, s1))
    return out


def build_static():
    w, h = int((X1 - X0) * ST), int((Y1 - Y0) * ST)
    shadow = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(shadow)
    track = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    td = ImageDraw.Draw(track)

    def sh_line(pts, width):
        sd.line([(x + 9 * ST, y + 14 * ST) for x, y in pts], fill=150, width=int(width * ST), joint="curve")

    paths = [p for k, p in M.PATHS.items() if k != "BOWL"]
    for p in paths:
        sh_line(offset_line(p), 15)
    # Leitwände neben dem Trichter und Auslauf
    guides = [((-420, -330), (-420, 330)), ((420, -330), (420, 330)), ((-24, 540), (-24, 600)), ((24, 540), (24, 590))]
    for a, b in guides:
        sh_line([wp(*a), wp(*b)], 10)
    # Mechanik-Sockel
    tr = M.TRAMP
    tramp_parts = [((tr["x"] - tr["half"], tr["y"]), (tr["x"] - tr["half"] + 10, tr["y"] + 70)),
                   ((tr["x"] + tr["half"], tr["y"]), (tr["x"] + tr["half"] - 10, tr["y"] + 70)),
                   ((tr["x"] - tr["half"] - 10, tr["y"] + 70), (tr["x"] + tr["half"] + 10, tr["y"] + 70))]
    for a, b in tramp_parts:
        sh_line([wp(*a), wp(*b)], 10)
    px, py = M.SEESAW["p"]
    seesaw_base = [wp(px, py + 4), wp(px - 26, py + 60), wp(px + 26, py + 60)]
    cx, cy = M.CATA["p"]
    cata_base = [wp(cx, cy), wp(cx - 40, cy + 110), wp(cx + 40, cy + 110)]
    for poly in (seesaw_base, cata_base):
        sd.polygon([(x + 9 * ST, y + 14 * ST) for x, y in poly], fill=150)
    bx, by = M.BOWL["c"]
    br = M.BOWL["r"] + 7
    sd.arc([*wp(bx - br + 9, by - br + 14), *wp(bx + br + 9, by + br + 14)], 0, 180, fill=150, width=int(18 * ST))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9 * ST))
    shadow_rgba = Image.new("RGBA", (w, h), (8, 4, 24, 0))
    shadow_rgba.putalpha(shadow.point(lambda v: int(v * 0.75)))
    # Stützen (dezente Pfosten unter der Bahn)
    for p in paths:
        if p.name.startswith("U") or p.name == "LOOP":
            continue
        for i in range(0, len(p.pts), 90):
            (x, y), (nx, ny) = p.pts[i], p.nrm[i]
            if ny < 0.5:
                continue
            x0, y0 = wp(x + nx * (M.R_BALL + 6), y + ny * (M.R_BALL + 6))
            td.line([x0, y0, x0, y0 + 70 * ST], fill=PEARL_DARK + (60,), width=int(3 * ST))
    # Bahnen
    for p in paths:
        draw_tube(td, offset_line(p), PEARL + (255,), 13)
    for a, b in guides:
        draw_tube(td, [wp(*a), wp(*b)], PEARL + (255,), 9)
    # Speedbänder: Bahnstück leuchtend türkis
    for p, s0, s1 in band_ranges():
        i0 = next(i for i, s in enumerate(p.s) if s >= s0)
        i1 = next(i for i, s in enumerate(p.s) if s >= s1)
        off = M.R_BALL + 6
        pts = [wp(x + nx * off, y + ny * off) for (x, y), (nx, ny) in zip(p.pts[i0:i1], p.nrm[i0:i1])]
        draw_tube(td, pts, BAND + (255,), 15)
    # Trampolin-Gestell
    for a, b in tramp_parts:
        draw_tube(td, [wp(*a), wp(*b)], PEARL + (255,), 9)
    # Wippe + Katapult Sockel
    td.polygon(seesaw_base, fill=PEARL + (255,))
    td.polygon(cata_base, fill=PEARL + (255,))
    # Ziel-Schale
    td.arc([*wp(bx - br, by - br), *wp(bx + br, by + br)], 0, 180, fill=GOLD + (255,), width=int(16 * ST))
    td.arc([*wp(bx - br + 3, by - br + 3), *wp(bx + br - 3, by + br - 3)], 20, 160, fill=(255, 240, 200, 160), width=int(4 * ST))
    for sx in (-1, 1):
        X, Y = wp(bx + sx * br, by)
        rr = 11 * ST
        td.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=GOLD + (255,))
    return Image.alpha_composite(shadow_rgba, track)


# ------------------------------------------------------------------ Kamera

def camera_track(sim):
    n = len(sim.frames)
    fy = np.zeros(n)
    first_escape = min(t for t, k, d in sim.events if k == "escape")
    fire = next((t for t, k, d in sim.events if k == "fire"), None)
    for i, fr in enumerate(sim.frames):
        t = fr["t"]
        trav = [b[1] for b in fr["balls"] if b[2] not in ("circle", "cup", "cup_in", "bowl")]
        if t < first_escape + 0.5:
            y = 60.0
        elif fire is not None and t >= fire - 0.2:
            ys = [b[1] for b in fr["balls"]]
            y = sum(ys) / len(ys) - 60
        elif trav:
            y = sum(trav) / len(trav)
        else:
            y = fy[i - 1] if i else 60.0
        fy[i] = y
    # nullphasige Glättung (vor- und rückwärts)
    a = 1 - math.exp(-1 / (M.FPS * 0.75))
    for rng in (range(1, n), range(n - 2, -1, -1)):
        for i in rng:
            j = i - 1 if rng.step == 1 else i + 1
            fy[i] = fy[j] + a * (fy[i] - fy[j])
    z = np.full(n, ZOOM)
    fx = np.zeros(n)
    for i, fr in enumerate(sim.frames):
        trav = [b[0] for b in fr["balls"] if b[2] not in ("circle", "cup", "cup_in", "bowl")]
        fx[i] = (sum(trav) / len(trav)) * 0.55 if trav and fr["t"] > first_escape + 0.5 else (fx[i - 1] if i else 0.0)
    for rng in (range(1, n), range(n - 2, -1, -1)):
        for i in rng:
            j = i - 1 if rng.step == 1 else i + 1
            fx[i] = fx[j] + a * 0.6 * (fx[i] - fx[j])
    fx = np.clip(fx, -120, 120)
    if sim.finish_t is not None:  # Rückzoom auf die ganze Bahn
        t0 = sim.finish_t + 0.6
        zf = H1 / (3480.0 + 300)
        mid = (-380 + 3340) / 2
        for i, fr in enumerate(sim.frames):
            p = M.ease((fr["t"] - t0) / 4.2)
            if p > 0:
                z[i] = ZOOM + (zf - ZOOM) * p
                fy[i] = fy[i] + (mid - fy[i]) * p
                fx[i] = fx[i] * (1 - p)
    return fx, fy, z


# ------------------------------------------------------------------ Hintergrund

class Background:
    def __init__(self, seed=5):
        rng = random.Random(seed)
        yy, xx = np.mgrid[0:H1, 0:W1].astype(np.float32)
        self.yn = (yy / H1)[..., None]
        d = np.sqrt(((xx - FOCUS_SX) / (W1 * 0.8)) ** 2 + ((yy - FOCUS_SY) / (H1 * 0.55)) ** 2)
        self.glow = np.clip(1 - d, 0, 1)[..., None] ** 2
        vd = np.sqrt(((xx - W1 / 2) / (W1 * 0.75)) ** 2 + ((yy - H1 / 2) / (H1 * 0.75)) ** 2)
        self.vignette = np.clip(1.15 - vd ** 2 * 0.55, 0.45, 1)[..., None]
        cols = [(255, 190, 150), (150, 200, 255), (205, 165, 255), (255, 225, 170)]
        self.bokeh = []
        for i in range(34):
            r = rng.uniform(14, 70)
            spr = soft_sprite(r, cols[i % 4], int(rng.uniform(26, 70)), r * 0.35)
            self.bokeh.append((spr, rng.uniform(-100, W1 + 100), rng.uniform(-400, 4000), rng.uniform(0.15, 0.45),
                               rng.uniform(0, 6.28), rng.uniform(6, 18)))
        g = np.random.default_rng(seed)
        self.grain = [g.normal(0, 3.2, (H1, W1, 1)).astype(np.float32) for _ in range(6)]

    def render(self, cam_y, z, t):
        k = min(1.0, max(0.0, (cam_y + 400) / 3800))
        top = np.array([16, 18, 48], np.float32) * (1 - k) + np.array([28, 18, 52], np.float32) * k
        bot = np.array([48, 28, 72], np.float32) * (1 - k) + np.array([70, 34, 66], np.float32) * k
        warm = np.array([60, 36, 70], np.float32) * (0.8 + 0.2 * math.sin(t * 0.3))
        arr = top + (bot - top) * self.yn + warm * self.glow
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
        for spr, x, y, par, ph, sp in self.bokeh:
            sy = (y - cam_y * par * z) % 2600 - 340
            sx = x + math.sin(t * 0.12 + ph) * sp
            img.alpha_composite(spr, (int(sx - spr.width / 2), int(sy - spr.height / 2)))
        return img


# ------------------------------------------------------------------ Frame-Renderer

class Renderer:
    def __init__(self, sim):
        self.sim = sim
        self.static = build_static()
        self.fx, self.fy, self.z = camera_track(sim)
        self.bg = Background()
        self.ball_spr = {}
        self.shadow_spr = soft_sprite(M.R_BALL * SS, (6, 2, 20), 150, 5 * SS)
        self.glow_spr = {c: soft_sprite(M.R_BALL * SS * 1.5, c, 60, 9 * SS) for c in M.BALL_COLORS}
        self.goal_glow = soft_sprite(160 * SS, GOLD, 70, 50 * SS)
        self.events = sim.events
        self.tramp_ev = list(sim.tramp_ev)
        self.bounces = [(t, d) for t, k, d in sim.events if k in ("bounce", "land")]
        self.goals = [(t, d) for t, k, d in sim.events if k == "goal"]
        self.fire_t = sim.fire_t

    def ball_img(self, color, z):
        key = (color, round(z, 3))
        if key not in self.ball_spr:
            if len(self.ball_spr) > 400:
                self.ball_spr.clear()
            self.ball_spr[key] = ball_sprite(color, M.R_BALL * z)
        return self.ball_spr[key]

    def frame(self, i):
        sim = self.sim
        fr = sim.frames[i]
        t = fr["t"]
        fx, fy, z = self.fx[i], self.fy[i], self.z[i]
        k = SS * z  # Pixel pro Welteinheit auf der 2x-Leinwand

        def P(x, y):
            return (FOCUS_SX * SS + (x - fx) * k, FOCUS_SY * SS + (y - fy) * k)

        # Statische Bahn ausschneiden (crop füllt außerhalb transparent auf)
        f = ST / k  # Bahn-Pixel je Leinwand-Pixel
        left = (fx - FOCUS_SX * SS / k - X0) * ST
        top = (fy - FOCUS_SY * SS / k - Y0) * ST
        if abs(f - 1) < 1e-6:
            canvas = self.static.crop((int(round(left)), int(round(top)), int(round(left)) + W2, int(round(top)) + H2))
        else:
            box = (int(math.floor(left)), int(math.floor(top)), int(math.ceil(left + W2 * f)), int(math.ceil(top + H2 * f)))
            canvas = self.static.crop(box).resize((W2, H2), Image.BILINEAR, reducing_gap=2.0)
        d = ImageDraw.Draw(canvas)
        self.draw_dynamic(canvas, d, P, k, t, fr)
        small = canvas.reduce(SS)
        # Hintergrund + Glow + Vignette + Korn
        bg = self.bg.render(fy, z, t)
        bg.alpha_composite(small)
        rgb = bg.convert("RGB")
        g = rgb.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5))
        arr = np.asarray(rgb, np.float32)
        glow = np.asarray(g.resize((W1, H1), Image.BILINEAR), np.float32)
        arr = arr + np.clip(glow - 40, 0, 255) * 0.45
        arr = arr * self.bg.vignette + self.bg.grain[i % len(self.bg.grain)]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    # ---- dynamische Teile

    def draw_dynamic(self, canvas, d, P, k, t, fr):
        sim = self.sim
        # Start-Kreis mit wandernder Lücke
        gc = sim.gap_center(t)
        R = (M.R0 + 6) * k
        cx, cy = P(0, 0)
        a0, a1 = math.degrees(gc + M.GAP_HALF), math.degrees(gc - M.GAP_HALF) + 360
        d.arc([cx - R + 9 * k, cy - R + 14 * k, cx + R + 9 * k, cy + R + 14 * k], a0, a1, fill=(6, 2, 20, 70), width=int(16 * k))
        d.arc([cx - R, cy - R, cx + R, cy + R], a0, a1, fill=MINT + (255,), width=int(12 * k))
        d.arc([cx - R + 3 * k, cy - R + 3 * k, cx + R - 3 * k, cy + R - 3 * k], a0 + 4, a1 - 4, fill=(220, 255, 245, 140), width=int(3 * k))
        for a in (gc + M.GAP_HALF, gc - M.GAP_HALF):
            X, Y = P(math.cos(a) * (M.R0 + 6), math.sin(a) * (M.R0 + 6))
            r = 8 * k
            d.ellipse([X - r, Y - r, X + r, Y + r], fill=(235, 255, 250, 255))
        # Speedband-Pfeile
        for p, s0, s1 in band_ranges():
            active = any(b[2] == "rail" for b in fr["balls"])
            off = M.R_BALL + 6
            sp = 0.0
            step = 34.0
            ph = (t * 220.0) % step
            s = s0 + ph
            while s < s1:
                (x, y), (tx, ty) = p.at(s)
                n = p.nrm[min(len(p.nrm) - 1, int(s / p.length * (len(p.nrm) - 1)))]
                bx, by = x + n[0] * off, y + n[1] * off
                f = 0.55 + 0.45 * math.sin((s - s0) / (s1 - s0) * math.pi)
                tip = P(bx + tx * 7, by + ty * 7)
                l1 = P(bx - tx * 4 + n[0] * 5, by - ty * 4 + n[1] * 5)
                l2 = P(bx - tx * 4 - n[0] * 5, by - ty * 4 - n[1] * 5)
                d.line([l1, tip, l2], fill=(240, 255, 255, int(230 * f)), width=max(2, int(2.5 * k)), joint="curve")
                s += step
        # Trampolin-Membran
        tr = M.TRAMP
        defl = 0.0
        for tl in self.tramp_ev:
            e = t - tl
            if 0 <= e < 0.16:
                defl += tr["depth"] * math.sin(math.pi * e / 0.16)
            elif 0.16 <= e < 1.5:
                e2 = e - 0.16
                defl += -tr["depth"] * 0.35 * math.exp(-5 * e2) * math.sin(22 * e2)
        pts = []
        for j in range(21):
            u = -1 + 2 * j / 20
            x = tr["x"] + u * tr["half"]
            y = tr["y"] + defl * math.cos(u * math.pi / 2) ** 1.5
            pts.append(P(x, y))
        d.line(pts, fill=MINT + (255,), width=int(7 * k), joint="curve")
        # Wippe
        phi = fr["seesaw"]
        px, py = M.SEESAW["p"]
        hl, th = M.SEESAW["half"] + 10, M.SEESAW["th"]
        dx, dy = math.cos(phi), math.sin(phi)
        nx, ny = dy, -dx
        poly = [P(px - hl * dx + th * nx, py - hl * dy + th * ny), P(px + hl * dx + th * nx, py + hl * dy + th * ny),
                P(px + hl * dx - th * nx, py + hl * dy - th * ny), P(px - hl * dx - th * nx, py - hl * dy - th * ny)]
        sh = [(x + 9 * k, y + 14 * k) for x, y in poly]
        d.polygon(sh, fill=(6, 2, 20, 80))
        d.polygon(poly, fill=PEACH + (255,))
        d.line([poly[0], poly[1]], fill=(255, 225, 200, 255), width=max(2, int(2 * k)))
        X, Y = P(px, py)
        r = 7 * k
        d.ellipse([X - r, Y - r, X + r, Y + r], fill=PEARL + (255,))
        # Katapult-Arm mit Becher
        a = fr["cata"]
        cpx, cpy = M.CATA["p"]
        dx, dy = math.cos(a), math.sin(a)
        nx, ny = dy, -dx

        def A(rho, lift):
            return P(cpx + rho * dx + lift * nx, cpy + rho * dy + lift * ny)
        arm = [A(-18, 7), A(182, 7), A(182, -7), A(-18, -7)]
        d.polygon([(x + 9 * k, y + 14 * k) for x, y in arm], fill=(6, 2, 20, 80))
        d.polygon(arm, fill=LAVENDER + (255,))
        cup = [A(96, 5), A(96, 54), A(104, 54), A(108, 12), A(170, 12), A(174, 44), A(182, 44), A(182, 5)]
        d.polygon([A(104, 8), A(104, 54), A(174, 44), A(174, 8)], fill=LAVENDER + (70,))  # Becher-Innenseite
        d.polygon(cup, fill=LAVENDER + (255,))
        d.line([A(-18, 7), A(182, 7)], fill=(225, 210, 255, 255), width=max(2, int(2 * k)))
        X, Y = P(cpx, cpy)
        r = 11 * k
        d.ellipse([X - r, Y - r, X + r, Y + r], fill=PEARL + (255,))
        # Ziel-Glühen
        n_in = sum(1 for b in fr["balls"] if b[2] == "bowl")
        if n_in:
            bx, by = M.BOWL["c"]
            gz = self.goal_glow
            f = min(1.0, 0.35 * n_in + 0.1 * math.sin(t * 2))
            zz = k / SS
            spr = gz.resize((max(1, int(gz.width * zz)), max(1, int(gz.height * zz))))
            a_ = spr.getchannel("A").point(lambda v: int(v * f))
            spr.putalpha(a_)
            X, Y = P(bx, by + 40)
            canvas.alpha_composite(spr, (int(X - spr.width / 2), int(Y - spr.height / 2)))
        # sanfte Aufprall-Wellen
        for t0, ev in self.bounces:
            e = t - t0
            if 0 <= e < 0.45 and "x" in ev:
                p_ = e / 0.45
                X, Y = P(ev["x"], ev["y"])
                r = (8 + 40 * (1 - (1 - p_) ** 3)) * k
                d.ellipse([X - r, Y - r, X + r, Y + r], outline=(255, 255, 255, int(120 * (1 - p_) * ev.get("strength", 0.5))),
                          width=max(1, int(2 * k)))
        # Ziel: aufsteigende goldene Lichtpunkte
        for t0, ev in self.goals:
            e = t - t0
            if 0 <= e < 2.2:
                rng = random.Random(int(t0 * 1000))
                for j in range(9):
                    vx, vy = rng.uniform(-40, 40), rng.uniform(-120, -60)
                    X, Y = P(ev["x"] + vx * e, ev["y"] + vy * e)
                    r = (2.5 + 2 * rng.random()) * k
                    al = int(200 * (1 - e / 2.2))
                    d.ellipse([X - r, Y - r, X + r, Y + r], fill=(255, 235, 180, al))
        # Bälle: Schatten, Glow, Kugel
        spr_sh = self.shadow_spr
        for b, (x, y, st, v) in zip(sim.balls, fr["balls"]):
            X, Y = P(x, y)
            s2 = spr_sh if abs(k - SS) < 1e-6 else spr_sh.resize((max(1, int(spr_sh.width * k / SS)),) * 2)
            canvas.alpha_composite(s2, (int(X + 7 * k - s2.width / 2), int(Y + 11 * k - s2.height / 2)))
        for b, (x, y, st, v) in zip(sim.balls, fr["balls"]):
            X, Y = P(x, y)
            gl = self.glow_spr[b.color]
            if abs(k - SS) > 1e-6:
                gl = gl.resize((max(1, int(gl.width * k / SS)),) * 2)
            canvas.alpha_composite(gl, (int(X - gl.width / 2), int(Y - gl.height / 2)))
            spr = self.ball_img(b.color, k / SS)
            canvas.alpha_composite(spr, (int(round(X - spr.width / 2)), int(round(Y - spr.height / 2))))


# ------------------------------------------------------------------ Klang

SR = 44100
F_PENTA = [349.23 * 2 ** (s / 12) for s in (-12, -10, -8, -5, -3, 0, 2, 4, 7, 9, 12)]  # F-Dur pentatonisch
_cache = {}


def felt_piano(f, dur=2.5):
    key = ("fp", round(f, 2))
    if key not in _cache:
        n = int(SR * dur)
        t = np.arange(n) / SR
        s = sum(np.sin(2 * np.pi * f * h * t * (1 + 0.0004 * h)) / h ** 1.7 * np.exp(-t * (1.4 + h * 0.9)) for h in range(1, 7))
        s *= np.minimum(t / 0.006, 1)
        rng = np.random.default_rng(int(f))
        thump = np.convolve(rng.standard_normal(n), np.ones(40) / 40, mode="same") * np.exp(-t * 90) * 0.4
        _cache[key] = (s + thump) * 0.35
    return _cache[key]


def harp(f):
    key = ("harp", round(f, 2))
    if key not in _cache:
        n = int(SR * 2.5)
        N = max(2, int(SR / f))
        rng = np.random.default_rng(int(f) + 7)
        y = np.zeros(n)
        y[:N] = rng.uniform(-1, 1, N)
        for i in range(N, n):
            y[i] = 0.4992 * (y[i - N] + y[i - N - 1])
        t = np.arange(n) / SR
        _cache[key] = (y * 0.6 + np.sin(2 * np.pi * f * t) * np.exp(-t * 2.5) * 0.4) * 0.4
    return _cache[key]


def boing():
    n = int(SR * 0.6)
    t = np.arange(n) / SR
    f = 150 + 120 * (1 - np.exp(-t * 9)) + 18 * np.sin(2 * np.pi * 11 * t) * np.exp(-t * 5)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5) * 0.55


def tock():
    n = int(SR * 0.18)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 820 * t) * 0.6 + np.sin(2 * np.pi * 410 * t)) * np.exp(-t * 45) * 0.45


def swoosh(dur=0.9, rng=None):
    rng = rng or np.random.default_rng(3)
    n = int(SR * dur)
    t = np.linspace(0, 1, n)
    noise = rng.standard_normal(n)
    lp = np.convolve(noise, np.ones(14) / 14, mode="same")
    shimmer = np.sin(2 * np.pi * np.cumsum(600 + 900 * t) / SR) * 0.15
    return (lp * 0.8 + shimmer) * np.sin(np.pi * t) ** 2 * 0.45


def thump():
    n = int(SR * 0.7)
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * (70 + 40 * np.exp(-t * 20)) * t) * np.exp(-t * 6) * 0.8


def pad(total):
    n = int(SR * total)
    t = np.arange(n) / SR
    chords = [(174.61, 220.0, 261.63, 329.63, 392.0), (146.83, 174.61, 220.0, 261.63, 329.63),
              (116.54, 146.83, 174.61, 220.0, 261.63), (130.81, 164.81, 196.0, 293.66, 329.63)]
    out = np.zeros(n)
    seg = 8.0
    for ci in range(int(total / seg) + 2):
        ch = chords[ci % 4]
        t0 = ci * seg
        env = np.clip((t - t0 + 2.0) / 2.0, 0, 1) * np.clip((t0 + seg + 2.0 - t) / 2.0, 0, 1)
        if not env.any():
            continue
        for k, f in enumerate(ch):
            out += env * (np.sin(2 * np.pi * f * t + k) + 0.25 * np.sin(2 * np.pi * f * 2.003 * t)) / (k + 2)
    out *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.07 * t)
    return out * 0.05 * np.minimum(1, t / 2.5)


def reverb(x, seconds=2.4, seed=1):
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t * 3.2)
    ir = np.convolve(ir, np.ones(6) / 6, mode="same")
    ir /= np.sqrt(np.sum(ir ** 2))
    L = len(x) + n
    nfft = 1 << (L - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft), nfft)[:len(x)]
    return y


def build_audio(sim, path):
    total = len(sim.frames) / M.FPS
    n = int(SR * total)
    dry = np.zeros(n + SR * 3)
    rng = random.Random(7)
    nrng = np.random.default_rng(11)
    pos = 5

    def put(t, s, vol):
        i = int(t * SR)
        if i >= len(dry):
            return
        j = min(len(dry), i + len(s))
        dry[i:j] += s[: j - i] * vol

    cup_notes = [523.25, 659.25, 783.99]
    for t, kind, dd in sim.events:
        if kind == "bounce":
            pos = min(len(F_PENTA) - 1, max(2, pos + rng.choice((-2, -1, 1, 2))))
            put(t, felt_piano(F_PENTA[pos]), 0.35 + 0.45 * dd["strength"])
        elif kind == "escape":
            put(t, felt_piano(F_PENTA[8]), 0.5)
        elif kind == "land":
            put(t, felt_piano(F_PENTA[3]), 0.45)
            put(t, tock(), 0.25)
        elif kind == "boost":
            put(t, swoosh(0.9, nrng), 0.55)
        elif kind == "jump":
            put(t, swoosh(0.5, nrng), 0.3)
        elif kind == "tramp":
            put(t, boing(), 0.7)
        elif kind == "seesaw":
            put(t, tock(), 0.7)
            put(t, felt_piano(F_PENTA[4]), 0.3)
        elif kind == "loop":
            put(t, swoosh(1.0, nrng), 0.5)
        elif kind == "cup":
            put(t + 0.35, felt_piano(cup_notes[dd["k"]]), 0.55)
            put(t + 0.35, tock(), 0.35)
        elif kind == "wind":
            for j in range(6):
                put(dd["at"] + j * 0.12, tock(), 0.18)
        elif kind == "fire":
            put(t, thump(), 0.6)
            put(t, swoosh(0.8, nrng), 0.5)
        elif kind == "goal":
            base = [349.23, 440.0, 523.25, 659.25, 698.46]
            for j, f in enumerate(base):
                put(t + j * 0.07, harp(f * (1 + 0.5 * (dd["n"] == 3))), 0.45)
        elif kind == "settled":
            for j, f in enumerate((349.23, 440.0, 523.25, 698.46, 880.0)):
                put(t + j * 0.18, harp(f), 0.4)
    # Rollgeräusch: gefiltertes Rauschen, Lautstärke nach Rollgeschwindigkeit
    env = np.zeros(len(sim.frames))
    for i, fr in enumerate(sim.frames):
        env[i] = sum(min(1.0, abs(b[3]) / 450) ** 0.8 for b in fr["balls"] if b[2] in ("rail", "bowl"))
    env = np.interp(np.arange(n) / SR, np.arange(len(env)) / M.FPS, env)
    brown = np.cumsum(nrng.standard_normal(n))
    brown -= np.convolve(brown, np.ones(400) / 400, mode="same")
    brown /= (np.max(np.abs(brown)) or 1)
    rolling = brown * env * 0.18
    dry = dry[:n] + rolling
    wet = reverb(dry)
    mix = dry * 0.8 + wet * 0.45 + pad(total)
    fade = np.clip((total - np.arange(n) / SR) / 1.5, 0, 1)
    mix *= fade
    peak = np.max(np.abs(mix)) or 1
    mix = np.tanh(mix / peak * 1.1) * 0.9
    st = np.stack([mix, mix], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


# ------------------------------------------------------------------ Ausgabe

_R = None


def _chunk(args):
    i0, i1, out = args
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W1}x{H1}", "-r", str(M.FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-maxrate", "14M", "-bufsize", "28M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_R.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(sim, spec, fmt, out_path, preview_dir=None, preview_times=(), jobs=4):
    global _R
    _R = Renderer(sim)
    n = len(sim.frames)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fr = _R.frame(min(int(pt * M.FPS), n - 1))
            base = os.path.join(preview_dir, f"{spec['id']}_{fmt}_{pt:05.1f}")
            fr.save(base + ".png")
            tiktok_ui_mock(fr, fmt).save(base + "_ui.png")
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
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path], check=True)
