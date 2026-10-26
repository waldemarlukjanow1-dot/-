"""Hypnose-Renderer: kein Text, nur das System. Starke Aufprall-Effekte und ein Finale mit Knall.

Effekte pro Aufprall: Schockwelle, Funkenstreifen, aufleuchtende Wand, Ball-Blitz.
Ringbruch: weißer Blitz, Splitter mit Schweif, große Schockwelle, Kamera-Ruck.
"""
import bisect
import math
import os
import random
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tiktok.render import ease, tiktok_ui_mock
from . import sound
from .sim import ARENA, FPS, hue_rgb

# Arena füllt die Safe-Zone (rechts frei für TikTok-Buttons, unten für die Beschreibung).
LAYOUTS = {
    "vertical": dict(W=1080, H=1920, arena=(490, 825, 420)),
    "landscape": dict(W=1920, H=1080, arena=(870, 540, 450)),
}

HIT_LIFE = 0.42
SHARD_LIFE = 1.3


def mix(col, f, white=0.0):
    return tuple(int(min(255, (c + (255 - c) * white) * f)) for c in col)


def ease_out(p):
    p = min(max(p, 0.0), 1.0)
    return 1 - (1 - p) ** 3


class HypnoScene:
    def __init__(self, sim, kind, fmt):
        self.sim, self.kind = sim, kind
        L = LAYOUTS[fmt]
        self.W, self.H = L["W"], L["H"]
        self.cx, self.cy, ar = L["arena"]
        self.s = ar / ARENA
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        d = np.sqrt((xx - self.cx) ** 2 + (yy - self.cy) ** 2) / (ar * 2.0)
        self.grad = (np.clip(1 - d, 0, 1) ** 1.6)[..., None]
        self.hit_t = [h[0] for h in sim.hits]
        self.breaks = [(t, d) for t, k, d in sim.events if k == "break"]

    def P(self, x, y):
        return self.cx + x * self.s, self.cy + y * self.s

    # ------------------------------------------------------------ Effekte

    def hit_effects(self, d, t):
        lo = bisect.bisect_left(self.hit_t, t - HIT_LIFE)
        hi = bisect.bisect_right(self.hit_t, t)
        cap = 28 if self.kind == "multiply" else 70
        idx = range(max(lo, hi - cap), hi)  # bei sehr vielen Treffern nur die neuesten
        for i in idx:
            t0, x, y, col = self.sim.hits[i]
            if self.sim.finish_t is not None and t0 > self.sim.finish_t:
                break  # nach dem Finale keine Aufprall-Effekte mehr
            age = t - t0
            p = age / HIT_LIFE
            fade = 1 - p
            X, Y = self.P(x, y)
            R0 = math.hypot(x, y)
            # aufleuchtende Wand
            if age < 0.25 and R0 > 1:
                a = math.degrees(math.atan2(y, x))
                rr = R0 * self.s
                d.arc([self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr], a - 9, a + 9,
                      fill=mix(col, 1 - age / 0.25, 0.6), width=max(6, int(12 * self.s)))
            # Schockwelle
            r = (6 + (40 if self.kind == "multiply" else 75) * ease_out(p)) * self.s
            d.ellipse([X - r, Y - r, X + r, Y + r], outline=mix(col, fade, 0.3), width=max(1, int(6 * fade)))
            # Funken
            rng = random.Random(i * 7919)
            nx, ny = (-x / R0, -y / R0) if R0 > 1 else (0, -1)
            base = math.atan2(ny, nx)
            for _ in range(9):
                a = base + rng.uniform(-1.3, 1.3)
                sp = rng.uniform(300, 850)
                def pos(tt):
                    return self.P(x + math.cos(a) * sp * tt, y + math.sin(a) * sp * tt + 600 * tt * tt)
                if age > 0.32:
                    continue
                x1, y1 = pos(age)
                x0, y0 = pos(max(0.0, age - 0.035))
                d.line([x0, y0, x1, y1], fill=mix(col, 1 - age / 0.32, 0.5), width=3)

    def ring_break_effects(self, d, t):
        sim = self.sim
        for t0, ev in self.breaks:
            age = t - t0
            if age < 0 or age > SHARD_LIFE:
                continue
            ring = sim.rings[ev["ring"]]
            R, col = ring["R"], ring["color"]
            rr = R * self.s
            if age < 0.12:  # weißer Blitz des ganzen Rings
                d.ellipse([self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr], outline=mix(col, 1, 1 - age / 0.12),
                          width=max(6, int(12 * self.s)))
            p = age / 0.6
            if p < 1:  # große Schockwelle
                r2 = (R + 260 * ease_out(p)) * self.s
                d.ellipse([self.cx - r2, self.cy - r2, self.cx + r2, self.cy + r2], outline=mix(col, 1 - p, 0.4),
                          width=max(1, int(9 * (1 - p))))
            rng = random.Random(ev["ring"] * 104729)
            fade = 1 - age / SHARD_LIFE
            n = 56
            for k in range(n):
                a = ev["gap"] + ring["half"] + (2 * math.pi - 2 * ring["half"]) * (k + 0.5) / n
                va = a + rng.uniform(-0.35, 0.35)
                sp = rng.uniform(250, 750)
                spin = rng.uniform(-10, 10)
                L = 2 * math.pi * R / n * 0.5
                def pos(tt):
                    return (math.cos(a) * R + math.cos(va) * sp * tt, math.sin(a) * R + math.sin(va) * sp * tt + 450 * tt * tt)
                px, py = pos(age)
                qx, qy = pos(max(0.0, age - 0.08))
                X, Y = self.P(px, py)
                Q = self.P(qx, qy)
                d.line([Q[0], Q[1], X, Y], fill=mix(col, fade * 0.45), width=max(2, int(4 * self.s)))  # Schweif
                rot = a + math.pi / 2 + spin * age
                l = L * self.s
                d.line([X - math.cos(rot) * l, Y - math.sin(rot) * l, X + math.cos(rot) * l, Y + math.sin(rot) * l],
                       fill=mix(col, fade, 0.35 * fade), width=max(3, int(6 * self.s)))
            if age < 0.5:  # Funkenregen am Ausbruchspunkt
                bx, by = math.cos(ev["gap"]) * R, math.sin(ev["gap"]) * R
                rng2 = random.Random(ev["ring"])
                for _ in range(26):
                    a = rng2.uniform(0, 2 * math.pi)
                    sp = rng2.uniform(300, 1000)
                    X1, Y1 = self.P(bx + math.cos(a) * sp * age, by + math.sin(a) * sp * age)
                    X0, Y0 = self.P(bx + math.cos(a) * sp * max(0, age - 0.04), by + math.sin(a) * sp * max(0, age - 0.04))
                    d.line([X0, Y0, X1, Y1], fill=mix((255, 255, 255), 1 - age / 0.5), width=3)

    def comet(self, d, positions, r, col, flash=0.0):
        """Ball mit sich verjüngendem Leuchtschweif. positions: älteste -> aktuellste."""
        n = len(positions)
        for k in range(1, n):
            f = k / n
            (x0, y0), (x1, y1) = positions[k - 1], positions[k]
            X0, Y0 = self.P(x0, y0)
            X1, Y1 = self.P(x1, y1)
            d.line([X0, Y0, X1, Y1], fill=mix(col, f * 0.85), width=max(2, int(r * 2 * self.s * f)))
        X, Y = self.P(*positions[-1])
        R = r * self.s * (1 + 0.35 * flash)
        d.ellipse([X - R, Y - R, X + R, Y + R], fill=mix(col, 1, 0.55 + 0.45 * flash))
        R2 = R * 0.55
        d.ellipse([X - R2, Y - R2, X + R2, Y + R2], fill=(255, 255, 255))

    def finale(self, d, t, origin, cols):
        fin = self.sim.finish_t
        if fin is None or t < fin:
            return
        age = t - fin
        ox, oy = origin
        for k in range(4):  # mehrere Schockwellen nacheinander
            p = (age - k * 0.12) / 0.9
            if 0 < p < 1:
                r = (40 + 900 * ease_out(p)) * self.s
                X, Y = self.P(ox, oy)
                d.ellipse([X - r, Y - r, X + r, Y + r], outline=mix(cols[k % len(cols)], 1 - p, 0.3),
                          width=max(2, int(16 * (1 - p))))
        rng = random.Random(99)
        if age < 1.6:
            for i in range(220):
                a = rng.uniform(0, 2 * math.pi)
                sp = rng.uniform(200, 1400)
                tt = age
                X1, Y1 = self.P(ox + math.cos(a) * sp * tt, oy + math.sin(a) * sp * tt + 300 * tt * tt)
                X0, Y0 = self.P(ox + math.cos(a) * sp * max(0, tt - 0.05), oy + math.sin(a) * sp * max(0, tt - 0.05))
                d.line([X0, Y0, X1, Y1], fill=mix(cols[i % len(cols)], 1 - age / 1.6, 0.3), width=4)

    # ------------------------------------------------------------ Systeme

    def draw_ring(self, d, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        fin = sim.finish_t
        for k in range(fr["k"], len(sim.rings)):
            ring = sim.rings[k]
            R = ring["R"] * self.s
            g = sim.gap(ring, t)
            a0, a1 = math.degrees(g + ring["half"]), math.degrees(g - ring["half"]) + 360
            inner = k == fr["k"]
            pulse = 0.85 + 0.15 * math.sin(t * 6 - k * 0.5)
            col = mix(ring["color"], 1.0 if inner else 0.7 * pulse)
            d.arc([self.cx - R, self.cy - R, self.cx + R, self.cy + R], a0, a1, fill=col,
                  width=max(4, int((8 if inner else 5) * self.s)))
        self.ring_break_effects(d, t)
        self.hit_effects(d, t)
        if fin is None or t < fin + 0.05:
            pos = [sim.frames[j]["balls"][0][:2] for j in range(max(0, fi - 9), fi + 1)]
            col = fr["balls"][0][4]
            last = self.hit_t[bisect.bisect_right(self.hit_t, t) - 1] if self.hit_t and self.hit_t[0] <= t else -9
            flash = max(0.0, 1 - (t - last) / 0.12)
            self.comet(d, pos, sim.balls[0].r, col, flash)
        if fin is not None:
            bx, by = sim.frames[min(int(fin * FPS), len(sim.frames) - 1)]["balls"][0][:2]
            self.finale(d, t, (bx, by), [r["color"] for r in sim.rings[::7]])

    def draw_multiply(self, d, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        fin = sim.finish_t
        R = ARENA * self.s + 8
        if fin is None or t < fin:
            g = sim.gap(t)
            a0, a1 = math.degrees(g + sim.half), math.degrees(g - sim.half) + 360
            d.arc([self.cx - R, self.cy - R, self.cx + R, self.cy + R], a0, a1, fill=hue_rgb(t * 0.04, 0.6),
                  width=max(8, int(10 * self.s)))
        else:
            p = min(1.0, (t - fin) / 0.5)
            d.ellipse([self.cx - R, self.cy - R, self.cx + R, self.cy + R], outline=mix(hue_rgb(t * 0.04, 0.6), 0.4 + 0.6 * p),
                      width=max(8, int(10 * self.s)))
        rr = sim.r * self.s
        # ausgebrochene Bälle: Funkenregen am Ausgang + wegfliegen
        lo = 0
        for (t0, x, y, vx, vy, h) in sim.fallen[max(0, fr["n_fallen"] - 120): fr["n_fallen"]]:
            age = t - t0
            if age > 1.0:
                continue
            X, Y = self.P(x + vx * age, y + vy * age + 0.5 * sim.gravity * age ** 2)
            col = hue_rgb(h)
            d.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=mix(col, 1 - age))
            if age < 0.3:
                Xe, Ye = self.P(x, y)
                r = (10 + 60 * ease_out(age / 0.3)) * self.s
                d.ellipse([Xe - r, Ye - r, Xe + r, Ye + r], outline=mix(col, 1 - age / 0.3, 0.6), width=3)
        self.hit_effects(d, t)
        if fin is None or t < fin:
            prev = sim.frames[max(0, fi - 2)]
            n_prev = len(prev["x"])
            for i, (x, y, h) in enumerate(zip(fr["x"], fr["y"], fr["hue"])):
                X, Y = self.P(x, y)
                col = hue_rgb(h)
                if i < n_prev:  # kurzer Schweif
                    Xp, Yp = self.P(prev["x"][i], prev["y"][i])
                    if abs(Xp - X) + abs(Yp - Y) < 120:
                        d.line([Xp, Yp, X, Y], fill=mix(col, 0.5 if fr["n"] < 400 else 0.3), width=max(2, int(rr)))
                d.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=col)
                if fr["n"] < 250:
                    d.ellipse([X - rr * 0.45, Y - rr * 0.45, X + rr * 0.45, Y + rr * 0.45], fill=(255, 255, 255))
        else:  # Finale: alle Bälle explodieren nach außen
            ff = sim.frames[min(int(fin * FPS), len(sim.frames) - 1)]
            age = t - fin
            if age < 1.4:
                rng = np.random.default_rng(5)
                k = rng.uniform(500, 1300, len(ff["x"]))
                for x, y, h, sp in zip(ff["x"], ff["y"], ff["hue"], k):
                    dd = math.hypot(x, y) or 1
                    px = x + x / dd * sp * age
                    py = y + y / dd * sp * age + 300 * age * age
                    X, Y = self.P(px, py)
                    Xq, Yq = self.P(x + x / dd * sp * max(0, age - 0.05), y + y / dd * sp * max(0, age - 0.05))
                    d.line([Xq, Yq, X, Y], fill=mix(hue_rgb(h), 1 - age / 1.4, 0.3), width=max(2, int(rr)))
            self.finale(d, t, (0, 0), [hue_rgb(t * 0.04 + k * 0.25) for k in range(4)])

    def draw_string(self, d, fi, t):
        sim, fr = self.sim, self.sim.frames[fi]
        fin = sim.finish_t
        R = ARENA * self.s + 4
        d.ellipse([self.cx - R, self.cy - R, self.cx + R, self.cy + R], outline=(150, 150, 190), width=max(4, int(5 * self.s)))
        flash = 0.0
        if fin is not None and t >= fin:
            age = t - fin
            flash = max(0.0, 1 - age / 0.5)
        balls = fr["balls"]
        pts = sim.points[: fr["n"]]
        pulse = 0.75 + 0.25 * math.sin(t * 3)
        for (px, py, h, bi) in pts:
            Px, Py = self.P(px, py)
            X, Y = self.P(*balls[bi])
            d.line([Px, Py, X, Y], fill=mix(hue_rgb(h, 0.8), 0.8 * pulse + 0.2, flash), width=2)
        for (px, py, h, bi) in pts:
            Px, Py = self.P(px, py)
            d.ellipse([Px - 5, Py - 5, Px + 5, Py + 5], fill=hue_rgb(h))
        self.hit_effects(d, t)
        if fin is None or t < fin + 0.05:
            for bi in range(len(balls)):
                pos = [sim.frames[j]["balls"][bi] for j in range(max(0, fi - 8), fi + 1)]
                self.comet(d, pos, sim.balls[bi].r, hue_rgb(fr["hue"] + bi * 0.5))
        if fin is not None:
            self.finale(d, t, balls[0], [hue_rgb(fr["hue"] + k * 0.2) for k in range(5)])

    # ------------------------------------------------------------ Frame

    def frame(self, fi):
        t = fi / FPS
        geo = Image.new("RGB", (self.W, self.H), (0, 0, 0))
        d = ImageDraw.Draw(geo)
        getattr(self, "draw_" + self.kind)(d, fi, t)
        arr = np.asarray(geo, np.float32)
        g1 = geo.resize((self.W // 4, self.H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
        g2 = geo.resize((self.W // 10, self.H // 10), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
        gk = 1.0
        if self.kind == "multiply":  # viele Bälle -> weniger Glow, sonst wird alles weiß
            gk = 1.0 - 0.6 * min(1.0, len(self.sim.frames[fi]["x"]) / 600)
        glow = (np.asarray(g1.resize((self.W, self.H), Image.BILINEAR), np.float32) * 1.3 +
                np.asarray(g2.resize((self.W, self.H), Image.BILINEAR), np.float32) * 1.4) * gk
        bgc = np.array(hue_rgb(0.68 + 0.08 * math.sin(t * 0.3), 0.7, 0.22), np.float32)
        out = self.grad * bgc + arr + glow
        fin = self.sim.finish_t
        if fin is not None and fin <= t < fin + 0.35:  # Finale-Blitz
            out += 255 * 0.85 * (1 - (t - fin) / 0.35)
        img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
        # Kamera-Ruck bei Ringbruch / Finale
        punch = 0.0
        for t0, _ in self.breaks:
            if 0 <= t - t0 < 0.3:
                punch = max(punch, 0.025 * (1 - (t - t0) / 0.3))
        if fin is not None and 0 <= t - fin < 0.5:
            punch = max(punch, 0.06 * (1 - (t - fin) / 0.5))
        if punch > 0:
            z = 1 + punch
            cw, ch = self.W / z, self.H / z
            jx = math.sin(t * 90) * punch * 120
            x0 = min(max(self.cx - self.cx / z + jx, 0), self.W - cw)
            y0 = min(max(self.cy - self.cy / z, 0), self.H - ch)
            img = img.resize((self.W, self.H), Image.BILINEAR, box=(x0, y0, x0 + cw, y0 + ch))
        return img


def render(sim, spec, fmt, out_path, preview_dir=None, preview_times=()):
    scene = HypnoScene(sim, spec["kind"], fmt)
    n = len(sim.frames)
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)
        for pt in preview_times:
            fr = scene.frame(min(int(pt * FPS), n - 1))
            base = os.path.join(preview_dir, f"{spec['id']}_{fmt}_{pt:05.1f}")
            fr.save(base + ".png")
            tiktok_ui_mock(fr, fmt).save(base + "_ui.png")
        return
    events = list(sim.events)
    if sim.finish_t is not None:
        events += [(sim.finish_t, "riser", {}), (sim.finish_t, "boom", {})]
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "a.wav")
        sound.build(n / FPS, events, wav)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{scene.W}x{scene.H}",
               "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-maxrate", "8M", "-bufsize", "16M",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fi in range(n):
            proc.stdin.write(scene.frame(fi).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg fehlgeschlagen")
