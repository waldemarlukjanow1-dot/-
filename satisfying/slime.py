"""Schleimpilz baut ein Bahnnetz (Physarum-Simulation, Agenten-Modell nach Jeff Jones 2010).

Tausende Agenten laufen über eine Spur-Karte: jeder riecht mit drei Sensoren (links, vorne, rechts), dreht sich
zur stärksten Spur, läuft einen Schritt und hinterlässt selbst Spur. Die Spur verteilt sich und verblasst.
Städte geben Duft ab. Der Schleim startet in einer Stadt, breitet sich im Land aus (die Grenze ist eine Wand)
und bildet Adern zwischen den Städten. Eine Stadt gilt als verbunden, sobald der Schleim sie erreicht.
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
W1, H1 = 1080, 1920
SR = 44100
SC = 2                      # Simulationsgitter: 1 Zelle = 2 Pixel

# Deutschland (vereinfachte Grenze, lon/lat) und Städte
GERMANY = [(8.6, 54.9), (9.4, 54.83), (9.9, 54.8), (10.15, 54.45), (11.0, 54.4), (11.2, 54.5), (10.9, 53.95),
           (11.5, 54.0), (12.1, 54.18), (12.5, 54.47), (13.1, 54.3), (13.4, 54.6), (13.7, 54.5), (14.2, 53.95),
           (14.4, 53.3), (14.1, 52.9), (14.6, 52.6), (14.7, 52.1), (14.6, 51.6), (15.0, 51.1), (14.8, 50.87),
           (14.0, 50.8), (13.5, 50.6), (12.9, 50.4), (12.2, 50.3), (12.1, 50.2), (12.5, 49.9), (12.4, 49.7),
           (12.9, 49.35), (13.4, 49.1), (13.8, 48.8), (13.45, 48.57), (13.0, 48.3), (12.8, 48.2), (12.9, 47.95),
           (13.0, 47.75), (12.9, 47.5), (12.2, 47.6), (11.3, 47.4), (10.5, 47.5), (10.2, 47.3), (9.6, 47.55),
           (8.6, 47.65), (7.6, 47.6), (7.6, 48.1), (8.0, 48.8), (8.2, 48.98), (7.6, 49.05), (7.0, 49.1),
           (6.4, 49.45), (6.4, 49.5), (6.1, 50.1), (6.4, 50.3), (6.0, 50.7), (6.0, 50.8), (5.9, 51.0), (6.2, 51.4),
           (5.95, 51.8), (6.8, 51.95), (7.0, 52.25), (7.05, 52.6), (7.2, 53.2), (7.0, 53.4), (7.2, 53.6),
           (8.0, 53.6), (8.1, 53.5), (8.5, 53.6), (8.7, 53.87), (8.9, 53.9), (8.9, 54.3), (8.6, 54.5)]
CITIES_DE = [("Berlin", 52.52, 13.40), ("Hamburg", 53.55, 9.99), ("Munich", 48.14, 11.58), ("Cologne", 50.94, 6.96),
             ("Frankfurt", 50.11, 8.68), ("Stuttgart", 48.78, 9.18), ("Dortmund", 51.51, 7.47),
             ("Leipzig", 51.34, 12.37), ("Dresden", 51.05, 13.74), ("Hanover", 52.37, 9.73),
             ("Nuremberg", 49.45, 11.08), ("Bremen", 53.08, 8.80), ("Kiel", 54.32, 10.13), ("Rostock", 54.09, 12.10),
             ("Freiburg", 47.99, 7.85), ("Erfurt", 50.98, 11.03), ("Saarbrücken", 49.24, 6.99),
             ("Magdeburg", 52.13, 11.63)]


class Region:
    """Projiziert lon/lat ins Bild (Feld zentriert, sicher vor TikTok-Elementen)."""

    def __init__(self, poly, box=(100, 300, 940, 1450)):
        lat0 = sum(p[1] for p in poly) / len(poly)
        self.k = math.cos(math.radians(lat0))
        xs = [p[0] * self.k for p in poly]
        ys = [-p[1] for p in poly]
        bx0, by0, bx1, by1 = box
        s = min((bx1 - bx0) / (max(xs) - min(xs)), (by1 - by0) / (max(ys) - min(ys)))
        self.s = s
        self.ox = bx0 + ((bx1 - bx0) - s * (max(xs) - min(xs))) / 2 - s * min(xs)
        self.oy = by0 + ((by1 - by0) - s * (max(ys) - min(ys))) / 2 - s * min(ys)

    def px(self, lon, lat):
        return self.ox + self.s * lon * self.k, self.oy - self.s * lat


class SlimeSim:
    def __init__(self, seed, poly=GERMANY, cities=CITIES_DE, start="Berlin", n_agents=120000, spawn_time=8.0,
                 steps_per_frame=2, total=56.0, sa=0.4, ra=0.785, so=9.0, speed=1.0, deposit=4.0, decay=0.93,
                 food=3.0, connect_at=4.0, cap=3, trail_max=60.0):
        self.rng = np.random.default_rng(seed)
        self.reg = Region(poly)
        self.W, self.H = W1 // SC, H1 // SC
        # Land-Maske
        m = Image.new("L", (self.W, self.H), 0)
        ImageDraw.Draw(m).polygon([tuple(v / SC for v in self.reg.px(*p)) for p in poly], fill=255)
        self.mask = np.asarray(m) > 0
        self.poly_px = [self.reg.px(*p) for p in poly]
        self.cities = [(n, *self.reg.px(lon, lat)) for n, lat, lon in cities]
        self.start = next(i for i, c in enumerate(self.cities) if c[0] == start)
        self.trail = np.zeros((self.H, self.W), np.float32)
        self.n_agents, self.spawn_time, self.spf, self.total = n_agents, spawn_time, steps_per_frame, total
        self.sa, self.ra, self.so, self.speed, self.dep, self.decay = sa, ra, so, speed, deposit, decay
        self.food, self.connect_at, self.cap, self.trail_max = food, connect_at, cap, trail_max
        self.occ = np.zeros((self.H, self.W), np.int32)
        self.ax = np.zeros(0, np.float32)
        self.ay = np.zeros(0, np.float32)
        self.ah = np.zeros(0, np.float32)
        self.connected = {}          # Stadt-Index -> Zeit
        self.all_t = None
        self.gain = 2.0
        self.frames = []             # komprimierte Spur-Karten (uint8)
        self.events = []
        # Duft-Quellen (Städte) als kleine Scheiben
        yy, xx = np.mgrid[0:self.H, 0:self.W]
        self.city_masks = []
        for n, x, y in self.cities:
            d = np.hypot(xx - x / SC, yy - y / SC)
            self.city_masks.append(d <= 3.0)
        self.food_map = np.zeros((self.H, self.W), np.float32)
        for cm in self.city_masks:
            self.food_map[cm] = 1.0

    def spawn(self, n):
        _, x, y = self.cities[self.start]
        r = np.sqrt(self.rng.random(n)) * 6
        a = self.rng.random(n) * 2 * np.pi
        self.ax = np.concatenate([self.ax, (x / SC + np.cos(a) * r).astype(np.float32)])
        self.ay = np.concatenate([self.ay, (y / SC + np.sin(a) * r).astype(np.float32)])
        self.ah = np.concatenate([self.ah, (self.rng.random(n) * 2 * np.pi).astype(np.float32)])

    def sense(self, ang):
        x = np.clip((self.ax + np.cos(ang) * self.so).astype(np.int32), 0, self.W - 1)
        y = np.clip((self.ay + np.sin(ang) * self.so).astype(np.int32), 0, self.H - 1)
        return self.trail[y, x]

    def step(self):
        h = self.ah
        f = self.sense(h)
        l = self.sense(h - self.sa)
        r = self.sense(h + self.sa)
        rnd = self.rng.random(len(h)).astype(np.float32)
        turn = np.zeros_like(h)
        both = (l > f) & (r > f)
        turn[both] = np.where(rnd[both] < 0.5, -self.ra, self.ra)
        only_l = (l > f) & (l > r) & ~both
        only_r = (r > f) & (r > l) & ~both
        turn[only_l] = -self.ra
        turn[only_r] = self.ra
        h = h + turn + (rnd - 0.5) * 0.1
        nx = self.ax + np.cos(h) * self.speed
        ny = self.ay + np.sin(h) * self.speed
        xi = np.clip(nx.astype(np.int32), 0, self.W - 1)
        yi = np.clip(ny.astype(np.int32), 0, self.H - 1)
        ok = self.mask[yi, xi]
        # Platzregel: in eine volle Zelle darf niemand nachrücken (hält den Schleim locker und lässt ihn wachsen)
        cx = np.clip(self.ax.astype(np.int32), 0, self.W - 1)
        cy = np.clip(self.ay.astype(np.int32), 0, self.H - 1)
        same = (cx == xi) & (cy == yi)
        ok &= same | (self.occ[yi, xi] < self.cap)
        # an der Grenze: stehen bleiben, neue zufällige Richtung
        h = np.where(ok, h, self.rng.random(len(h)).astype(np.float32) * 2 * np.pi)
        self.ax = np.where(ok, nx, self.ax)
        self.ay = np.where(ok, ny, self.ay)
        self.ah = h
        xi = np.clip(self.ax.astype(np.int32), 0, self.W - 1)
        yi = np.clip(self.ay.astype(np.int32), 0, self.H - 1)
        np.add.at(self.trail, (yi, xi), self.dep)
        self.occ[:] = 0
        np.add.at(self.occ, (yi, xi), 1)
        # Städte duften
        self.trail += self.food_map * self.food
        # verteilen + verblassen
        t = self.trail
        b = (t + np.roll(t, 1, 0) + np.roll(t, -1, 0) + np.roll(t, 1, 1) + np.roll(t, -1, 1) +
             np.roll(np.roll(t, 1, 0), 1, 1) + np.roll(np.roll(t, 1, 0), -1, 1) +
             np.roll(np.roll(t, -1, 0), 1, 1) + np.roll(np.roll(t, -1, 0), -1, 1)) / 9.0
        self.trail = np.minimum(b * self.decay, self.trail_max) * self.mask

    def run(self):
        n_frames = int(self.total * FPS)
        dens = np.zeros((self.H, self.W), np.float32)
        for fi in range(n_frames):
            t = fi / FPS
            want = int(self.n_agents * min(1.0, (t + 0.3) / self.spawn_time))
            if want > len(self.ax):
                self.spawn(want - len(self.ax))
            for _ in range(self.spf):
                self.step()
            # Agentendichte = sichtbarer Schleim
            dens[:] = 0
            xi = np.clip(self.ax.astype(np.int32), 0, self.W - 1)
            yi = np.clip(self.ay.astype(np.int32), 0, self.H - 1)
            np.add.at(dens, (yi, xi), 1.0)
            for i, cm in enumerate(self.city_masks):
                if i not in self.connected and (i == self.start or dens[cm].sum() >= self.connect_at):
                    self.connected[i] = t
                    self.events.append((t, "city", {"i": i, "n": len(self.connected)}))
            if self.all_t is None and len(self.connected) == len(self.cities):
                self.all_t = t
                self.events.append((t, "all", {}))
            v = np.log1p(dens * self.gain) / math.log1p(40.0)
            self.frames.append((np.clip(v, 0, 1) * 255).astype(np.uint8))
        self.total_t_for_clock = self.all_t or self.total
        return self


# ------------------------------------------------------------------ Darstellung

F_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
SLIME_LUT = None


def slime_lut():
    """Farbverlauf für den Schleim: dunkles Ocker -> leuchtendes Gelb -> fast Weiß."""
    stops = [(0.0, (0, 0, 0)), (0.12, (70, 45, 0)), (0.35, (220, 160, 10)), (0.6, (255, 220, 40)),
             (0.85, (255, 245, 150)), (1.0, (255, 255, 235))]
    lut = np.zeros((256, 3), np.float32)
    for i in range(256):
        v = i / 255
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if a <= v <= b:
                f = (v - a) / (b - a)
                lut[i] = [x + (y - x) * f for x, y in zip(ca, cb)]
                break
    return lut


class SlimeRenderer:
    def __init__(self, sim, title=("This slime has no brain.", "Can it build Germany's rail network?"),
                 ending=("No brain. Just biology.",), hours=26.0):
        self.sim, self.title, self.ending, self.hours = sim, title, ending, hours
        self.lut = slime_lut()
        # statische Ebene: Hintergrund, Land, Grenze (2x gerechnet, dann verkleinert)
        k = 2
        yy, xx = np.mgrid[0:H1, 0:W1].astype(np.float32)
        vig = 0.5 + 0.5 * np.exp(-(((xx - W1 / 2) / W1) ** 2 + ((yy - H1 * 0.45) / H1) ** 2) * 3.0)
        base = np.array((7, 9, 18), np.float32) * vig[..., None]
        land = Image.new("L", (W1 * k, H1 * k), 0)
        ImageDraw.Draw(land).polygon([(x * k, y * k) for x, y in sim.poly_px], fill=255)
        landm = np.asarray(land.reduce(k), np.float32) / 255
        base = base + landm[..., None] * np.array((10, 14, 26), np.float32)
        edge = Image.new("L", (W1 * k, H1 * k), 0)
        ImageDraw.Draw(edge).line([(x * k, y * k) for x, y in sim.poly_px + sim.poly_px[:1]], fill=255, width=3 * k,
                                  joint="curve")
        e = np.asarray(edge.reduce(k), np.float32) / 255
        eg = np.asarray(edge.reduce(k).filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
        base = base + e[..., None] * np.array((60, 90, 140), np.float32) + eg[..., None] * np.array((30, 50, 90),
                                                                                                    np.float32)
        self.base = base
        self.f_label = ImageFont.truetype(F_SEMI, 22)
        self.f_label_b = ImageFont.truetype(F_BOLD, 23)
        self.f_hud = ImageFont.truetype(F_BOLD, 34)
        self.f_hud_s = ImageFont.truetype(F_SEMI, 18)

    def frame(self, fi):
        sim = self.sim
        t = (fi + 1) / FPS
        v = Image.fromarray(sim.frames[fi]).resize((W1, H1), Image.BILINEAR)
        col = self.lut[np.asarray(v)]
        arr = self.base + col
        # Glow
        g = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)).resize((W1 // 6, H1 // 6), Image.BILINEAR)
        glow = np.asarray(g.filter(ImageFilter.GaussianBlur(5)).resize((W1, H1), Image.BILINEAR), np.float32)
        arr = arr + glow * 0.9
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        d = ImageDraw.Draw(im, "RGBA")
        # Städte
        for i, (name, x, y) in enumerate(sim.cities):
            ct = sim.connected.get(i)
            on = ct is not None and ct <= t
            if on:
                e = t - ct
                if e < 0.9:   # Ring beim Verbinden
                    R = 8 + 46 * (e / 0.9)
                    d.ellipse([x - R, y - R, x + R, y + R], outline=(255, 235, 120, int(230 * (1 - e / 0.9))), width=3)
                d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 240, 160, 255), outline=(255, 255, 255, 255), width=2)
            else:
                d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(20, 26, 44, 230), outline=(200, 210, 235, 200), width=2)
            right = x < 780
            tx = x + 15 if right else x - 15
            f = self.f_label_b if on else self.f_label
            d.text((tx, y), name, font=f, fill=(255, 255, 255, 240) if on else (175, 185, 210, 200),
                   anchor="lm" if right else "rm", stroke_width=3, stroke_fill=(5, 6, 12, 220))
        # HUD oben
        n = sum(1 for ct in sim.connected.values() if ct <= t)
        N = len(sim.cities)
        hrs = min(self.hours, t / max(1e-6, sim.total_t_for_clock) * self.hours)
        for (txt, sub), cx in (((f"{n}/{N}", "CITIES CONNECTED"), 300), ((f"{int(hrs)} h", "TIME-LAPSE"), 780)):
            d.rounded_rectangle([cx - 120, 222, cx + 120, 288], radius=16, fill=(0, 0, 0, 160),
                                outline=(255, 220, 80, 120), width=2)
            d.text((cx, 244), txt, font=self.f_hud, fill=(255, 235, 140, 255), anchor="mm")
            d.text((cx, 276), sub, font=self.f_hud_s, fill=(210, 210, 225, 220), anchor="mm")
        # Hook am Anfang, Auflösung am Ende
        a = min(1.0, t / 0.25) * min(1.0, max(0.0, (3.8 - t) / 0.5))
        self.banner(d, [(self.title[0], 60, (255, 255, 255)), (self.title[1], 38, (255, 225, 110))], a, 860)
        if sim.all_t is not None and t > sim.all_t + 0.8:
            a = min(1.0, (t - sim.all_t - 0.8) / 0.6)
            self.banner(d, [(f"All {N} cities connected.", 52, (255, 235, 140))] +
                        [(x, 38, (235, 235, 245)) for x in self.ending], a, 860)
        return im

    @staticmethod
    def banner(d, lines, alpha, cy):
        if alpha <= 0.01:
            return
        fonts = [ImageFont.truetype(F_BOLD, sz) for _, sz, _ in lines]
        ws = [d.textlength(txt, font=fo) for (txt, _, _), fo in zip(lines, fonts)]
        H = sum(sz for _, sz, _ in lines) * 1.3 + 44
        W = min(860, max(ws) + 70)
        cx = 520
        d.rounded_rectangle([cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2], radius=28,
                            fill=(5, 6, 14, int(225 * alpha)), outline=(255, 220, 90, int(90 * alpha)), width=2)
        yy = cy - H / 2 + 22
        for (txt, sz, col), fo in zip(lines, fonts):
            d.text((cx, yy), txt, font=fo, fill=col + (int(255 * alpha),), anchor="ma")
            yy += sz * 1.3


# ------------------------------------------------------------------ Klang

def build_audio(sim, path):
    from .viral import soft_tone, warm_pad
    n = int(SR * sim.total)
    out = np.zeros(n + SR * 3)

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    penta = [0, 2, 4, 7, 9]
    for t, kind, d in sim.events:
        if kind == "city":
            k = d["n"] - 1
            f = 392.0 * 2 ** ((12 * (k // 5) + penta[k % 5]) / 12)
            put(t, soft_tone(f), 0.55)
            put(t + 0.08, soft_tone(f * 2), 0.18)
    if sim.all_t is not None:
        for j, f in enumerate((523.25, 659.25, 783.99, 1046.5)):
            put(sim.all_t + 0.8 + j * 0.13, soft_tone(f), 0.45)
    out = out[:n]
    tt = np.arange(n) / SR
    mix_ = out * 0.85 + reverb(out, 2.2) * 0.35 + warm_pad(sim.total, sim.total - 6) * 1.4
    mix_ *= np.clip((sim.total - tt) / 1.5, 0, 1) * np.clip(tt / 0.8, 0, 1)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = np.tanh(mix_ / peak * 1.1) * 0.9
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


def render(sim, name, out_path, preview_dir=None, preview_times=(), jobs=4, **rkw):
    global _R
    _R = SlimeRenderer(sim, **rkw)
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
