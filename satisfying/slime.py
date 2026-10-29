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
GERMANY = [(8.6, 54.9), (9.4, 54.83), (9.9, 54.8), (10.15, 54.45), (11.0, 54.35),
           (11.5, 54.05), (12.1, 54.18), (12.5, 54.47), (13.1, 54.3), (13.4, 54.6), (13.7, 54.5), (14.2, 53.95),
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
    """Wachstum eines Schleimpilz-Netzes von einer Start-Stadt aus.

    Das Netz folgt dem, was Physarum im Experiment tut: zuerst ein minimales Verbindungsnetz (kürzeste Gesamtlänge),
    später einige Querverbindungen, die Ausfallsicherheit geben. Die Adern wachsen mit konstanter Geschwindigkeit,
    an der Spitze tasten feine Fühler; seitlich wachsen Suchfühler ins Leere und ziehen sich wieder zurück."""

    def __init__(self, seed, poly=GERMANY, cities=CITIES_DE, start="Berlin", reach_t=40.0, loops=5, total=None):
        self.rng = random.Random(seed)
        self.reg = Region(poly)
        self.poly_px = [self.reg.px(*p) for p in poly]
        m = Image.new("L", (W1, H1), 0)
        ImageDraw.Draw(m).polygon(self.poly_px, fill=255)
        self.mask = np.asarray(m) > 0
        self.cities = [(n, *self.reg.px(lon, lat)) for n, lat, lon in cities]
        P = np.array([(x, y) for _, x, y in self.cities])
        n = len(P)
        D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
        self.start = next(i for i, c in enumerate(self.cities) if c[0] == start)
        # minimaler Spannbaum (Prim) ab Start-Stadt
        inside, edges = {self.start}, []
        while len(inside) < n:
            a, b = min(((a, b) for a in inside for b in range(n) if b not in inside), key=lambda e: D[e])
            edges.append((a, b))
            inside.add(b)
        # Ankunftszeiten (Geschwindigkeit so, dass die letzte Stadt bei reach_t erreicht wird)
        arr = {self.start: 0.0}
        for a, b in edges:
            arr[b] = arr[a] + D[a, b]
        lead = 1.2          # kurzer Auftakt, dann wächst der Schleim los
        v = max(arr.values()) / (reach_t - lead - 0.35 * self._depth(edges))
        self.speed = v
        self.edges = []      # (a, b, t_start, t_end, punkte)
        t_city = {self.start: 0.0}
        for a, b in edges:
            t0 = t_city[a] + 0.35
            t1 = t0 + D[a, b] / v
            t_city[b] = t1
            self.edges.append((a, b, t0, t1, self.path(P[a], P[b])))
        # Zeitverlauf: anfangs schneller (früh die erste Stadt), dann gleichmäßig
        T = max(t_city.values())
        warp = lambda x: lead + (reach_t - lead) * (x / T) ** 1.35 if x > 0 else lead
        self.edges = [(a, b, warp(t0), warp(t1), pts) for a, b, t0, t1, pts in self.edges]
        self.connected = {i: warp(t) for i, t in t_city.items()}
        self.connected[self.start] = 0.0
        # Querverbindungen (Schleifen): kurze, nicht kreuzende Zusatz-Kanten, wachsen später
        tree = {frozenset(e) for e in edges}
        cand = sorted(((D[a, b], a, b) for a in range(n) for b in range(a + 1, n) if frozenset((a, b)) not in tree))
        added = 0
        segs = [(P[a], P[b]) for a, b in edges]
        for d, a, b in cand:
            if added >= loops or d > 330:
                break
            if any(_cross(P[a], P[b], s0, s1) for s0, s1 in segs):
                continue
            t0 = max(self.connected[a], self.connected[b]) + 1.5 + added * 1.2
            self.edges.append((a, b, t0, t0 + d / v, self.path(P[a], P[b])))
            segs.append((P[a], P[b]))
            added += 1
        self.all_t = max(self.connected.values())
        self.done_t = max(e[3] for e in self.edges)
        self.total = total or max(self.all_t, self.done_t) + 6.0
        self.n_frames = int(self.total * FPS)
        self.total_t_for_clock = self.all_t
        self.events = sorted([(t, "city", {"i": i}) for i, t in self.connected.items() if i != self.start])
        for k, ev in enumerate(self.events):
            ev[2]["n"] = k + 2
        self.events.append((self.all_t, "all", {}))
        self.tendrils = self.make_tendrils()

    @staticmethod
    def _depth(edges):
        depth = {}
        for a, b in edges:
            depth[b] = depth.get(a, 0) + 1
        return max(depth.values())

    def path(self, p0, p1, n=48):
        """Geschwungene Ader zwischen zwei Städten."""
        rng = self.rng
        p0, p1 = np.array(p0, float), np.array(p1, float)
        d = p1 - p0
        L = np.hypot(*d)
        nrm = np.array([-d[1], d[0]]) / max(L, 1e-6)
        u = np.linspace(0, 1, n)
        ph = [rng.uniform(0, 6.28) for _ in range(3)]
        off = (np.sin(np.pi * u) * (0.07 * np.sin(2 * np.pi * u + ph[0]) + 0.04 * np.sin(5 * np.pi * u + ph[1]))
               + 0.012 * np.sin(13 * np.pi * u + ph[2]) * np.sin(np.pi * u)) * L
        pts = p0[None] + d[None] * u[:, None] + nrm[None] * off[:, None]
        return pts

    def make_tendrils(self):
        """Suchfühler: wachsen seitlich aus fertigen Adern, verzweigen sich, ziehen sich zurück."""
        rng = self.rng
        out = []
        t = 1.0
        while t < self.total - 3:
            t += rng.uniform(0.12, 0.3)
            ready = [e for e in self.edges if e[2] < t]
            if not ready:
                continue
            a, b, t0, t1, pts = rng.choice(ready)
            frac = min(1.0, (t - t0) / max(1e-6, t1 - t0))
            k = int(rng.uniform(0.05, max(0.06, frac * 0.95)) * (len(pts) - 1))
            x, y = pts[k]
            dx, dy = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
            base = math.atan2(dy, dx) + rng.choice((-1, 1)) * rng.uniform(0.9, 2.2)
            branches = []
            def grow(x, y, ang, L, depth):
                seg = [(x, y)]
                for _ in range(7):
                    ang += rng.uniform(-0.45, 0.45)
                    x += math.cos(ang) * L / 7
                    y += math.sin(ang) * L / 7
                    if not (0 <= int(x) < W1 and 0 <= int(y) < H1 and self.mask[int(y), int(x)]):
                        break
                    seg.append((x, y))
                if len(seg) > 2:
                    branches.append(seg)
                    if depth < 2:
                        j = rng.randint(1, len(seg) - 1)
                        grow(*seg[j], ang + rng.choice((-1, 1)) * rng.uniform(0.5, 1.0), L * 0.55, depth + 1)
            grow(x, y, base, rng.uniform(40, 110), 0)
            if branches:
                out.append((t, rng.uniform(1.0, 1.6), rng.uniform(0.8, 1.8), rng.uniform(1.2, 2.0), branches))
        return out


def _cross(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    if any(np.allclose(x, y) for x in (a, b) for y in (c, d)):
        return False
    return (ccw(a, c, d) * ccw(b, c, d) < 0) and (ccw(a, b, c) * ccw(a, b, d) < 0)


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
        v = self.intensity(t)
        col = self.lut[v]
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
        if sim.all_t is not None and sim.all_t + 0.8 < t < sim.all_t + 5.5:
            e = t - sim.all_t - 0.8
            a = min(1.0, e / 0.6) * min(1.0, (4.7 - e) / 0.8)
            self.banner(d, [(f"All {N} cities connected.", 52, (255, 235, 140))] +
                        [(x, 38, (235, 235, 245)) for x in self.ending], a, 860)
        return im

    def intensity(self, t):
        """Helligkeit des Schleims (0..255) zum Zeitpunkt t, 2x gezeichnet und verkleinert."""
        sim, k = self.sim, 2
        im = Image.new("L", (W1 * k, H1 * k), 0)
        d = ImageDraw.Draw(im)
        rng = random.Random(int(t * 1000))
        # Suchfühler (unter den Adern)
        for t0, g, hold, fade, branches in sim.tendrils:
            e = t - t0
            if e < 0 or e > g + hold + fade:
                continue
            frac = min(1.0, e / g) if e < g + hold else max(0.0, 1 - (e - g - hold) / fade)
            val = int(150 * (0.5 + 0.5 * frac))
            for br in branches:
                m = max(2, int(round(frac * len(br))))
                if m >= 2:
                    d.line([(x * k, y * k) for x, y in br[:m]], fill=val, width=int(1.6 * k), joint="curve")
        # Adern
        for a_, b_, t0, t1, pts in sim.edges:
            if t < t0:
                continue
            frac = min(1.0, (t - t0) / (t1 - t0))
            n = len(pts)
            fk = frac * (n - 1)
            j = int(fk)
            seg = [tuple(p) for p in pts[:j + 1]]
            if j < n - 1:
                tip = pts[j] + (pts[j + 1] - pts[j]) * (fk - j)
                seg.append(tuple(tip))
            w = 2.4 + (3.2 * min(1.0, (t - t1) / 6.0) if t > t1 else 0.0)
            if len(seg) >= 2:
                d.line([(x * k, y * k) for x, y in seg], fill=235, width=max(2, int(w * k)), joint="curve")
            if frac < 1.0:   # tastende Fühler an der Spitze
                tx, ty = seg[-1]
                ang = math.atan2(pts[min(j + 1, n - 1)][1] - pts[j][1], pts[min(j + 1, n - 1)][0] - pts[j][0])
                for q in range(5):
                    a2 = ang + (q - 2) * 0.32 + rng.uniform(-0.12, 0.12)
                    L = rng.uniform(14, 30)
                    d.line([(tx * k, ty * k), ((tx + math.cos(a2) * L) * k, (ty + math.sin(a2) * L) * k)],
                           fill=130, width=k)
            else:            # Strömung: helle Pulse wandern durch fertige Adern
                L = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
                for q in range(max(1, int(L / 90))):
                    u = ((t - t1) * 70 / L + q / max(1, int(L / 90))) % 1.0
                    x, y = pts[int(u * (n - 1))]
                    r = (w * 0.9 + 1.2) * k
                    d.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=255)
        # Körper in der Start-Stadt
        _, sx, sy = sim.cities[sim.start]
        R = (6 + 14 * min(1.0, t / 2.5) + 1.5 * math.sin(t * 2.3)) * k
        d.ellipse([sx * k - R, sy * k - R, sx * k + R, sy * k + R], fill=245)
        return np.asarray(im.reduce(k))

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
    n = sim.n_frames
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


SLIME = {"s01_slime_germany": lambda: SlimeSim(SEED_SLIME)}
SEED_SLIME = 1

if __name__ == "__main__":
    import sys
    preview = "--preview" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SLIME)
    for name in names:
        sim = SLIME[name]()
        print(name, "Dauer", round(sim.total, 1), "alle verbunden", round(sim.all_t, 1))
        if preview:
            ts = [1.5, 5.0, 10.0, 16.0, 22.0, 28.0, 34.0, 40.0, sim.all_t + 0.5, sim.total - 1]
            render(sim, name, None, preview_dir="output/preview", preview_times=ts)
        else:
            out = f"output/{name}_vertical.mp4"
            render(sim, name, out)
            print(out)
