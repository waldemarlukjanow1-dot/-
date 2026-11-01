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

CX, CY = 520.0, 885.0
A, B = 590.0, 380.0                 # Halbachsen (vertikal, horizontal)
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
            cols = [hue((th % (2 * math.pi)) / (2 * math.pi) + 0.02 * k) for th in ths]
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
    def __init__(self, sim):
        self.sim = sim
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        vig = 0.35 + 0.65 * np.exp(-(((xx - CX) / 700) ** 2 + ((yy - CY) / 1000) ** 2))
        base = np.array((14, 10, 18), np.float32) * vig[..., None]
        self.base = Image.fromarray(base.astype(np.uint8)).convert("RGBA")
        # Ellipse mit weichem Glow
        ring = Image.new("L", (W1 * k, H1 * k), 0)
        ImageDraw.Draw(ring).ellipse([(CX - B) * k, (CY - A) * k, (CX + B) * k, (CY + A) * k], outline=255,
                                     width=5 * k)
        gl = ring.filter(ImageFilter.GaussianBlur(9 * k))
        arr = np.asarray(self.base, np.float32)
        arr[..., :3] += (np.asarray(gl, np.float32) / 255 * 120)[..., None] + (np.asarray(ring, np.float32) / 255 *
                                                                               235)[..., None]
        self.base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")
        self.sprites = {}

    def sprite(self, r, col):
        key = (round(r * 2) / 2, col)
        if key not in self.sprites:
            self.sprites[key] = dot_sprite(key[0] * SS, col)
        return self.sprites[key]

    def frame(self, fi):
        sim = self.sim
        k = SS
        t = (fi + 1) / FPS
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
        rr = 9.0 if n < 40 else 7.0 if n < 200 else 5.0 if n < 600 else 4.0
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
            d.ellipse([fx * k - R, fy * k - R, fx * k + R, fy * k + R], outline=(235, 235, 240, 255), width=3 * k,
                      fill=(14, 10, 18, 255))
        # Treffen: Regenbogen-Ringe am Brennpunkt
        for j, mt in enumerate(sim.meets):
            e = t - mt
            if 0 <= e < 1.2:
                fx, fy = FOCI[1 - (j % 2)]
                nn = len(sim.trips[j][3])
                for q in range(7):
                    R = (24 + 120 * e ** 0.6 + q * 7) * k
                    al = int(210 * (1 - e / 1.2) * (1 - q / 8))
                    d.ellipse([fx * k - R, fy * k - R, fx * k + R, fy * k + R], outline=hue(q / 7 + e) + (al,),
                              width=3 * k)
                if e < 0.25:
                    R = (30 + 40 * e) * k
                    d.ellipse([fx * k - R, fy * k - R, fx * k + R, fy * k + R],
                              fill=(255, 255, 255, int(170 * (1 - e / 0.25))))
        # Schluss: alle Bälle vereint als leuchtende Regenbogen-Kugel im Brennpunkt
        if t > sim.end_t:
            e = t - sim.end_t
            fx, fy = F1
            a_ = min(1.0, e / 0.6)
            R = (34 + 4 * math.sin(e * 2.5)) * k
            for q in range(12):
                ang0 = q / 12 * 2 * math.pi + e * 0.8
                d.pieslice([fx * k - R, fy * k - R, fx * k + R, fy * k + R], math.degrees(ang0),
                           math.degrees(ang0) + 31, fill=hue(q / 12) + (int(235 * a_),))
            Ri = R * 0.45
            d.ellipse([fx * k - Ri, fy * k - Ri, fx * k + Ri, fy * k + Ri], fill=(255, 255, 255, int(240 * a_)))
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


# ------------------------------------------------------------------ Klang

PENTA = [0, 2, 4, 7, 9]


def bell(f, dur=1.6):
    t = np.arange(int(SR * dur)) / SR
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t * 3.2) + 0.3 * np.sin(2 * np.pi * f * 2 * t) * np.exp(-t * 6)
         + 0.1 * np.sin(2 * np.pi * f * 3 * t) * np.exp(-t * 10))
    return s * np.minimum(t / 0.004, 1) * 0.3


def pluck(f):
    t = np.arange(int(SR * 0.35)) / SR
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 16) * np.minimum(t / 0.002, 1) * 0.22


def build_audio(sim, path):
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
        if kind == "wall":
            if t - last < 0.05:
                continue
            last = t
            v = (d["th"] % (2 * math.pi)) / (2 * math.pi)
            j = int(v * 10)
            put(t, pluck(523.25 * 2 ** ((PENTA[j % 5] + 12 * (j // 5)) / 12)), 0.4)
        else:
            k = d["k"]
            semi = PENTA[k % 5] + 12 * min(2, k // 5)
            f = 261.63 * 2 ** (semi / 12)
            for q, m in enumerate((1.0, 1.5, 2.0)):
                put(t + q * 0.03, bell(f * m), 0.5 / (q + 1))
    out = out[:n]
    tt = np.arange(n) / SR
    pad = sum(np.sin(2 * np.pi * f * tt + j) / (j + 2) for j, f in enumerate((130.81, 196.0, 261.63)))
    mix_ = out + reverb(out, 1.8) * 0.3 + pad * 0.015 * np.minimum(1, tt / 2)
    mix_ *= np.clip((total - tt) / 1.5, 0, 1)
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
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-maxrate", "14M", "-bufsize", "28M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_R.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(sim, out_path, jobs=4):
    global _R
    _R = LoopRenderer(sim)
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
        build_audio(sim, wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


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
        render(sim, "output/e01_ellipse_loop_vertical.mp4")
        print("output/e01_ellipse_loop_vertical.mp4")
