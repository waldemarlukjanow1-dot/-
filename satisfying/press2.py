"""Neue Presse im Kreis: schmaler Stempel an einer Stange fährt von oben herab bis auf einen kleinen Amboss am
tiefsten Punkt des Kreises. Ein Ball, der unter dem Stempel eingeklemmt wird (gegen den Amboss oder gegen andere
Bälle darunter), wird zerquetscht und verdoppelt sich. Der Kreis füllt sich ruhig und gleichmäßig.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tiktok.render import tiktok_ui_mock
from . import multiply as M
from .multiply import CX, CY, R_AR, FPS, W1, H1, SS, Scene, poly_body, ball_col, _font, F_BOLD

ANVIL_TOP = CY + R_AR - 26          # Oberkante des Ambosses (tiefster Punkt des Kreises)


def rounded_rect(x0, y0, x1, y1, rad, steps=5):
    pts = []
    for cx_, cy_, a0 in ((x1 - rad, y0 + rad, -90), (x1 - rad, y1 - rad, 0), (x0 + rad, y1 - rad, 90),
                         (x0 + rad, y0 + rad, 180)):
        for q in range(steps + 1):
            a = math.radians(a0 + 90 * q / steps)
            pts.append((cx_ + math.cos(a) * rad, cy_ + math.sin(a) * rad))
    return np.array(pts, np.float64)


class PressCircle(Scene):
    R = 11.0
    CAP = 2400
    DUR = 70.0
    SUB = 8
    HW = 62.0                        # halbe Breite des Stempels
    HEAD_H = 70.0
    REST = CY - 150                  # Unterkante des Stempels in Ruhe
    BEAT = 1.8
    GAP = 1.2                        # jeder Ball höchstens einmal pro Pressvorgang
    N_FULL = 2000
    OUTRO = 4.0

    def __init__(self, seed, n0=1):
        super().__init__(seed, n0)
        # Amboss = Kreisboden: zwei leicht schräge Platten, die Bälle rollen sanft in die Mitte unter den Stempel
        hw = math.sqrt(R_AR ** 2 - (ANVIL_TOP - 16 - CY) ** 2) + 30
        self.anvils = [np.array([(CX - hw, ANVIL_TOP - 16), (CX, ANVIL_TOP), (CX, CY + R_AR + 80),
                                 (CX - hw, CY + R_AR + 80)], np.float64),
                       np.array([(CX, ANVIL_TOP), (CX + hw, ANVIL_TOP - 16), (CX + hw, CY + R_AR + 80),
                                 (CX, CY + R_AR + 80)], np.float64)]
        self.anvil = np.concatenate([self.anvils[0][:2], self.anvils[1][1:3], self.anvils[0][3:]])
        self.full_t = None

    def start(self, P, V):
        P[0] = (CX + 40, CY - 120)
        V[0] = (-30, 0)

    def press_y(self, t):
        """Ruhig: warten, gleichmäßig herab, kurz drücken, sanft hoch."""
        low = ANVIL_TOP - 2 * self.R * 0.62
        if self.full_t is not None and t >= self.full_t:
            e = min(1.0, (t - self.full_t) / 1.2)
            y0 = self._y_full
            return y0 + (self.REST - y0) * e * e * (3 - 2 * e)
        u = (t / self.BEAT) % 1.0
        if u < 0.06:
            f = 0.0
        elif u < 0.42:
            v = (u - 0.06) / 0.36
            f = 0.5 * v * v * (3 - 2 * v) + 0.5 * v * v
        elif u < 0.5:
            f = 1.0
        else:   # langsam und gleichmäßig wieder hoch (wirft nichts durch die Gegend)
            v = (u - 0.5) / 0.5
            f = 1 - (v * v * (3 - 2 * v) * 0.4 + 0.6 * v)
        return self.REST + (low - self.REST) * f

    def press_vy(self, t):
        e = 1e-3
        return (self.press_y(t + e) - self.press_y(t - e)) / (2 * e)

    def head(self, y):
        return rounded_rect(CX - self.HW, y - self.HEAD_H, CX + self.HW, y, 12)

    def machine(self, P, V, n, t, touch, impact):
        y = self.press_y(t)
        vy = self.press_vy(t)
        poly_body(P, V, n, self.R, self.head(y), 0.0, vy, 0.0, 0.0, 0.0, 0.2, touch, 1, impact, 1e9)
        rod = np.array([(CX - 14, CY - R_AR - 80), (CX + 14, CY - R_AR - 80), (CX + 14, y - self.HEAD_H + 2),
                        (CX - 14, y - self.HEAD_H + 2)], np.float64)
        poly_body(P, V, n, self.R, rod, 0.0, vy, 0.0, 0.0, 0.0, 0.2, touch, 4, impact, 1e9)
        for q in self.anvils:
            poly_body(P, V, n, self.R, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, touch, 2, impact, 1e9)

    def crushed(self, P, V, n, t, touch, impact):
        if self.full_t is not None or self.press_vy(t) < 30:
            return np.zeros(0, np.int64)
        y = self.press_y(t)
        under = ((touch[:n] & 1) == 1) & (np.abs(P[:n, 0] - CX) < self.HW) & (P[:n, 1] > y)
        idx = np.nonzero(under)[0]
        if len(idx) == 0:
            return idx
        out = []
        for i in idx:   # eingeklemmt: Amboss oder ein anderer Ball direkt darunter
            if touch[i] & 2:
                out.append(i)
                continue
            dx = P[:n, 0] - P[i, 0]
            dy = P[:n, 1] - P[i, 1]
            if np.any((dy > self.R * 0.8) & (dy < self.R * 2.3) & (np.abs(dx) < self.R * 1.6)):
                out.append(i)
        return np.array(out, np.int64)

    def eject(self, P, V, i, j, t):
        """Ruhig: die zwei Hälften gleiten links und rechts am Stempel vorbei nach außen."""
        rng = self.rng
        s = 1 if P[i, 0] >= CX else -1
        for k, sg in ((i, s), (j, -s)):
            P[k, 0] = CX + sg * (self.HW + self.R + 1 + rng.uniform(0, 4))
            a = -math.pi / 2 + sg * rng.uniform(0.9, 1.3)
            sp = rng.uniform(180, 320)
            V[k] = (math.cos(a) * sp, math.sin(a) * sp)

    def state(self, t):
        return (self.press_y(t),)

    def frame_hook(self, P, n, t):
        if self.full_t is None and n >= self.N_FULL:
            self._y_full = self.press_y(t)
            self.full_t = t
            self.events.append((t, "full", {}))
        return self.full_t is not None and t > self.full_t + self.OUTRO


# ------------------------------------------------------------------ Darstellung (neuer Kreis)

import bisect
import os
import subprocess
import tempfile
import wave
from multiprocessing import Pool
from .marble_render import reverb

RING_A, RING_B = (255, 90, 190), (90, 200, 255)     # Verlauf des Rings: Pink -> Himmelblau
LAV, LAV_HI, LAV_D = (178, 168, 235), (225, 220, 255), (120, 110, 190)
HEAD_C, HEAD_HI, HEAD_D = (255, 214, 120), (255, 240, 200), (215, 160, 70)


class PressRenderer:
    def __init__(self, sc):
        self.sc = sc
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        f = (yy / H1)[..., None]
        base = np.array((12, 10, 28), np.float32) * (1 - f) + np.array((26, 10, 36), np.float32) * f
        base += np.exp(-(((xx - CX) / 560) ** 2 + ((yy - CY) / 560) ** 2))[..., None] * np.array((26, 18, 50))
        d2 = (xx - CX) ** 2 + (yy - CY) ** 2
        inside = d2 < R_AR ** 2
        base[inside] = base[inside] * 0.5 + np.array((5, 4, 12))
        # glatter, dicker Ring mit Farbverlauf und Glow
        dist = np.sqrt(d2)
        ang = (np.arctan2(yy - CY, xx - CX) + math.pi) / (2 * math.pi)
        mixv = (0.5 + 0.5 * np.cos(ang * 2 * math.pi))[..., None]
        col = np.array(RING_A, np.float32) * mixv + np.array(RING_B, np.float32) * (1 - mixv)
        band = np.clip(1 - np.abs(dist - (R_AR + 12)) / 12, 0, 1) ** 0.6
        glow = np.exp(-((dist - (R_AR + 12)) / 34) ** 2) * 0.55
        base = base * (1 - band[..., None]) + col * band[..., None] + col * glow[..., None] * 0.6
        hl = np.clip(1 - np.abs(dist - (R_AR + 7)) / 2.5, 0, 1) * (0.5 + 0.5 * np.clip(-(yy - CY) / R_AR, 0, 1))
        base = base + hl[..., None] * 90
        self.base = Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("RGBA")
        m = Image.new("L", self.base.size, 0)
        ImageDraw.Draw(m).ellipse([(CX - R_AR) * k, (CY - R_AR) * k, (CX + R_AR) * k, (CY + R_AR) * k], fill=255)
        self.mask = m
        # Amboss (statisch)
        lay = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        d.polygon([(x * k, y * k) for x, y in sc.anvil], fill=LAV + (255,))
        a0, a1 = sc.anvils
        d.line([(a0[0][0] * k, a0[0][1] * k), (a0[1][0] * k, a0[1][1] * k), (a1[1][0] * k, a1[1][1] * k)],
               fill=LAV_HI + (255,), width=5 * k, joint="curve")
        self.anvil_lay = Image.new("RGBA", lay.size, (0, 0, 0, 0))
        self.anvil_lay.paste(lay, (0, 0), Image.composite(lay.getchannel("A"), Image.new("L", lay.size, 0), m))
        self.splits = [e for e in sc.events if e[1] == "split"]  # nur Teilungen
        self.split_t = [e[0] for e in self.splits]

    def press(self, d, y):
        sc, k = self.sc, SS
        top = CY - R_AR - 60
        d.rounded_rectangle([(CX - 14) * k, top * k, (CX + 14) * k, (y - sc.HEAD_H + 6) * k], radius=6 * k,
                            fill=LAV + (255,))
        d.rectangle([(CX - 9) * k, top * k, (CX - 4) * k, (y - sc.HEAD_H + 6) * k], fill=LAV_HI + (255,))
        x0, x1, y0 = CX - sc.HW, CX + sc.HW, y - sc.HEAD_H
        d.rounded_rectangle([x0 * k, y0 * k, x1 * k, y * k], radius=12 * k, fill=HEAD_C + (255,))
        d.rounded_rectangle([(x0 + 6) * k, (y0 + 5) * k, (x1 - 6) * k, (y0 + 16) * k], radius=5 * k,
                            fill=HEAD_HI + (255,))
        d.rounded_rectangle([(x0 + 4) * k, (y - 14) * k, (x1 - 4) * k, (y - 3) * k], radius=4 * k,
                            fill=HEAD_D + (255,))
        d.rounded_rectangle([(CX - 30) * k, (y0 - 10) * k, (CX + 30) * k, (y0 + 4) * k], radius=6 * k,
                            fill=LAV_D + (255,))

    def frame(self, fi):
        sc, k = self.sc, SS
        t = (fi + 1) / FPS
        Pn, st = sc.frames[fi]
        n = len(Pn)
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        r = sc.R * k
        for i in range(n):
            x, y = Pn[i, 0] * k, Pn[i, 1] * k
            c = ball_col(i * 3)
            d.ellipse([x - r, y - r, x + r, y + r], fill=c + (255,))
            d.ellipse([x - r * 0.55, y - r * 0.6, x - r * 0.05, y - r * 0.15], fill=(255, 255, 255, 150))
        im.alpha_composite(self.anvil_lay)
        lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.press(ImageDraw.Draw(lay, "RGBA"), st[0])
        clip = Image.new("L", im.size, 0)
        ImageDraw.Draw(clip).ellipse([(CX - R_AR - 26) * k, (CY - R_AR - 26) * k, (CX + R_AR + 26) * k,
                                      (CY + R_AR + 26) * k], fill=255)
        im.paste(lay, (0, 0), Image.composite(lay.getchannel("A"), Image.new("L", lay.size, 0), clip))
        d = ImageDraw.Draw(im, "RGBA")
        lo = bisect.bisect_left(self.split_t, t - 0.35)
        hi = bisect.bisect_right(self.split_t, t)
        for t0, _, dd in self.splits[lo:hi][-60:]:
            e = (t - t0) / 0.35
            x, y = dd["x"] * k, dd["y"] * k
            R = (6 + 20 * e) * k
            d.ellipse([x - R, y - R, x + R, y + R], outline=(255, 255, 255, int(200 * (1 - e))), width=2 * k)
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(g, np.float32) - 50, 0, 255) * 0.35
        if sc.full_t is not None and t > sc.full_t:     # Kreis voll: Ring leuchtet sanft auf
            e = t - sc.full_t
            a = min(1.0, e / 0.8) * (0.7 + 0.3 * math.cos(e * 3.0))
            fl = Image.new("L", out.size, 0)
            ImageDraw.Draw(fl).ellipse([CX - R_AR - 12, CY - R_AR - 12, CX + R_AR + 12, CY + R_AR + 12], outline=255,
                                       width=24)
            fl = np.asarray(fl.filter(ImageFilter.GaussianBlur(14)), np.float32)[..., None] / 255
            arr = arr + fl * np.array((255, 225, 140)) * a * 1.2
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(out, "RGBA")
        last = [e[0] for e in self.splits if e[0] <= t]
        pop = 1 + 0.12 * max(0.0, 1 - (t - last[-1]) / 0.15) if last else 1.0
        d.text((CX, CY + R_AR + 84), f"{n:,}", font=_font(F_BOLD, int(54 * pop)), fill=(255, 255, 255, 235),
               anchor="mm")
        return out


# ------------------------------------------------------------------ Klang (wie vorher: weiches Plopp + Presse)

SR = 44100


def build_audio(sc, path):
    from .breaker import plop
    dur = sc.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + SR * 2)

    def put(t, s_, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s_))
        if i < e:
            out[i:e] += s_[: e - i] * vol

    last = -1.0
    for t, kind, d in sc.events:
        if kind == "full":
            for j, f in enumerate((300, 400, 520)):
                put(t + j * 0.08, plop(f, big=True), 0.7)
            continue
        if t - last < 0.05:
            continue
        last = t
        f = 420 + 380 * min(1.0, math.log10(max(1, d["n"])) / 3.0)
        put(t, plop(round(f / 20) * 20), 0.45)
    # Presse setzt auf: tiefes weiches Plopp
    tt, prev = 0.0, None
    end = sc.full_t if sc.full_t is not None else dur
    while tt < end:
        u = (tt / sc.BEAT) % 1.0
        if prev is not None and prev < 0.42 <= u:
            put(tt, plop(160, big=True), 0.35)
        prev = u
        tt += 1 / FPS
    out = out[:n]
    mix_ = out + reverb(out, 0.6) * 0.08
    mix_ *= np.clip((dur - np.arange(n) / SR) / 1.5, 0, 1)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = mix_ / peak * 0.6
    st = np.stack([mix_, mix_], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


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


def render(sc, out_path, jobs=4):
    global _R
    _R = PressRenderer(sc)
    n = sc.n_frames
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
        build_audio(sc, wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav, "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-17:TP=-2:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


SEED_P2 = 1

if __name__ == "__main__":
    import sys
    sc = PressCircle(SEED_P2).run()
    print("voll bei", sc.full_t, "Dauer", sc.n_frames / FPS)
    if "--preview" in sys.argv:
        R = PressRenderer(sc)
        ts = [1.5, 8.0, 20.0, 32.0, sc.full_t - 0.2, sc.n_frames / FPS - 0.1]
        ims = [R.frame(min(sc.n_frames - 1, int(x * FPS))) for x in ts]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/p2_check.png")
        tiktok_ui_mock(ims[3], "vertical").save("output/preview/p2_ui.png")
    else:
        render(sc, "output/m06_press_circle_vertical.mp4")
        print("output/m06_press_circle_vertical.mp4")
