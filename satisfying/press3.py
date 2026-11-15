"""Katapult + Presse: unten schleudert ein Katapult-Kolben die liegenden Bälle senkrecht nach oben. Oben schnappen
zwei Pressbacken zusammen – wer dazwischen gerät, wird zerquetscht und verdoppelt sich, die zwei fallen nach unten.
Anfangs verdoppelt sich praktisch jeder Wurf (schnell spannend), später füllt sich der Kreis bis zum Rand.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tiktok.render import tiktok_ui_mock
from .multiply import CX, CY, R_AR, FPS, W1, H1, SS, Scene, poly_body, ball_col, _font, F_BOLD
from . import press2 as P2

PAD_HW = 150.0                      # halbe Breite des Katapult-Kolbens
PAD_REST = CY + R_AR - 62           # Oberkante in Ruhe
PAD_LIFT = 118.0
JAW_Y0, JAW_Y1 = CY - R_AR - 10, CY - 230  # Pressbacken ganz oben am Rand
GAP_OPEN, GAP_SHUT = 420.0, 14.0


class CatapultPress(Scene):
    R = 11.0
    CAP = 3000
    DUR = 80.0
    SUB = 8
    BEAT = 1.9
    GAP = 1.0                        # jeder Ball höchstens einmal pro Pressvorgang
    N_FULL = 2850
    OUTRO = 4.0

    def __init__(self, seed, n0=2):
        super().__init__(seed, n0)
        h = PAD_HW + 2
        self.slopes = [np.array([(CX - h, PAD_REST), (CX - R_AR - 40, PAD_REST - 190), (CX - R_AR - 40, CY + R_AR + 80),
                                 (CX - h, CY + R_AR + 80)], np.float64),
                       np.array([(CX + h, PAD_REST), (CX + h, CY + R_AR + 80), (CX + R_AR + 40, CY + R_AR + 80),
                                 (CX + R_AR + 40, PAD_REST - 190)], np.float64)]
        self.full_t = None

    def start(self, P, V):
        for k in range(self.n0):
            P[k] = (CX - 30 + 60 * k, PAD_REST - 60 - 30 * k)
            V[k] = (0.0, 0.0)

    # ---------------------------------------------------------------- Bewegungen
    def u(self, t):
        return (t / self.BEAT) % 1.0

    def pad_y(self, t):
        if self.full_t is not None and t >= self.full_t:
            return PAD_REST
        u = self.u(t) * self.BEAT          # Sekunden im Takt
        if u < 0.12:                        # Schuss: schnell hoch (Halbsinus-Geschwindigkeit)
            f = 0.5 - 0.5 * math.cos(math.pi * u / 0.12)
        elif u < 0.35:
            f = 1.0
        elif u < 0.9:
            v = (u - 0.35) / 0.55
            f = 1 - v * v * (3 - 2 * v)
        else:
            f = 0.0
        return PAD_REST - PAD_LIFT * f

    def gap(self, t):
        if self.full_t is not None and t >= self.full_t:
            e = min(1.0, (t - self.full_t) / 0.8)   # Ende: Backen schließen sich zum Deckel, Kreis ist voll
            return self._gap_full + (GAP_SHUT - self._gap_full) * e * e * (3 - 2 * e)
        u = self.u(t) * self.BEAT
        if u < 0.74:
            f = 0.0
        elif u < 0.98:                      # zuschnappen, wenn die Bälle oben am Scheitelpunkt schweben
            v = (u - 0.74) / 0.24
            f = v * v
        elif u < 1.12:
            f = 1.0
        elif u < 1.45:
            v = (u - 1.12) / 0.33
            f = 1 - v * v * (3 - 2 * v)
        else:
            f = 0.0
        return GAP_OPEN + (GAP_SHUT - GAP_OPEN) * f

    def vel(self, fn, t):
        e = 1e-3
        return (fn(t + e) - fn(t - e)) / (2 * e)

    def jaws(self, g):
        """Backen mit schräger Oberseite: Bälle rutschen oben in den Spalt statt liegen zu bleiben."""
        L = np.array([(CX - R_AR - 60, JAW_Y0 - 150), (CX - g / 2, JAW_Y0), (CX - g / 2, JAW_Y1),
                      (CX - R_AR - 60, JAW_Y1)], np.float64)
        Rr = np.array([(CX + g / 2, JAW_Y0), (CX + R_AR + 60, JAW_Y0 - 150), (CX + R_AR + 60, JAW_Y1),
                       (CX + g / 2, JAW_Y1)], np.float64)
        return L, Rr

    def pad(self, y):
        return np.array([(CX - PAD_HW, y), (CX + PAD_HW, y), (CX + PAD_HW, y + 420), (CX - PAD_HW, y + 420)],
                        np.float64)

    SCRIPTED_LAUNCH = False
    CUP = 2000.0                     # die ganze Säule über dem Kolben wird geschleudert

    def launch(self, P, V, n, t):
        """Katapult: einmal pro Takt bekommt die Ladung über dem Kolben genau den Schwung bis in die Presse."""
        stroke = int(t / self.BEAT)
        if self.full_t is not None or stroke == getattr(self, "_launched", -1) or self.u(t) * self.BEAT < 0.05:
            return
        self._launched = stroke
        py = self.pad_y(t)
        apex = (JAW_Y0 + JAW_Y1) / 2
        m = (np.abs(P[:n, 0] - CX) < PAD_HW + self.R) & (P[:n, 1] > max(py - self.CUP, JAW_Y1 + 2 * self.R)) & (P[:n, 1] < py + self.R)
        idx = np.nonzero(m)[0]
        h = np.maximum(P[idx, 1] - apex, 50.0)
        V[idx, 1] = -np.sqrt(2 * 1500.0 * h) * self.rng.uniform(0.98, 1.02, len(idx))
        V[idx, 0] = (CX - P[idx, 0]) * 0.35 + self.rng.normal(0, 25, len(idx))
        if len(idx):
            self.events.append((t, "launch", {"n": int(len(idx))}))

    def machine(self, P, V, n, t, touch, impact):
        if self.SCRIPTED_LAUNCH:   # alt: Geschwindigkeit wurde gesetzt – jetzt rein physikalisch über den Kolben
            self.launch(P, V, n, t)
        poly_body(P, V, n, self.R, self.pad(self.pad_y(t)), 0.0, self.vel(self.pad_y, t), 0.0, 0.0, 0.0, 0.05,
                  touch, 2, impact, 1e9)
        for q in self.slopes:
            poly_body(P, V, n, self.R, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, touch, 4, impact, 1e9)
        g = self.gap(t)
        vg = self.vel(self.gap, t)
        L, Rr = self.jaws(g)
        poly_body(P, V, n, self.R, L, -vg / 2, 0.0, 0.0, 0.0, 0.0, 0.2, touch, 1, impact, 1e9)
        poly_body(P, V, n, self.R, Rr, vg / 2, 0.0, 0.0, 0.0, 0.0, 0.2, touch, 8, impact, 1e9)

    def crushed(self, P, V, n, t, touch, impact):
        g = self.gap(t)
        if self.full_t is not None or g > 6 * self.R or self.vel(self.gap, t) >= 0:
            return np.zeros(0, np.int64)
        m = (np.abs(P[:n, 0] - CX) < g / 2 + self.R * 0.6) & (P[:n, 1] > JAW_Y0) & (P[:n, 1] < JAW_Y1)
        return np.nonzero(m)[0]

    def eject(self, P, V, i, j, t):
        """Zerquetscht -> zwei Bälle, die unten aus der Presse herausfallen."""
        rng = self.rng
        x = P[i, 0]
        for k, s in ((i, -1), (j, 1)):
            P[k, 0] = x + s * (self.R + 0.5)
            P[k, 1] = JAW_Y1 + self.R + 1 + rng.uniform(0, 6)
            V[k] = (s * rng.uniform(60, 180), rng.uniform(50, 160))

    def state(self, t):
        return (self.pad_y(t), self.gap(t))

    def frame_hook(self, P, n, t):
        """Voll = die Bälle reichen dauerhaft (1,5 s) bis ganz oben an den Rand."""
        top = int(np.sum(P[:n, 1] < CY - R_AR + 45))
        if top >= 10:
            self._top_since = getattr(self, "_top_since", None) or t
        else:
            self._top_since = None
        if self.full_t is None and (n >= self.N_FULL or (self._top_since and t - self._top_since > 1.5)):
            self._gap_full = self.gap(t)
            self.full_t = t
            self.events.append((t, "full", {}))
        return self.full_t is not None and t > self.full_t + self.OUTRO


# ------------------------------------------------------------------ Darstellung

LAV, LAV_HI, LAV_D = P2.LAV, P2.LAV_HI, P2.LAV_D
HEAD_C, HEAD_HI, HEAD_D = P2.HEAD_C, P2.HEAD_HI, P2.HEAD_D


class CatapultRenderer(P2.PressRenderer):
    def __init__(self, sc):
        self.sc = sc
        k = SS
        # Basis (Hintergrund + Ring) vom Press-Circle übernehmen, ohne dessen Amboss
        sc.anvil = np.zeros((3, 2))
        sc.anvils = [np.zeros((2, 2)), np.zeros((3, 2))]
        P2.PressRenderer.__init__(self, sc)
        lay = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for q in sc.slopes:
            d.polygon([(x * k, y * k) for x, y in q], fill=LAV + (255,))
            d.line([(q[0][0] * k, q[0][1] * k), (q[1][0] * k, q[1][1] * k)] if q[0][0] < CX else
                   [(q[0][0] * k, q[0][1] * k), (q[3][0] * k, q[3][1] * k)], fill=LAV_HI + (255,), width=5 * k)
        self.anvil_lay = Image.new("RGBA", lay.size, (0, 0, 0, 0))
        self.anvil_lay.paste(lay, (0, 0), Image.composite(lay.getchannel("A"), Image.new("L", lay.size, 0), self.mask))

    def machines(self, d, st):
        sc, k = self.sc, SS
        py, g = st
        # Katapult-Kolben
        d.rounded_rectangle([(CX - PAD_HW) * k, py * k, (CX + PAD_HW) * k, (py + 420) * k], radius=10 * k,
                            fill=HEAD_C + (255,))
        d.rounded_rectangle([(CX - PAD_HW + 6) * k, (py + 5) * k, (CX + PAD_HW - 6) * k, (py + 15) * k], radius=4 * k,
                            fill=HEAD_HI + (255,))
        for q in range(3):
            yy = py + 34 + q * 22
            d.line([(CX - PAD_HW + 14) * k, yy * k, (CX + PAD_HW - 14) * k, yy * k], fill=HEAD_D + (255,), width=4 * k)
        # Pressbacken
        L, Rr = sc.jaws(g)
        for poly, s in ((L, 1), (Rr, -1)):
            d.polygon([(x * k, y * k) for x, y in poly], fill=LAV + (255,))
            top = (poly[0], poly[1]) if s == 1 else (poly[0], poly[1])
            d.line([(top[0][0] * k, top[0][1] * k), (top[1][0] * k, top[1][1] * k)], fill=LAV_HI + (255,),
                   width=6 * k)
            x_in = poly[1][0] if s == 1 else poly[0][0]
            xf = x_in - s * 16       # goldene Pressfläche innen
            d.rounded_rectangle([min(xf, x_in) * k, (JAW_Y0 + 2) * k, max(xf, x_in) * k, (JAW_Y1 - 2) * k],
                                radius=6 * k, fill=HEAD_C + (255,))

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
        self.machines(ImageDraw.Draw(lay, "RGBA"), st)
        im.paste(lay, (0, 0), Image.composite(lay.getchannel("A"), Image.new("L", lay.size, 0), self.mask))
        d = ImageDraw.Draw(im, "RGBA")
        import bisect
        lo = bisect.bisect_left(self.split_t, t - 0.35)
        hi = bisect.bisect_right(self.split_t, t)
        for t0, _, dd in self.splits[lo:hi][-60:]:
            e = (t - t0) / 0.35
            x, y = dd["x"] * k, dd["y"] * k
            pass   # keine Ring-Effekte mehr (Wunsch)
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(g, np.float32) - 50, 0, 255) * 0.35
        if sc.full_t is not None and t > sc.full_t:
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


def build_audio(sc, path):
    """Wie vorher: weiches Plopp beim Zerquetschen, tiefes Plopp beim Pressen, dumpfes Plopp beim Katapult."""
    import wave
    from .breaker import plop
    from .marble_render import reverb
    SR = 44100
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
        if kind == "full":   # kein Schluss-Sound (Wunsch)
            continue
        if kind != "split":
            continue
        if t - last < 0.05:
            continue
        last = t
        f = 420 + 380 * min(1.0, math.log10(max(1, d["n"])) / 3.0)
        put(t, plop(round(f / 20) * 20), 0.45)
    end = sc.full_t if sc.full_t is not None else dur
    b = 0
    while b * sc.BEAT < end:
        put(b * sc.BEAT, plop(120, big=True), 0.3)              # Katapult
        put(b * sc.BEAT + 0.98, plop(160, big=True), 0.35)      # Presse schnappt zu
        b += 1
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


def render(sc, out_path, jobs=4):
    P2._R = CatapultRenderer(sc)
    old = P2.build_audio
    P2.build_audio = build_audio
    try:
        import os, subprocess, tempfile
        from multiprocessing import Pool
        n = sc.n_frames
        with tempfile.TemporaryDirectory() as tmp:
            bounds = [n * j // jobs for j in range(jobs + 1)]
            parts = [(bounds[j], bounds[j + 1], os.path.join(tmp, f"p{j}.mp4")) for j in range(jobs)]
            with Pool(jobs) as p:
                p.map(P2._chunk, parts)
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w") as f:
                for _, _, pth in parts:
                    f.write(f"file '{pth}'\n")
            wav = os.path.join(tmp, "a.wav")
            build_audio(sc, wav)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-af",
                            "loudnorm=I=-17:TP=-2:LRA=11", "-movflags", "+faststart", out_path], check=True)
    finally:
        P2.build_audio = old


SEED_CP = 1

if __name__ == "__main__":
    import sys
    sc = CatapultPress(SEED_CP).run()
    print("voll bei", sc.full_t, "Dauer", sc.n_frames / FPS,
          [len(sc.frames[min(sc.n_frames - 1, k * FPS)][0]) for k in range(0, 80, 5)])
    if "--preview" in sys.argv:
        R = CatapultRenderer(sc)
        ts = [0.5, 0.95, 6.0, 15.0, 30.0, sc.n_frames / FPS - 0.1]
        ims = [R.frame(min(sc.n_frames - 1, int(x * FPS))) for x in ts]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/p3_check.png")
        tiktok_ui_mock(ims[4], "vertical").save("output/preview/p3_ui.png")
    else:
        render(sc, "output/m07_catapult_press_vertical.mp4")
        print("output/m07_catapult_press_vertical.mp4")
