"""Arc Squeeze – optisch optimierte Version (alle Vorschläge):
  1. fast schwarzer Hintergrund mit leichtem Lichtschein, schlichter weiß-grünlicher Rahmen
  2. Bälle: Glossy-3D mit dunklem Rand und kleinem Schatten; Glow nimmt ab, je voller es wird;
     Farben je Generation in Regenbogen-Reihenfolge (Rot -> Orange -> Gelb -> Grün -> Blau -> Lila -> Pink)
  3. Bögen: kurzer grüner Nachzieh-Schweif, kurzes Aufleuchten des Bogens beim Quetschen (kein Ring)
  4. Squash: frisch geteilte Bälle sind kurz plattgedrückt (zur Wand hin) und federn zurück;
     Mikro-Zoom (~2 %) bei 100 / 250 / 500 Bällen
  5. Ende: Bögen blenden sanft aus, der Rahmen leuchtet einmal weich auf (ohne Ton)
"""
import bisect
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import arcbox as A
from .arcbox import (CX, CY, X0, X1, Y0, Y1, HALF, ARC_R, ARC_W, SPAN, FPS, W1, H1, SS, ArcBox, NEON, NEON_HI,
                     NEON_D, NEON_DD)
from .multiply import _font, F_BOLD
from tiktok.render import tiktok_ui_mock

RAINBOW = [(255, 40, 60), (255, 130, 0), (255, 230, 0), (60, 255, 70), (0, 150, 255), (150, 50, 255),
           (255, 40, 190), (0, 240, 255)]


def sprite(r, col, glow_amt):
    n = int(r * 5)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2 + 0.5
    d = np.hypot(xx, yy)
    core = np.clip(r - d + 0.7, 0, 1)
    glow = np.clip(1 - d / (r * 2.6), 0, 1) ** 2.0 * glow_amt
    c = np.array(col, np.float32)
    light = np.clip(1.0 - ((xx + r * 0.35) ** 2 + (yy + r * 0.45) ** 2) / (2.2 * r * r), 0, 1)
    rgb = c * (0.7 + 0.45 * light[..., None])
    rim = np.clip((d - r * 0.78) / (r * 0.22), 0, 1)            # dunkler Rand -> Bälle bleiben einzeln sichtbar
    rgb = rgb * (1 - 0.5 * rim[..., None])
    hard = np.exp(-(((xx + r * 0.36) / (r * 0.2)) ** 2 + ((yy + r * 0.42) / (r * 0.16)) ** 2))
    soft = np.exp(-(((xx + r * 0.25) / (r * 0.45)) ** 2 + ((yy + r * 0.3) / (r * 0.32)) ** 2))
    rgb = rgb + (hard * 0.95 + soft * 0.25)[..., None] * (255 - rgb)
    gl = np.zeros_like(rgb)
    gl[:] = c
    rgb = rgb * core[..., None] + gl * (1 - core[..., None])
    # kleiner Schatten unten rechts
    sh = np.clip(r - np.hypot(xx - r * 0.18, yy - r * 0.28) + 0.7, 0, 1) * (1 - core) * 0.55
    a = np.maximum(np.maximum(glow, core), sh)
    rgb = rgb * (1 - (sh * (1 - core))[..., None])
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA")


class OptRenderer:
    GLOW_LEVELS = (0.8, 0.55, 0.35, 0.2)

    def __init__(self, sim):
        self.sim = sim
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        base = np.full((H1 * k, W1 * k, 3), 6.0, np.float32)
        base += np.exp(-(((xx - CX) / 520) ** 2 + ((yy - CY) / 520) ** 2))[..., None] * np.array((10, 14, 12))
        dist = np.minimum(np.minimum(np.abs(xx - X0), np.abs(xx - X1)), np.minimum(np.abs(yy - Y0), np.abs(yy - Y1)))
        on = (xx > X0 - 20) & (xx < X1 + 20) & (yy > Y0 - 20) & (yy < Y1 + 20)
        dist = np.where(on, dist, 999)
        band = np.clip(1 - dist / 3.0, 0, 1)
        glow = np.exp(-(dist / 14) ** 2) * 0.25
        fr = np.array((225, 255, 235), np.float32)
        base = base * (1 - band[..., None]) + fr * band[..., None] + np.array((120, 255, 150)) * glow[..., None] * 0.4
        self.base = Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("RGBA")
        self.frame_mask = Image.fromarray((band * 255).astype(np.uint8))
        self.sprites = {g: [sprite(sim.R * k, c, g) for c in RAINBOW] for g in self.GLOW_LEVELS}
        self.ghost = {}
        for g, sps in self.sprites.items():
            out = []
            for sp in sps:
                a = np.asarray(sp).copy()
                a[..., 3] = (a[..., 3] * 0.26).astype(np.uint8)
                out.append(Image.fromarray(a, "RGBA"))
            self.ghost[g] = out
        self.splits = [e[0] for e in sim.events]
        self.counts = [(e[0], e[2]["n"]) for e in sim.events]
        self.miles = []
        for m in (100, 250, 500):
            hit = next((t for t, nn in self.counts if nn >= m), None)
            if hit is not None:
                self.miles.append(hit)

    def glow_level(self, n):
        return self.GLOW_LEVELS[0 if n < 80 else 1 if n < 300 else 2 if n < 600 else 3]

    def arc_layer(self, d, angs, alpha, pulse):
        k = SS
        for idx, (px, py, a) in enumerate(angs):
            p = pulse[idx]
            col = tuple(int(c + (255 - c) * 0.55 * p) for c in NEON)
            hi = tuple(int(c + (255 - c) * 0.6 * p) for c in NEON_HI)

            def bb(dr):
                return [(px - ARC_R - dr) * k, (py - ARC_R - dr) * k, (px + ARC_R + dr) * k, (py + ARC_R + dr) * k]
            a0, a1 = math.degrees(a), math.degrees(a + SPAN)
            A_ = int(255 * alpha)
            d.arc(bb(0), a0, a1, fill=NEON_D + (A_,), width=int((ARC_W + 4) * k))
            d.arc(bb(0), a0, a1, fill=col + (A_,), width=int(ARC_W * k))
            d.arc(bb(-3), a0 + 0.5, a1 - 0.5, fill=hi + (A_,), width=int(4 * k))
            d.arc(bb(5), a0 + 0.5, a1 - 0.5, fill=NEON_DD + (A_,), width=int(2 * k))
            for q in range(1, 18):
                e = a + SPAN * q / 18
                c, s_ = math.cos(e), math.sin(e)
                r0, r1 = ARC_R - ARC_W / 2 + 2, ARC_R + ARC_W / 2 - 2
                d.line([((px + c * r0) * k, (py + s_ * r0) * k), ((px + c * r1) * k, (py + s_ * r1) * k)],
                       fill=NEON_D + (int(200 * alpha),), width=int(2 * k))
            for e in (a, a + SPAN):
                ex, ey = px + math.cos(e) * ARC_R, py + math.sin(e) * ARC_R
                rr = ARC_W / 2 + 1
                d.ellipse([(ex - rr) * k, (ey - rr) * k, (ex + rr) * k, (ey + rr) * k], fill=col + (A_,),
                          outline=NEON_D + (A_,), width=k)

    def frame(self, fi):
        sim, k = self.sim, SS
        t = (fi + 1) / FPS
        Pn, gens, angs, born, nang = sim.frames[fi]
        n = len(Pn)
        gl = self.glow_level(n)
        sprs, ghs = self.sprites[gl], self.ghost[gl]
        end = 0.0 if sim.full_t is None or t < sim.full_t else min(1.0, (t - sim.full_t) / 1.5)
        im = self.base.copy()
        # 3) Nachzieh-Schweif der Bögen
        trail = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dtr = ImageDraw.Draw(trail, "RGBA")
        for back, al in ((6, 0.10), (3, 0.18)):
            if fi - back >= 0:
                for (px, py, a) in sim.frames[fi - back][2]:
                    bb = [(px - ARC_R) * k, (py - ARC_R) * k, (px + ARC_R) * k, (py + ARC_R) * k]
                    dtr.arc(bb, math.degrees(a), math.degrees(a + SPAN), fill=NEON + (int(255 * al * (1 - end)),),
                            width=int(ARC_W * k))
        im.alpha_composite(trail.filter(ImageFilter.GaussianBlur(3 * k)))
        # 2) Bälle mit Bewegungsunschärfe
        prev = sim.frames[fi - 1][0] if fi > 0 else Pn
        m = min(len(prev), n)
        for f in (0.33, 0.66):
            Q = prev[:m] + (Pn[:m] - prev[:m]) * f
            for (x, y), g in zip(Q, gens[:m]):
                sp = ghs[int(g) % len(ghs)]
                im.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        for i in range(n):
            x, y = Pn[i]
            sp = sprs[int(gens[i]) % len(sprs)]
            e = (t - born[i]) / 0.16
            if 0 <= e < 1:              # 4) Squash: zur Wand hin plattgedrückt, federt zurück
                q = (1 - e) ** 2
                sx, sy = 1 + 0.38 * q, 1 - 0.34 * q
                if nang[i] > 0.1:       # Wand oben/unten -> vertikal platt
                    sx, sy = sy, sx
                sp = sp.resize((max(2, int(sp.width * sx)), max(2, int(sp.height * sy))), Image.BILINEAR)
            im.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        # 3) Bögen mit kurzem Aufleuchten beim Quetschen
        pulse = []
        for idx in range(4):
            ts = sim.arc_squeeze[idx]
            j = bisect.bisect_right(ts, t) - 1
            pulse.append(max(0.0, 1 - (t - ts[j]) / 0.18) if j >= 0 else 0.0)
        arcs = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.arc_layer(ImageDraw.Draw(arcs, "RGBA"), angs, 1 - end, pulse)
        g2 = arcs.resize((W1 // 4, H1 // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5)).resize(im.size)
        im.alpha_composite(g2)
        im.alpha_composite(arcs)
        for px, py in A.CORNERS:
            d = ImageDraw.Draw(im, "RGBA")
            d.ellipse([(px - 13) * k, (py - 13) * k, (px + 13) * k, (py + 13) * k], fill=(225, 255, 235, 255))
            d.ellipse([(px - 5) * k, (py - 5) * k, (px + 5) * k, (py + 5) * k], fill=(10, 30, 15, 255))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        gg = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(gg, np.float32) - 60, 0, 255) * 0.3
        # 5) Ende: Rahmen leuchtet einmal weich auf
        if end > 0:
            e = t - sim.full_t
            a = math.sin(min(1.0, e / 2.0) * math.pi) * 1.0
            fm = Image.fromarray(np.asarray(self.frame_mask)).reduce(SS).filter(ImageFilter.GaussianBlur(10))
            fmv = np.asarray(fm, np.float32)[..., None] / 255
            arr = arr + fmv * np.array((150, 255, 170)) * a * 2.2
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        # 4) Mikro-Zoom bei Meilensteinen
        z = 0.0
        for mt in self.miles:
            e = t - mt
            if 0 <= e < 0.5:
                z = max(z, math.sin(e / 0.5 * math.pi) * 0.02)
        if z > 0:
            w, h = int(W1 * (1 + z)), int(H1 * (1 + z))
            big = out.resize((w, h), Image.BICUBIC)
            ox, oy = int(CX * (1 + z) - CX), int(CY * (1 + z) - CY)
            out = big.crop((ox, oy, ox + W1, oy + H1))
        d = ImageDraw.Draw(out, "RGBA")
        last = [s for s in self.splits if s <= t]
        pop = 1 + 0.12 * max(0.0, 1 - (t - last[-1]) / 0.15) if last else 1.0
        d.text((CX, Y1 + 120), f"{n:,}", font=_font(F_BOLD, int(54 * pop)), fill=(235, 255, 240, 235), anchor="mm")
        return out


def render(sim, out_path):
    old = A.ArcRenderer
    A.ArcRenderer = OptRenderer
    try:
        A.render(sim, out_path)
    finally:
        A.ArcRenderer = old


if __name__ == "__main__":
    import sys
    sim = ArcBox(A.SEED_ARC).run()
    print("voll", sim.full_t, "Dauer", sim.n_frames / FPS)
    if "--preview" in sys.argv:
        R = OptRenderer(sim)
        mt = R.miles[1] if len(R.miles) > 1 else 20
        ts = [1.0, 8.0, mt + 0.25, 28.0, sim.full_t + 0.9, sim.n_frames / FPS - 0.1]
        ims = [R.frame(min(sim.n_frames - 1, int(x * FPS))) for x in ts]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/arcopt_check.png")
        ims[3].crop((100, 450, 960, 1300)).save("output/preview/arcopt_zoom.png")
    else:
        render(sim, "output/a05_arc_squeeze_optimized_vertical.mp4")
        print("output/a05_arc_squeeze_optimized_vertical.mp4")
