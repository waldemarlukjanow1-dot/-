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

THEME = dict(bg=(6, 6, 6), bg_glow=(10, 14, 12), frame=(225, 255, 235), frame_glow=(120, 255, 150),
             pivot=(10, 30, 15), text=(235, 255, 240))

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


BLOOM = 0.3                     # Leuchten über hellen Stellen (0 = aus, für helle Designs)


class OptRenderer:
    GLOW_LEVELS = (0.8, 0.55, 0.35, 0.2)

    def __init__(self, sim):
        self.sim = sim
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        base = np.zeros((H1 * k, W1 * k, 3), np.float32) + np.array(THEME['bg'], np.float32)
        base += np.exp(-(((xx - CX) / 520) ** 2 + ((yy - CY) / 520) ** 2))[..., None] * np.array(THEME['bg_glow'])
        dist = np.minimum(np.minimum(np.abs(xx - X0), np.abs(xx - X1)), np.minimum(np.abs(yy - Y0), np.abs(yy - Y1)))
        on = (xx > X0 - 20) & (xx < X1 + 20) & (yy > Y0 - 20) & (yy < Y1 + 20)
        dist = np.where(on, dist, 999)
        band = np.clip(1 - dist / 3.0, 0, 1)
        glow = np.exp(-(dist / 14) ** 2) * 0.25
        fr = np.array(THEME['frame'], np.float32)
        base = base * (1 - band[..., None]) + fr * band[..., None] + np.array(THEME['frame_glow']) * glow[..., None] * 0.4
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
        self.splits = [e[0] for e in sim.events if e[1] == "split"]
        self.counts = [(e[0], e[2]["n"]) for e in sim.events if e[1] == "split"]
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
            d.ellipse([(px - 13) * k, (py - 13) * k, (px + 13) * k, (py + 13) * k], fill=THEME['frame'] + (255,))
            d.ellipse([(px - 5) * k, (py - 5) * k, (px + 5) * k, (py + 5) * k], fill=THEME['pivot'] + (255,))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        gg = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(gg, np.float32) - 60, 0, 255) * BLOOM
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
        d.text((CX, Y1 + 120), f"{n:,}", font=_font(F_BOLD, int(54 * pop)), fill=THEME['text'] + (235,), anchor="mm")
        return out


def render(sim, out_path):
    old = A.ArcRenderer
    A.ArcRenderer = OptRenderer
    try:
        A.render(sim, out_path)
    finally:
        A.ArcRenderer = old


if __name__ == "__main__" and "--loop" not in __import__("sys").argv:
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


# ------------------------------------------------------------------ Nahtloser Loop

from .multiply import collide as _collide


class ArcLoop(ArcBox):
    """Wie Arc Squeeze, aber:
      * alle vier Halbkreise drehen gleichzeitig im Uhrzeigersinn mit derselben Geschwindigkeit
      * nahtloser Loop: wenn der Rahmen voll ist, ploppen die Bälle nacheinander weg, bis genau ein Ball übrig ist.
        Dieser Ball (gleiche Farbe wie der Start-Ball) gleitet weich in Position und Richtung des Start-Balls,
        die Bögen stehen genau wie am Anfang -> das letzte Bild geht flüssig ins erste über.
    """
    OM = 2 * math.pi / (204 / 60)      # genau 204 Bilder pro Umdrehung -> Loop liegt exakt auf dem Bildraster
    PH = (0.3, 0.3 + math.pi / 2, 0.3 + math.pi, 0.3 + 3 * math.pi / 2)
    HOLD = 0.5            # kurz voll stehen lassen
    GLIDE = 1.1           # so lange fliegt der letzte Ball geführt in den Startzustand

    def bezier(self, p0, v0, p1, v1, L):
        d0, d1 = v0 / (np.hypot(*v0) + 1e-9), v1 / (np.hypot(*v1) + 1e-9)
        return [p0, p0 + d0 * L / 3, p1 - d1 * L / 3, p1]

    def bez_len(self, b):
        u = np.linspace(0, 1, 200)[:, None]
        pts = (1 - u) ** 3 * b[0] + 3 * (1 - u) ** 2 * u * b[1] + 3 * (1 - u) * u ** 2 * b[2] + u ** 3 * b[3]
        seg = np.hypot(*np.diff(pts, axis=0).T)
        return np.concatenate([[0], np.cumsum(seg)]), u[:, 0], pts

    def choose(self, P, V, P0, V0):
        """Kandidat, dessen Weg in den Startzustand bei normalem Tempo genau GLIDE Sekunden dauert und im Viereck bleibt."""
        want = self.SPEED * self.GLIDE
        best, bscore = None, 1e18
        for i in self.cand:
            for L in np.linspace(0.3, 1.6, 14) * want:
                b = self.bezier(P[i].copy(), V[i].copy(), P0, V0, L)
                cs, uu, pts = self.bez_len(b)
                inside = ((pts[:, 0] > X0 + self.R) & (pts[:, 0] < X1 - self.R) & (pts[:, 1] > Y0 + self.R)
                          & (pts[:, 1] < Y1 - self.R)).all()
                score = abs(cs[-1] - want) + (0 if inside else 1e6)
                if score < bscore:
                    best, bscore = (i, b, cs, uu), score
        i, b, cs, uu = best
        self.bez, self.arc_s, self.arc_u, self.path_len = b, cs, uu, cs[-1]
        return i
    POP = 1.5             # Dauer der Plopp-Kaskade

    def arc_angle(self, k, t):
        return self.PH[k] + self.OM * t

    def omega(self, k, t):
        return self.OM

    def run(self):
        rng = self.rng
        N, r = self.CAP, self.R
        P, V = np.zeros((N, 2)), np.zeros((N, 2))
        gen = np.zeros(N, np.int64)
        last = np.full(N, -9.0)
        born = np.full(N, -9.0)
        nang = np.zeros(N)
        gone = np.full(N, 1e9)            # Zeitpunkt, ab dem ein Ball wegploppt
        self.arc_squeeze = [[] for _ in range(4)]
        P0 = np.array((X1 - 60.0, Y1 - 120.0))
        a = math.radians(-35)
        V0 = np.array((math.cos(a), math.sin(a))) * self.SPEED
        P[0], V[0] = P0, V0
        n = 1
        cell = 2 * r
        x0, y0 = X0 - 10, Y0 - 10
        gw = gh = int((2 * HALF + 20) / cell) + 1
        head = np.full(gw * gh, -1, np.int64)
        nxt = np.full(N, -1, np.int64)
        hit = np.zeros(N, np.int64)
        dt = 1.0 / (FPS * self.SUB)
        period = 2 * math.pi / self.OM
        self.full_t = None
        self.loop_t = None
        surv = -1
        t = 0.0
        self.frames, self.events = [], []
        fi = 0
        while True:
            for _ in range(self.SUB):
                P[:n] += V[:n] * dt
                hit[:n] = 0
                for k, (qx, qy) in enumerate(A.CORNERS):
                    cx_, cy_, a0 = self.arc_geom(k, t)
                    A.arc_hit(P, V, n, r, cx_, cy_, a0, self.OM, hit, qx, qy, ARC_R, k + 1)
                wallnear = ((P[:n, 0] < X0 + r + 2.5) | (P[:n, 0] > X1 - r - 2.5) |
                            (P[:n, 1] < Y0 + r + 2.5) | (P[:n, 1] > Y1 - r - 2.5))
                hit[:n] = np.where(wallnear, hit[:n], 0)
                alive = gone[:n] > t
                for _ in range(2):
                    _collide(P, V, n, r, cell, head, nxt, gw, gh, x0, y0)
                    A.box_walls(P, V, n, r)
                A.keep_speed(V, n, self.SPEED)
                P[:n][~alive] = (-500.0, -500.0)       # weggeploppte Bälle aus dem Spiel nehmen
                if self.full_t is None:
                    for i in np.nonzero(hit[:n])[0]:
                        if n >= N or t - last[i] < self.GAP or not self.allow(n, t):
                            continue
                        g = gen[i] + 1
                        vx, vy = V[i]
                        ang = math.atan2(vy, vx)
                        sp = math.hypot(vx, vy)
                        ox, oy = -math.sin(ang) * r * 0.6, math.cos(ang) * r * 0.6
                        P[n] = (P[i, 0] + ox, P[i, 1] + oy)
                        P[i] = (P[i, 0] - ox, P[i, 1] - oy)
                        V[n] = (math.cos(ang + 0.25) * sp, math.sin(ang + 0.25) * sp)
                        V[i] = (math.cos(ang - 0.25) * sp, math.sin(ang - 0.25) * sp)
                        gen[i] = gen[n] = g
                        last[i] = last[n] = born[i] = born[n] = t
                        dl, dr_, dt_, db = P[i, 0] - X0, X1 - P[i, 0], P[i, 1] - Y0, Y1 - P[i, 1]
                        mn = min(dl, dr_, dt_, db)
                        nang[i] = nang[n] = 0.0 if mn in (dl, dr_) else math.pi / 2
                        self.arc_squeeze[int(hit[i]) - 1].append(t)
                        self.events.append((t, "split", {"n": n + 1, "gen": int(g)}))
                        n += 1
                # Überlebender fliegt mit gleichbleibendem Tempo auf einer sanften Kurve in den Startzustand
                if surv >= 0 and t >= self.pick_t:
                    sdist = min(self.path_len, (t - self.pick_t) * self.path_len / (self.loop_t - self.pick_t))
                    u = float(np.interp(sdist, self.arc_s, self.arc_u))
                    b = self.bez
                    P[surv] = ((1 - u) ** 3 * b[0] + 3 * (1 - u) ** 2 * u * b[1] + 3 * (1 - u) * u ** 2 * b[2]
                               + u ** 3 * b[3])
                    tg = 3 * (1 - u) ** 2 * (b[1] - b[0]) + 6 * (1 - u) * u * (b[2] - b[1]) + 3 * u ** 2 * (b[3] - b[2])
                    V[surv] = tg / (np.hypot(*tg) + 1e-9) * self.path_len / (self.loop_t - self.pick_t)
                t += dt
            fi += 1
            if self.full_t is None and n >= self.N_FULL:
                self.full_t = t
                # Loop-Ende: ganzzahlige Anzahl Bogen-Umdrehungen, damit die Bögen wieder wie am Anfang stehen
                k_ = math.ceil((t + self.HOLD + self.POP + 1.2) / period)
                self.loop_t = k_ * period
                self.cand = [i for i in range(n) if gen[i] % len(RAINBOW) == 0]   # gleiche Farbe wie Start
                cset = set(self.cand)
                order = [i for i in rng.permutation(n) if i not in cset]
                t0 = t + self.HOLD
                for j, i in enumerate(order):
                    gone[i] = t0 + self.POP * (j / len(order)) ** 0.8
                self.pick_t = self.loop_t - self.GLIDE
                self.events.append((t0, "cascade", {}))
            if self.full_t is not None and surv < 0 and t >= self.pick_t:
                surv = self.choose(P, V, P0, V0)
                rest = [i for i in self.cand if i != surv]
                for j, i in enumerate(rest):
                    gone[i] = t + 0.3 * j / max(1, len(rest))
            scale = np.clip((gone[:n] - t) / 0.12, 0, 1).astype(np.float32)
            gshow = gen[:n].astype(np.int8).copy()
            self.frames.append((P[:n].astype(np.float32).copy(), gshow, [self.arc_geom(k, t) for k in range(4)],
                                born[:n].astype(np.float32).copy(), nang[:n].astype(np.float32).copy(), scale))
            if self.loop_t is not None and t >= self.loop_t - 0.5 / FPS:
                break                        # dieses Bild entspricht t=0, danach folgt Bild 0 (t=1/60)
            if t > 120:
                break
        self.n_frames = len(self.frames)
        return self


class LoopRenderer(OptRenderer):
    def frame(self, fi):
        sim, k = self.sim, SS
        t = (fi + 1) / FPS
        Pn, gens, angs, born, nang, scale = sim.frames[fi]
        vis = scale > 0
        n_vis = int(np.sum(scale > 0.5))
        gl = self.glow_level(n_vis)
        sprs, ghs = self.sprites[gl], self.ghost[gl]
        im = self.base.copy()
        trail = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dtr = ImageDraw.Draw(trail, "RGBA")
        for back, al in ((6, 0.10), (3, 0.18)):        # Schweif periodisch -> auch am Loop-Anfang vorhanden
            for (px, py, a) in sim.frames[(fi - back) % sim.n_frames][2]:
                bb = [(px - ARC_R) * k, (py - ARC_R) * k, (px + ARC_R) * k, (py + ARC_R) * k]
                dtr.arc(bb, math.degrees(a), math.degrees(a + SPAN), fill=NEON + (int(255 * al),),
                        width=int(ARC_W * k))
        im.alpha_composite(trail.filter(ImageFilter.GaussianBlur(3 * k)))
        prev = sim.frames[fi - 1][0] if fi > 0 else None
        if prev is not None:
            m = min(len(prev), len(Pn))
            for f in (0.33, 0.66):
                Q = prev[:m] + (Pn[:m] - prev[:m]) * f
                for i in range(m):
                    if scale[i] < 1:
                        continue
                    sp = ghs[int(gens[i]) % len(ghs)]
                    x, y = Q[i]
                    im.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        for i in np.nonzero(vis)[0]:
            x, y = Pn[i]
            sp = sprs[int(gens[i]) % len(sprs)]
            sx = sy = 1.0
            e = (t - born[i]) / 0.16
            if 0 <= e < 1:
                q = (1 - e) ** 2
                sx, sy = 1 + 0.38 * q, 1 - 0.34 * q
                if nang[i] > 0.1:
                    sx, sy = sy, sx
            if scale[i] < 1:                 # wegploppen: kurz aufblähen, dann klein werden
                s_ = scale[i]
                f = (1 + 0.35 * (1 - s_)) * s_ ** 0.5
                sx, sy = sx * f, sy * f
            if sx != 1.0 or sy != 1.0:
                sp = sp.resize((max(2, int(sp.width * sx)), max(2, int(sp.height * sy))), Image.BILINEAR)
            im.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        pulse = []
        for idx in range(4):
            ts = sim.arc_squeeze[idx]
            j = bisect.bisect_right(ts, t) - 1
            pulse.append(max(0.0, 1 - (t - ts[j]) / 0.18) if j >= 0 else 0.0)
        arcs = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.arc_layer(ImageDraw.Draw(arcs, "RGBA"), angs, 1.0, pulse)
        g2 = arcs.resize((W1 // 4, H1 // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5)).resize(im.size)
        im.alpha_composite(g2)
        im.alpha_composite(arcs)
        d = ImageDraw.Draw(im, "RGBA")
        for px, py in A.CORNERS:
            d.ellipse([(px - 13) * k, (py - 13) * k, (px + 13) * k, (py + 13) * k], fill=THEME['frame'] + (255,))
            d.ellipse([(px - 5) * k, (py - 5) * k, (px + 5) * k, (py + 5) * k], fill=THEME['pivot'] + (255,))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        gg = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(gg, np.float32) - 60, 0, 255) * BLOOM
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
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
        d.text((CX, Y1 + 120), f"{n_vis:,}", font=_font(F_BOLD, 54), fill=THEME['text'] + (235,), anchor="mm")
        return out


def build_loop_audio(sim, path):
    """Plopp beim Quetschen; die Kaskade am Ende ploppt leise mit – kein Ton am Übergang."""
    import wave
    from .breaker import plop
    from .marble_render import reverb
    SR = 44100
    dur = sim.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + SR * 2)
    penta = [0, 2, 4, 7, 9, 12, 14, 16]
    last = -1.0
    for t, kind, d in sim.events:
        if kind != "split" or t - last < 0.05:
            continue
        last = t
        f = 440.0 * 2 ** (penta[d["gen"] % len(penta)] / 12)
        s = plop(round(f / 10) * 10)
        i = int(t * SR)
        e = min(len(out), i + len(s))
        out[i:e] += s[: e - i] * 0.45
    out = out[:n]
    # Hall vom Ende in den Anfang falten -> auch der Ton läuft nahtlos weiter
    tail = reverb(np.concatenate([out, np.zeros(SR)]), 0.6) * 0.08
    mix_ = out + tail[:n]
    mix_[:SR] += tail[n:n + SR]
    peak = np.max(np.abs(mix_)) or 1
    mix_ = mix_ / peak * 0.6
    st = np.stack([mix_, mix_], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def render_loop(sim, out_path):
    old_r, old_a = A.ArcRenderer, A.build_audio
    A.ArcRenderer, A.build_audio = LoopRenderer, build_loop_audio
    try:
        A.render(sim, out_path)
    finally:
        A.ArcRenderer, A.build_audio = old_r, old_a


if __name__ == "__main__" and "--loop" in __import__("sys").argv:
    sim = ArcLoop(A.SEED_ARC).run()
    print("voll", sim.full_t, "Loop", sim.loop_t, "Bilder", sim.n_frames)
    render_loop(sim, "output/a06_arc_squeeze_loop_vertical.mp4")
    print("output/a06_arc_squeeze_loop_vertical.mp4")
