"""Miau-Stampfwerk (a26): neues Prinzip mit schweren, sich bewegenden Stangen.
  * drei schwere Stahl-Stempel (Stangen mit Pressplatte) fahren nacheinander wie eine Welle langsam nach unten,
    pressen kurz und fahren wieder hoch
  * ein Ball, der unter einer Platte gegen den Boden gedrückt wird und nicht mehr ausweichen kann, wird
    zerquetscht und teilt sich genau dort in 2 (nächste Farbe) – neue Bälle entstehen nur so
  * Bälle bewegen sich nur durch Physik (konstantes Tempo, Stöße), keine Schrift außer der Ballzahl
  * Ton: Deep Meow bei JEDER Quetschung (überlappend), sonst nichts
  * ruhiger Start mit 1 Ball, erste Quetschung nach ein paar Sekunden, ~50 s, nahtloser Loop
"""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import arcbox as A
from . import arcbox6 as F
from .arcbox import CX, CY, HALF, X0, X1, Y0, Y1, FPS, W1, H1, SS
from .multiply import collide as _collide, _font, F_BOLD
from .jelly import capsule, jelly_sprite


class Stampfwerk:
    R = 15.0
    SUB = 6
    SPEED = 420.0
    CAP = 1400
    N_FULL = 900
    T_FULL = 46.0
    GAP = 0.22
    CRUSH = 0.28
    SPLIT = 2
    T = 3.0                      # Takt der Stempel (180 Bilder -> Loop exakt auf dem Bildraster)
    PW = 108.0                   # halbe Plattenbreite
    PT = 30.0                    # halbe Plattendicke (schwer)
    XS = (CX - 250.0, CX, CX + 250.0)
    PHASES = (0.0, 1 / 3, 2 / 3)
    HOLD, POP, GLIDE = 0.6, 1.5, 1.2

    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)

    # ---------------- Maschine
    def ypos(self, t, ph):
        u = (t / self.T + ph) % 1.0
        top = Y0 + self.PT + self.R * 3.2
        bot = Y1 - self.PT - self.R * 1.2
        if u < 0.42:                          # schwer und langsam nach unten (beschleunigt)
            e = u / 0.42
            s = e ** 2.2
        elif u < 0.52:                        # pressen
            s = 1.0
        else:                                 # gemächlich wieder hoch
            e = (u - 0.52) / 0.48
            s = 1 - e * e * (3 - 2 * e)
        return top + (bot - top) * s

    def parts(self, t):
        out = []
        for x, ph in zip(self.XS, self.PHASES):
            y = self.ypos(t, ph)
            vy = (self.ypos(t + 1e-3, ph) - self.ypos(t - 1e-3, ph)) / 2e-3
            out.append((x - self.PW, y, x + self.PW, y, self.PT, x, y, 0.0, 0.0, vy))
        return out

    def allow(self, n, t):
        T = self.T_FULL
        return n < 1 + (self.N_FULL - 1) * (min(t, T) / T) ** 2.1 + 2

    # ---------------- Simulation
    def run(self, until=None):
        N, r = self.CAP, self.R
        P, V = np.zeros((N, 2)), np.zeros((N, 2))
        gen = np.zeros(N, np.int64)
        last = np.full(N, -9.0)
        born = np.full(N, -9.0)
        gone = np.full(N, 1e9)
        SP, SV = self.STARTS
        m0 = len(SP)
        P[:m0], V[:m0] = SP, SV
        n = m0
        cell = 2 * r
        x0, y0 = X0 - 10, Y0 - 10
        gw = gh = int((2 * HALF + 20) / cell) + 1
        head = np.full(gw * gh, -1, np.int64)
        nxt = np.full(N, -1, np.int64)
        pen = np.zeros(N)
        nxa, nya = np.zeros(N), np.zeros(N)
        dt = 1.0 / (FPS * self.SUB)
        self.frames, self.events, self.parts_log = [], [], []
        self.full_t = self.loop_t = None
        self.glides, self.surv = {}, []
        t = 0.0
        while True:
            for _ in range(self.SUB):
                pts = self.parts(t)
                alive = gone[:n] > t
                P[:n] += V[:n] * dt
                for q in pts:
                    capsule(P, V, n, r, *q, 1.0, pen, nxa, nya, 0)
                for _ in range(2):
                    _collide(P, V, n, r, cell, head, nxt, gw, gh, x0, y0)
                    A.box_walls(P, V, n, r)
                pen[:n] = 0
                for q in pts:
                    capsule(P, V, n, r, *q, 1.0, pen, nxa, nya, 1)
                A.box_walls(P, V, n, r)
                A.keep_speed(V, n, self.SPEED)
                P[:n][~alive] = (-500.0, -500.0)
                if self.full_t is None:
                    under = np.zeros(n, bool)
                    for (ax, ay, bx, by, *_r) in pts:
                        under |= (P[:n, 0] > ax - 5) & (P[:n, 0] < bx + 5) & (P[:n, 1] > ay)
                    c = np.nonzero((pen[:n] > self.CRUSH * r) & under & (nya[:n] > 0.3))[0]
                    for i in c[np.argsort(-pen[c])]:
                        if n + self.SPLIT - 1 >= N or t - last[i] < self.GAP or not self.allow(n, t):
                            continue
                        g = gen[i] + 1
                        base, vb = P[i].copy(), V[i].copy()
                        tx = 1.0
                        offs = (-0.5, 0.5) if self.SPLIT == 2 else (0.0, 1.0, -1.0)
                        for j, o in enumerate(offs):
                            idx = i if j == 0 else n + j - 1
                            P[idx] = (base[0] + tx * o * r * 1.6, base[1])
                            V[idx] = (math.copysign(1, o if o else vb[0]) * self.SPEED * 0.8, -self.SPEED * 0.6)
                            gen[idx] = g
                            last[idx] = born[idx] = t
                        self.events.append((t, "split", {"n": n + self.SPLIT - 1, "gen": int(g)}))
                        n += self.SPLIT - 1
                    A.box_walls(P, V, n, r)
                for i, (b, cs, uu, L) in self.glides.items():
                    Tg = self.loop_t - self.pick_t
                    sd = min(L, (t - self.pick_t) * L / Tg)
                    u = float(np.interp(sd, cs, uu))
                    P[i] = ((1 - u) ** 3 * b[0] + 3 * (1 - u) ** 2 * u * b[1] + 3 * (1 - u) * u ** 2 * b[2]
                            + u ** 3 * b[3])
                    tg = 3 * (1 - u) ** 2 * (b[1] - b[0]) + 6 * (1 - u) * u * (b[2] - b[1]) + 3 * u ** 2 * (b[3] - b[2])
                    V[i] = tg / (np.hypot(*tg) + 1e-9) * L / Tg
                t += dt
            if until is not None and t >= until:
                self.n_frames = len(self.frames)
                return self
            na = int(np.count_nonzero(gone[:n] > t))
            if self.full_t is None and na >= self.N_FULL:
                self.full_t = t
                tmin = self.HOLD + self.POP + self.GLIDE + 0.4
                self.loop_t = math.ceil((t + tmin) / self.T) * self.T
                self.pick_t = self.loop_t - self.GLIDE
                self.cand = [i for i in range(n) if gen[i] == 0 or gen[i] % len(self.COLS) == 0]
                cset = set(self.cand)
                order = [i for i in self.rng.permutation(n) if i not in cset]
                t0 = self.pick_t - self.POP - 0.1
                for j, i in enumerate(order):
                    gone[i] = t0 + self.POP * (j / len(order)) ** 0.8
            if self.full_t is not None and not self.glides and t >= self.pick_t:
                self._pick(P, V, SP, SV, gone, t)
            scale = np.clip((gone[:n] - t) / 0.12, 0, 1).astype(np.float32)
            self.frames.append((P[:n].astype(np.float32).copy(), gen[:n].astype(np.int16).copy(),
                                born[:n].astype(np.float32).copy(), scale))
            self.parts_log.append(self.parts(t))
            if self.loop_t is not None and t >= self.loop_t - 0.5 / FPS:
                break
            if t > 64:
                break
        self.n_frames = len(self.frames)
        return self

    def _pick(self, P, V, SP, SV, gone, t):
        want = self.SPEED * self.GLIDE
        used = set()
        r = self.R
        for s in range(len(SP)):
            best, bs = None, 1e18
            for i in self.cand:
                if i in used:
                    continue
                for L in np.linspace(0.3, 1.6, 10) * want:
                    b = [P[i].copy(), P[i] + V[i] / (np.hypot(*V[i]) + 1e-9) * L / 3,
                         SP[s] - SV[s] / (np.hypot(*SV[s]) + 1e-9) * L / 3, SP[s].copy()]
                    u = np.linspace(0, 1, 120)[:, None]
                    pts = (1 - u) ** 3 * b[0] + 3 * (1 - u) ** 2 * u * b[1] + 3 * (1 - u) * u ** 2 * b[2] + u ** 3 * b[3]
                    cs = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
                    inside = ((pts[:, 0] > X0 + r) & (pts[:, 0] < X1 - r) & (pts[:, 1] > Y0 + r)
                              & (pts[:, 1] < Y1 - r)).all()
                    # Weg darf nicht durch die Stempel gehen (Zeitpunkt entlang des Wegs berücksichtigen)
                    ok = True
                    for k_ in range(0, 120, 6):
                        tt = t + self.GLIDE * cs[k_] / cs[-1]
                        for (ax, ay, bx, by, rad, *_r) in self.parts(tt):
                            if ax - r - 4 < pts[k_, 0] < bx + r + 4 and abs(pts[k_, 1] - ay) < rad + r + 4:
                                ok = False
                        if pts[k_, 1] < min(q[1] for q in self.parts(tt)) and False:
                            ok = False
                    sc = abs(cs[-1] - want) + (0 if inside else 1e6) + (0 if ok else 1e5)
                    if sc < bs:
                        best, bs = (i, b, cs, u[:, 0], cs[-1]), sc
            i, b, cs, uu, L = best
            used.add(i)
            self.glides[i] = (b, cs, uu, L)
            self.surv.append(i)
        rest = [i for i in self.cand if i not in used]
        for j, i in enumerate(rest):
            gone[i] = t + 0.3 * j / max(1, len(rest))


THEME = dict(bg=(10, 8, 18), bg_glow=(34, 24, 54), frame=(120, 230, 255), text=(235, 245, 255))
COLS = [(255, 70, 110), (255, 160, 40), (255, 230, 60), (90, 230, 110), (40, 190, 255), (170, 110, 255),
        (255, 120, 220), (245, 245, 255)]
Stampfwerk.COLS = COLS


class StampRenderer:
    def __init__(self, sim):
        self.sim = sim
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        bg = np.array(THEME["bg"], np.float32) + np.exp(-(((xx - CX) / 560) ** 2 + ((yy - CY) / 620) ** 2))[..., None] \
            * np.array(THEME["bg_glow"], np.float32)
        self.base = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(self.base, "RGBA")
        fr = THEME["frame"]
        for w_, al in ((22, 28), (14, 60), (7, 255)):
            d.rounded_rectangle([(X0 - w_ / 2) * k, (Y0 - w_ / 2) * k, (X1 + w_ / 2) * k, (Y1 + w_ / 2) * k],
                                radius=8 * k, outline=fr + (al,), width=int(w_ * k / 2))
        # Querträger oben, an dem die Stempel hängen
        d.rounded_rectangle([(X0 - 30) * k, (Y0 - 230) * k, (X1 + 30) * k, (Y0 - 170) * k], radius=12 * k,
                            fill=(46, 50, 62, 255), outline=(20, 22, 28, 255), width=3 * k)
        d.rectangle([(X0 - 20) * k, (Y0 - 222) * k, (X1 + 20) * k, (Y0 - 214) * k], fill=(120, 128, 145, 255))
        self.sprites = [jelly_sprite(sim.R * k * 1.05, c) for c in COLS]

    def draw_rods(self, d, parts):
        k = SS
        for (ax, ay, bx, by, rad, x, *_r) in parts:
            top = Y0 - 170
            for xr in (x - 40, x + 40):                       # Kolbenstangen (hinter den Bällen)
                d.rectangle([(xr - 10) * k, top * k, (xr + 10) * k, (ay - rad) * k], fill=(78, 84, 100, 255))
                d.rectangle([(xr - 6) * k, top * k, (xr - 2) * k, (ay - rad) * k], fill=(160, 168, 185, 255))
            d.rectangle([(x - 30) * k, (top - 6) * k, (x + 30) * k, (top + 26) * k], fill=(70, 76, 90, 255))

    def draw_stamps(self, d, parts):
        k = SS
        for (ax, ay, bx, by, rad, x, *_r) in parts:
            # schwere Pressplatte
            d.rounded_rectangle([(ax - rad) * k, (ay - rad) * k, (bx + rad) * k, (ay + rad) * k], radius=10 * k,
                                fill=(28, 30, 38, 255))
            d.rounded_rectangle([(ax - rad + 3) * k, (ay - rad + 3) * k, (bx + rad - 3) * k, (ay + rad - 3) * k],
                                radius=8 * k, fill=(128, 136, 152, 255))
            d.rounded_rectangle([(ax - rad + 8) * k, (ay - rad + 5) * k, (bx + rad - 8) * k, (ay - 3) * k],
                                radius=6 * k, fill=(205, 212, 226, 255))
            d.rectangle([(ax + 6) * k, (ay - rad + 7) * k, (bx - 40) * k, (ay - rad + 10) * k],
                        fill=(255, 255, 255, 210))
            for xs in np.linspace(ax - 8, bx + 8, 8):          # Warnstreifen an der Unterkante
                d.line([((xs - 7) * k, (ay + rad - 4) * k), ((xs + 7) * k, (ay + 5) * k)], fill=(255, 190, 0, 255),
                       width=5 * k)

    def frame(self, fi):
        sim, k = self.sim, SS
        t = (fi + 1) / FPS
        Pn, gens, born, scale = sim.frames[fi]
        im = self.base.copy()
        self.draw_rods(ImageDraw.Draw(im, "RGBA"), sim.parts_log[fi])
        for i in np.nonzero(scale > 0)[0]:
            x, y = Pn[i]
            sp = self.sprites[int(gens[i]) % len(self.sprites)]
            sx = sy = 1.0
            e = (t - born[i]) / 0.2
            if 0 <= e < 1:                                 # frisch zerquetscht: flach, federt zurück
                q = (1 - e) ** 2
                sx, sy = 1 + 0.4 * q, 1 - 0.38 * q
            if scale[i] < 1:
                f = (1 + 0.35 * (1 - scale[i])) * scale[i] ** 0.5
                sx, sy = sx * f, sy * f
            if sx != 1.0 or sy != 1.0:
                sp = sp.resize((max(2, int(sp.width * sx)), max(2, int(sp.height * sy))), Image.BILINEAR)
            im.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        top = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.draw_stamps(ImageDraw.Draw(top, "RGBA"), sim.parts_log[fi])
        sh = np.asarray(top.resize((W1 // 4, H1 // 4)).filter(ImageFilter.GaussianBlur(4)).resize(im.size)).copy()
        sh[..., :3] = 0
        sh[..., 3] = (sh[..., 3] * 0.5).astype(np.uint8)
        im.alpha_composite(Image.fromarray(sh, "RGBA"), (5 * k, 12 * k))
        im.alpha_composite(top)
        out = im.convert("RGB").reduce(SS)
        d = ImageDraw.Draw(out, "RGBA")
        n_vis = int(np.sum(scale > 0.5))
        d.text((CX, Y1 + 120), f"{n_vis:,}", font=_font(F_BOLD, 56), fill=THEME["text"] + (240,), anchor="mm")
        return out


def find_start(seed, lo=2.6, hi=4.2, tries=300):
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        SP = np.array([(rng.uniform(X0 + 60, X1 - 60), rng.uniform(CY - 100, Y1 - 60))])
        a = rng.uniform(0, 2 * math.pi)
        SV = np.array([(math.cos(a), math.sin(a))]) * Stampfwerk.SPEED
        s = Stampfwerk(1)
        s.STARTS = (SP, SV)
        s.run(until=hi + 0.1)
        if s.events and lo <= s.events[0][0] <= hi:
            return s.events[0][0], SP, SV
    raise RuntimeError("kein Start")


def audio(sim, path):
    F.audio(sim, path, sample="assets/meow.wav", vol=0.55)


if __name__ == "__main__":
    first, SP, SV = find_start(5)
    sim = Stampfwerk(3)
    sim.STARTS = (SP, SV)
    sim.run()
    vis = [int(np.sum(sim.frames[min(sim.n_frames - 1, q * FPS)][3] > 0.5)) for q in range(0, 60, 4)]
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", len(sim.events), "Bälle", vis)
    if "--preview" in sys.argv:
        R = StampRenderer(sim)
        ts = [1.0, first + 0.15, 15.0, 30.0, sim.full_t, sim.loop_t - 0.3]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, x in enumerate(ts):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a26_check.png")
    else:
        A.ArcRenderer, A.build_audio = StampRenderer, audio
        A.render(sim, "output/a26_miau_stampfwerk_vertical.mp4")
        print("output/a26_miau_stampfwerk_vertical.mp4")
