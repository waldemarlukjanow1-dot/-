"""Will It Overflow? – Kreisarena, unten zwei feste Rampen und ein schwingendes Paddel (Metronom).
Jeder Treffer des Paddels teilt einen Ball in zwei. Die Bälle sammeln sich, der Füllstand steigt –
erreicht er die OVERFLOW-Linie, läuft die Arena über.

Physik mit numba: Bälle (Schwerkraft, Stoß an Kreiswand, Rampen und Paddel, Ball-Ball über ein Raster).
"""
import math
import os
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb

FPS = 60
W1, H1, SS = 1080, 1920, 2
SR = 44100
F_BOLD = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
F_SEMI = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"

CX, CY, R_AR = 520.0, 930.0, 420.0          # Arena
PX, PY = CX, CY + R_AR - 46.0               # Drehpunkt des Paddels
PAD_L, PAD_HALF = 330.0, math.radians(8.5)  # Paddel: Länge, halber Öffnungswinkel
SWING, PERIOD = math.radians(52), 2.7       # Ausschlag, Periode
R_B = 8.0
G = 1500.0
OVER_Y = CY - R_AR * 0.62                   # OVERFLOW-Linie
HUB_R = 40.0


def ramp_polys():
    """Zwei feste Rampen (links/rechts) als Dreiecke, die mit der Kreiswand eine V-Mulde bilden."""
    out = []
    for s in (-1, 1):
        a = math.radians(90 - s * 74)
        far = (CX + math.cos(a) * (R_AR + 30), CY + math.sin(a) * (R_AR + 30))
        out.append(np.array([(PX, PY), far, (CX + s * (R_AR + 30), CY + R_AR + 30),
                             (PX, CY + R_AR + 30)], np.float64))
    return out


RAMPS = ramp_polys()


def paddle_poly(th):
    pts = [(PX, PY)]
    for q in range(9):
        a = -math.pi / 2 + th - PAD_HALF + 2 * PAD_HALF * q / 8
        pts.append((PX + math.cos(a) * PAD_L, PY + math.sin(a) * PAD_L))
    return np.array(pts, np.float64)


def paddle_angle(t):
    return SWING * math.sin(2 * math.pi * t / PERIOD)


def paddle_omega(t):
    return SWING * 2 * math.pi / PERIOD * math.cos(2 * math.pi * t / PERIOD)


@njit(cache=True)
def poly_push(poly, x, y, r):
    """Kollision Kreis gegen konvexes Polygon: (eindringtiefe, nx, ny, kx, ky) oder pen<=0."""
    n = poly.shape[0]
    pos = neg = False
    best_d = 1e18
    bnx = bny = 0.0
    bkx = bky = 0.0
    for i in range(n):
        ax, ay = poly[i, 0], poly[i, 1]
        bx, by = poly[(i + 1) % n, 0], poly[(i + 1) % n, 1]
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey
        u = ((x - ax) * ex + (y - ay) * ey) / L2
        u = min(1.0, max(0.0, u))
        qx, qy = ax + ex * u, ay + ey * u
        dx, dy = x - qx, y - qy
        d = math.sqrt(dx * dx + dy * dy)
        cross = ex * (y - ay) - ey * (x - ax)
        if cross > 0:
            pos = True
        elif cross < 0:
            neg = True
        if d < best_d:
            best_d = d
            bkx, bky = qx, qy
            if d > 1e-9:
                bnx, bny = dx / d, dy / d
            else:
                L = math.sqrt(L2)
                bnx, bny = ey / L, -ex / L
    if not (pos and neg):   # Mittelpunkt im Polygon: nach außen über die nächste Kante
        return best_d + r, -bnx, -bny, bkx, bky
    return r - best_d, bnx, bny, bkx, bky


@njit(cache=True)
def step(P, V, n, pad, ramp0, ramp1, th_w, dt, split_flag, rest):
    """Ein Teilschritt für n Bälle. split_flag[i]=1 wenn Ball i vom Paddel kräftig getroffen wurde."""
    for i in range(n):
        V[i, 1] += G * dt
        P[i, 0] += V[i, 0] * dt
        P[i, 1] += V[i, 1] * dt
        # Kreiswand (innen)
        dx, dy = P[i, 0] - CX, P[i, 1] - CY
        d = math.sqrt(dx * dx + dy * dy)
        if d > R_AR - R_B:
            nx, ny = dx / d, dy / d
            P[i, 0] = CX + nx * (R_AR - R_B)
            P[i, 1] = CY + ny * (R_AR - R_B)
            vn = V[i, 0] * nx + V[i, 1] * ny
            if vn > 0:
                V[i, 0] -= (1 + rest) * vn * nx
                V[i, 1] -= (1 + rest) * vn * ny
        # Rampen
        for poly in (ramp0, ramp1):
            pen, nx, ny, kx, ky = poly_push(poly, P[i, 0], P[i, 1], R_B)
            if pen > 0:
                P[i, 0] += nx * pen
                P[i, 1] += ny * pen
                vn = V[i, 0] * nx + V[i, 1] * ny
                if vn < 0:
                    V[i, 0] -= (1 + rest) * vn * nx
                    V[i, 1] -= (1 + rest) * vn * ny
                    V[i, 0] *= 0.995
        # Nabe (Drehpunkt-Scheibe)
        dx, dy = P[i, 0] - PX, P[i, 1] - PY
        d = math.sqrt(dx * dx + dy * dy)
        if d < HUB_R + R_B and d > 1e-9:
            nx, ny = dx / d, dy / d
            P[i, 0] = PX + nx * (HUB_R + R_B)
            P[i, 1] = PY + ny * (HUB_R + R_B)
            vn = V[i, 0] * nx + V[i, 1] * ny
            if vn < 0:
                V[i, 0] -= (1 + rest) * vn * nx
                V[i, 1] -= (1 + rest) * vn * ny
        # Paddel (bewegt sich: Oberflächengeschwindigkeit = omega x r)
        pen, nx, ny, kx, ky = poly_push(pad, P[i, 0], P[i, 1], R_B)
        if pen > 0:
            P[i, 0] += nx * pen
            P[i, 1] += ny * pen
            sx = -th_w * (ky - PY)
            sy = th_w * (kx - PX)
            rvx, rvy = V[i, 0] - sx, V[i, 1] - sy
            vn = rvx * nx + rvy * ny
            if vn < 0:
                rvx -= (1 + 0.85) * vn * nx
                rvy -= (1 + 0.85) * vn * ny
                V[i, 0], V[i, 1] = rvx + sx, rvy + sy
                if -vn > 260:
                    split_flag[i] = 1
        # Tempo begrenzen
        sp = math.sqrt(V[i, 0] ** 2 + V[i, 1] ** 2)
        if sp > 1700:
            V[i, 0] *= 1700 / sp
            V[i, 1] *= 1700 / sp


@njit(cache=True)
def collide(P, V, n, cell, head, nxt, gw, gh, x0, y0):
    head[:] = -1
    for i in range(n):
        cx = int((P[i, 0] - x0) / cell)
        cy = int((P[i, 1] - y0) / cell)
        if 0 <= cx < gw and 0 <= cy < gh:
            nxt[i] = head[cy * gw + cx]
            head[cy * gw + cx] = i
        else:
            nxt[i] = -1
    D = 2 * R_B
    for i in range(n):
        cx = int((P[i, 0] - x0) / cell)
        cy = int((P[i, 1] - y0) / cell)
        for ox in range(-1, 2):
            for oy in range(-1, 2):
                gx, gy = cx + ox, cy + oy
                if gx < 0 or gy < 0 or gx >= gw or gy >= gh:
                    continue
                j = head[gy * gw + gx]
                while j != -1:
                    if j > i:
                        dx, dy = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
                        d2 = dx * dx + dy * dy
                        if 1e-9 < d2 < D * D:
                            d = math.sqrt(d2)
                            nx, ny = dx / d, dy / d
                            ov = (D - d) * 0.5
                            P[i, 0] -= nx * ov
                            P[i, 1] -= ny * ov
                            P[j, 0] += nx * ov
                            P[j, 1] += ny * ov
                            rv = (V[j, 0] - V[i, 0]) * nx + (V[j, 1] - V[i, 1]) * ny
                            if rv < 0:
                                imp = -rv * 0.9 * 0.5
                                V[i, 0] -= imp * nx
                                V[i, 1] -= imp * ny
                                V[j, 0] += imp * nx
                                V[j, 1] += imp * ny
                    j = nxt[j]


class OverflowSim:
    def __init__(self, seed, n_cap=2600, total=50.0, sub=8, split_gap=0.12, rest=0.55, outro=4.5):
        self.rng = np.random.default_rng(seed)
        self.n_cap, self.total, self.sub, self.split_gap, self.rest, self.outro = n_cap, total, sub, split_gap, rest, outro

    def run(self):
        rng = self.rng
        N = self.n_cap
        P = np.zeros((N, 2))
        V = np.zeros((N, 2))
        born = np.zeros(N)
        gen = np.zeros(N, np.int32)
        last_split = np.full(N, -9.0)
        P[0] = (CX - 60, CY - R_AR * 0.55)
        V[0] = (140, 0)
        n = 1
        cell = 2 * R_B
        x0, y0 = CX - R_AR - 10, CY - R_AR - 10
        gw = gh = int((2 * R_AR + 20) / cell) + 1
        head = np.full(gw * gh, -1, np.int64)
        nxt = np.full(N, -1, np.int64)
        flag = np.zeros(N, np.int64)
        self.frames, self.events, self.level = [], [], []
        dt = 1.0 / (FPS * self.sub)
        t = 0.0
        self.over_t = None
        ramp0, ramp1 = RAMPS
        n_fr = int((self.total + self.outro) * FPS)
        for fi in range(n_fr):
            for _ in range(self.sub):
                th = paddle_angle(t) if self.over_t is None or t < self.over_t + 1.0 else \
                    paddle_angle(self.over_t + 1.0) * max(0.0, 1 - (t - self.over_t - 1.0) / 2.0)
                w = paddle_omega(t) if self.over_t is None or t < self.over_t + 1.0 else 0.0
                pad = paddle_poly(th)
                flag[:n] = 0
                step(P, V, n, pad, ramp0, ramp1, w, dt, flag, self.rest)
                collide(P, V, n, cell, head, nxt, gw, gh, x0, y0)
                if self.over_t is None:
                    hits = np.nonzero(flag[:n])[0]
                    for i in hits:
                        if n >= N or t - last_split[i] < self.split_gap:
                            continue
                        last_split[i] = t
                        a = math.atan2(V[i, 1], V[i, 0]) + rng.uniform(-0.6, 0.6)
                        sp = math.hypot(V[i, 0], V[i, 1]) * rng.uniform(0.85, 1.05)
                        P[n] = P[i] + rng.normal(0, 1.5, 2)
                        V[n] = (math.cos(a) * sp, math.sin(a) * sp)
                        born[n] = t
                        gen[n] = gen[i] + 1
                        last_split[n] = t
                        self.events.append((t, "split", {"n": n + 1}))
                        n += 1
                t += dt
            # Füllstand: Höhe der langsamen, liegenden Bälle (oberes Perzentil)
            slow = np.hypot(V[:n, 0], V[:n, 1]) < 180
            lvl = float(np.percentile(P[:n][slow, 1], 3)) if slow.sum() > 40 else CY + R_AR
            self.level.append(lvl)
            if self.over_t is None and slow.sum() > 40 and lvl < OVER_Y:
                self.over_t = t
                self.events.append((t, "over", {"n": n}))
            self.frames.append((P[:n].astype(np.float32).copy(), born[:n].astype(np.float32).copy(), paddle_angle(t)
                                if self.over_t is None or t < self.over_t + 1.0 else
                                paddle_angle(self.over_t + 1.0) * max(0.0, 1 - (t - self.over_t - 1.0) / 2.0)))
            if self.over_t is not None and t > self.over_t + self.outro:
                break
        self.n_frames = len(self.frames)
        self.dur = self.n_frames / FPS
        return self


# ------------------------------------------------------------------ Darstellung

def _font(path, size, cache={}):
    k = (path, size)
    if k not in cache:
        cache[k] = ImageFont.truetype(path, size)
    return cache[k]


BALL_PAL = [(255, 64, 160), (255, 120, 60), (255, 210, 50), (120, 255, 120), (40, 230, 255), (150, 110, 255)]


def ball_col(i):
    v = (i * 0.0137) % 1.0 * len(BALL_PAL)
    a, b = BALL_PAL[int(v) % 6], BALL_PAL[(int(v) + 1) % 6]
    f = v - int(v)
    return tuple(int(x + (y - x) * f) for x, y in zip(a, b))


class OverflowRenderer:
    def __init__(self, sim):
        self.sim = sim
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        f = (yy / H1)[..., None]
        base = np.array((16, 12, 34), np.float32) * (1 - f) + np.array((34, 12, 44), np.float32) * f
        glow = np.exp(-(((xx - CX) / 520) ** 2 + ((yy - CY) / 520) ** 2))[..., None] * np.array((30, 20, 60))
        base = base + glow
        inside = (xx - CX) ** 2 + (yy - CY) ** 2 < R_AR ** 2
        base[inside] = base[inside] * 0.55 + np.array((4, 3, 10))
        self.base = Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("RGBA")
        d = ImageDraw.Draw(self.base, "RGBA")
        # Ring in Segmenten (wie ein Messring)
        R0, R1 = R_AR * k, (R_AR + 26) * k
        for q in range(72):
            a0, a1 = q * 5 + 0.6, q * 5 + 4.4
            d.arc([CX * k - (R0 + R1) / 2, CY * k - (R0 + R1) / 2, CX * k + (R0 + R1) / 2, CY * k + (R0 + R1) / 2],
                  a0, a1, fill=(60, 230, 200) if q % 2 else (40, 180, 170), width=int(R1 - R0))
        # statische Teile: Rampen (Koralle) + Nabe, auf den Kreis beschnitten
        lay = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        dl = ImageDraw.Draw(lay)
        for poly in RAMPS:
            dl.polygon([(x * k, y * k) for x, y in poly], fill=(255, 120, 110, 255))
            # Glanzkante
            dl.line([(poly[0][0] * k, poly[0][1] * k), (poly[1][0] * k, poly[1][1] * k)], fill=(255, 210, 190, 255),
                    width=4 * k)
            for u in (0.35, 0.6):   # Rillen wie im Vorbild
                ax, ay = poly[0][0] + (poly[1][0] - poly[0][0]) * u, poly[0][1] + (poly[1][1] - poly[0][1]) * u
                dl.line([(ax * k, (ay + 40) * k), ((ax + (poly[1][0] - poly[0][0]) * 0.18) * k,
                                                   (ay + 40 + (poly[1][1] - poly[0][1]) * 0.18) * k)],
                        fill=(210, 80, 90, 255), width=3 * k)
        m = Image.new("L", self.base.size, 0)
        ImageDraw.Draw(m).ellipse([(CX - R_AR) * k, (CY - R_AR) * k, (CX + R_AR) * k, (CY + R_AR) * k], fill=255)
        self.ramps = Image.new("RGBA", self.base.size, (0, 0, 0, 0))
        self.ramps.paste(lay, (0, 0), Image.composite(lay.getchannel("A"), Image.new("L", lay.size, 0), m))
        self.mask = m

    def frame(self, fi):
        sim = self.sim
        k = SS
        t = (fi + 1) / FPS
        Pn, born, th = sim.frames[fi]
        n = len(Pn)
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        # OVERFLOW-Linie
        lvl = sim.level[fi]
        near = max(0.0, min(1.0, 1 - (lvl - OVER_Y) / 220))
        over = sim.over_t is not None and t >= sim.over_t
        hw = math.sqrt(max(0.0, R_AR ** 2 - (OVER_Y - CY) ** 2))
        col = (255, 70, 90) if over or near > 0.8 else (255, 255, 255)
        al = int(140 + 115 * near) if not over else int(255 * (0.6 + 0.4 * math.sin(t * 14)))
        x = CX - hw
        while x < CX + hw:
            d.line([x * k, OVER_Y * k, min(x + 16, CX + hw) * k, OVER_Y * k], fill=col + (al,), width=3 * k)
            x += 28
        # Bälle
        r = R_B * k
        for i in range(n):
            x, y = Pn[i, 0] * k, Pn[i, 1] * k
            c = ball_col(i)
            d.ellipse([x - r, y - r, x + r, y + r], fill=c + (255,))
            d.ellipse([x - r * 0.55, y - r * 0.6, x - r * 0.05, y - r * 0.15], fill=(255, 255, 255, 150))
        # Rampen, Paddel, Nabe
        im.alpha_composite(self.ramps)
        pad = paddle_poly(th)
        lay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dl = ImageDraw.Draw(lay)
        dl.polygon([(x * k, y * k) for x, y in pad], fill=(255, 230, 120, 255))
        tip = pad[1:]
        dl.line([(x * k, y * k) for x, y in tip], fill=(255, 250, 210, 255), width=4 * k)
        for q in range(1, 6):     # Skalenstriche wie ein Messbecher
            u = 0.3 + q * 0.11
            a = -math.pi / 2 + th
            cxp, cyp = PX + math.cos(a) * PAD_L * u, PY + math.sin(a) * PAD_L * u
            wq = PAD_L * u * math.tan(PAD_HALF) * 0.55
            nx_, ny_ = -math.sin(a), math.cos(a)
            dl.line([((cxp - nx_ * wq) * k, (cyp - ny_ * wq) * k), ((cxp + nx_ * wq * 0.2) * k,
                                                                    (cyp + ny_ * wq * 0.2) * k)],
                    fill=(200, 150, 40, 255), width=3 * k)
        dl.ellipse([(PX - HUB_R) * k, (PY - HUB_R) * k, (PX + HUB_R) * k, (PY + HUB_R) * k], fill=(255, 150, 120, 255))
        dl.ellipse([(PX - 9) * k, (PY - 9) * k, (PX + 9) * k, (PY + 9) * k], outline=(120, 40, 60, 255), width=3 * k)
        im.paste(lay, (0, 0), Image.composite(lay.getchannel("A"), Image.new("L", lay.size, 0), self.mask))
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(g, np.float32) - 50, 0, 255) * 0.35
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        self.hud(out, t, n, over)
        return out

    def hud(self, im, t, n, over):
        sim = self.sim
        d = ImageDraw.Draw(im, "RGBA")
        d.text((CX, 300), "Will It Overflow?", font=_font(F_BOLD, 66), fill=(255, 255, 255, 255), anchor="mm")
        d.text((CX, 362), "every hit splits a ball", font=_font(F_SEMI, 32), fill=(190, 185, 215, 255), anchor="mm")
        hw = math.sqrt(max(0.0, R_AR ** 2 - (OVER_Y - CY) ** 2))
        d.text((CX + hw - 34, OVER_Y - 18), "OVERFLOW", font=_font(F_BOLD, 22),
               fill=(255, 90, 110, 230) if over else (255, 255, 255, 170), anchor="rm")
        # Zähler unter der Arena
        d.text((CX, CY + R_AR + 74), f"{n:,}", font=_font(F_BOLD, 50), fill=(255, 255, 255, 240), anchor="mm")
        d.text((CX, CY + R_AR + 116), "BALLS", font=_font(F_SEMI, 24), fill=(170, 165, 200, 230), anchor="mm")
        if over:
            e = t - sim.over_t
            a = min(1.0, e / 0.35)
            sc = 1 + 0.2 * max(0.0, 1 - e / 0.3)
            d.rounded_rectangle([CX - 330, CY - 120, CX + 330, CY + 20], radius=32, fill=(15, 8, 25, int(225 * a)),
                                outline=(255, 90, 110, int(220 * a)), width=4)
            d.text((CX, CY - 72), "IT OVERFLOWED!", font=_font(F_BOLD, int(62 * sc)),
                   fill=(255, 90, 110, int(255 * a)), anchor="mm")
            d.text((CX, CY - 18), f"after {sim.over_t:.1f} seconds", font=_font(F_SEMI, 32),
                   fill=(255, 255, 255, int(235 * a)), anchor="mm")


# ------------------------------------------------------------------ Klang: nur weiches Ploppen

def build_audio(sim, path):
    from .breaker import plop
    n = int(SR * sim.dur)
    out = np.zeros(n + SR * 2)
    last = -1.0

    def put(t, s, vol):
        i = int(t * SR)
        e = min(len(out), i + len(s))
        if i < e:
            out[i:e] += s[: e - i] * vol

    for t, kind, d in sim.events:
        if kind == "split":
            if t - last < 0.06:
                continue
            last = t
            f = 420 + 380 * min(1.0, math.log10(max(1, d["n"])) / 3.6)
            put(t, plop(round(f / 20) * 20), 0.45)
        elif kind == "over":
            for j, f in enumerate((300, 380, 480)):
                put(t + j * 0.11, plop(f, big=True), 0.9)
    out = out[:n]
    tt = np.arange(n) / SR
    mix_ = out + reverb(out, 0.6) * 0.08
    mix_ *= np.clip((sim.dur - tt) / 1.5, 0, 1)
    peak = np.max(np.abs(mix_)) or 1
    mix_ = mix_ / peak * 0.6
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
    _R = OverflowRenderer(sim)
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
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-af", "loudnorm=I=-17:TP=-2:LRA=11",
                        "-movflags", "+faststart", out_path], check=True)


SEED_OVER = 0

if __name__ == "__main__":
    import sys
    import time
    t0 = time.time()
    sim = OverflowSim(SEED_OVER, n_cap=4200, total=60.0).run()
    print("Sim", round(time.time() - t0, 1), "s, overflow", sim.over_t, "Dauer", round(sim.dur, 1),
          "Bälle", len(sim.frames[-1][0]))
    if "--preview" in sys.argv:
        R = OverflowRenderer(sim)
        ts = [1.0, 10.0, 25.0, 38.0, sim.over_t + 0.6, sim.dur - 0.2]
        t1 = time.time()
        ims = [R.frame(min(sim.n_frames - 1, int(x * FPS))) for x in ts]
        print("render/frame", round((time.time() - t1) / len(ts), 2))
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/overflow_check.png")
        tiktok_ui_mock(ims[2], "vertical").save("output/preview/overflow_ui.png")
    else:
        render(sim, "output/o01_will_it_overflow_vertical.mp4")
        print("output/o01_will_it_overflow_vertical.mp4")
