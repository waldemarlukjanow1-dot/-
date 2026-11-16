"""Arc Box (nach dem viralen Vorbild "The Bouncing Ball"): ein quadratischer Rahmen mit Gelenkpunkten in den vier
Ecken. An jeder Ecke dreht sich ein großer Halbkreis-Bogen um den Eckpunkt und fegt dabei durch den Rahmen.
Die Bälle fliegen schwerelos mit konstantem Tempo und prallen an Rahmen, Bögen und einander ab. Berührt ein Bogen
einen Ball, teilt sich der Ball in zwei – und die zwei bekommen die nächste Farbe (jede Generation eine Farbe).

Unser Stil: dunkler Hintergrund, Neon-Rahmen mit Glow, goldene Bögen, glänzende Neon-Bälle, Zähler unten,
weiches Plopp – keine Ring-Effekte, kein Schluss-Sound.
"""
import math
import os
import subprocess
import tempfile
import wave
from multiprocessing import Pool

import numpy as np
from numba import njit
from PIL import Image, ImageDraw, ImageFilter

from tiktok.render import tiktok_ui_mock
from .marble_render import reverb
from .multiply import collide, _font, F_BOLD

FPS = 60
W1, H1, SS = 1080, 1920, 2
SR = 44100
CX, CY = 520.0, 870.0
HALF = 340.0                                      # halbe Kantenlänge des Rahmens
X0, X1, Y0, Y1 = CX - HALF, CX + HALF, CY - HALF, CY + HALF
CORNERS = [(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]
ARC_D = 1.25 * HALF                               # Durchmesser der Halbkreise
ARC_R = ARC_D / 2
SIDE = (1, 1, 1, 1)                               # Wölbungsrichtung je Ecke
ARC_W = 16.0                                      # Dicke der Bögen
SPAN = math.pi                                    # Halbkreis
GEN_COLS = [(57, 255, 20), (255, 20, 147), (190, 20, 255), (0, 120, 255), (255, 240, 0), (0, 255, 255),
            (255, 90, 0), (255, 0, 70)]


@njit(cache=True)
def box_walls(P, V, n, r):
    for i in range(n):
        if P[i, 0] < X0 + r:
            P[i, 0] = X0 + r
            V[i, 0] = abs(V[i, 0])
        elif P[i, 0] > X1 - r:
            P[i, 0] = X1 - r
            V[i, 0] = -abs(V[i, 0])
        if P[i, 1] < Y0 + r:
            P[i, 1] = Y0 + r
            V[i, 1] = abs(V[i, 1])
        elif P[i, 1] > Y1 - r:
            P[i, 1] = Y1 - r
            V[i, 1] = -abs(V[i, 1])


@njit(cache=True)
def arc_hit(P, V, n, r, px, py, a0, om, hit, qx0, qy0, ARC_R, val=1):
    """Ball gegen Halbkreis-Bogen mit Mittelpunkt (px,py), Radius ARC_R, Bogen von a0 bis a0+SPAN.
    Der Bogen dreht sich um den Gelenkpunkt (qx0,qy0) mit Winkelgeschwindigkeit om."""
    for i in range(n):
        dx, dy = P[i, 0] - px, P[i, 1] - py
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6 or abs(d - ARC_R) > ARC_W / 2 + r + 30:
            continue
        ang = math.atan2(dy, dx)
        rel = (ang - a0) % (2 * math.pi)
        nx = ny = 0.0
        pen = 0.0
        cx = cy = 0.0
        if rel <= SPAN:                               # auf dem Bogen: radial abstoßen
            dr = d - ARC_R
            if abs(dr) < ARC_W / 2 + r:
                s = 1.0 if dr >= 0 else -1.0
                nx, ny = s * dx / d, s * dy / d
                pen = ARC_W / 2 + r - abs(dr)
                cx, cy = px + dx / d * ARC_R, py + dy / d * ARC_R
        else:                                         # Enden des Bogens (runde Kappen)
            for e in (0, 1):
                a = a0 + e * SPAN
                ex, ey = px + math.cos(a) * ARC_R, py + math.sin(a) * ARC_R
                qx, qy = P[i, 0] - ex, P[i, 1] - ey
                q = math.sqrt(qx * qx + qy * qy)
                if q < ARC_W / 2 + r and q > 1e-6:
                    nx, ny = qx / q, qy / q
                    pen = ARC_W / 2 + r - q
                    cx, cy = ex, ey
        if pen > 0:
            P[i, 0] += nx * pen
            P[i, 1] += ny * pen
            sx, sy = -om * (cy - qy0), om * (cx - qx0)    # Geschwindigkeit der Bogenoberfläche
            rvx, rvy = V[i, 0] - sx, V[i, 1] - sy
            vn = rvx * nx + rvy * ny
            if vn < 0:
                rvx -= 2 * vn * nx
                rvy -= 2 * vn * ny
                V[i, 0], V[i, 1] = rvx + sx, rvy + sy
            hit[i] = val


@njit(cache=True)
def keep_speed(V, n, sp):
    for i in range(n):
        s = math.sqrt(V[i, 0] ** 2 + V[i, 1] ** 2)
        if s > 1e-6:
            f = 1 + (sp / s - 1) * 0.08                    # sanft zurück zum Grundtempo
            V[i, 0] *= f
            V[i, 1] *= f


class ArcBox:
    R = 14.0
    CAP = 1300
    SUB = 6
    SPEED = 480.0
    GAP = 0.35                     # ein Ball kann sich höchstens alle 0,35 s teilen
    OMS = (1.9, 1.6, 2.1, 1.75)     # alle im Uhrzeigersinn (rad/s)
    PH = (0.3, 2.0, 4.0, 5.3)
    N_FULL = 920
    DUR = 70.0
    OUTRO = 3.5

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def arc_angle(self, k, t):
        return self.PH[k] + self.OMS[k] * self.spin(t)

    def arc_geom(self, k, t):
        """Halbkreis, dessen eines Ende am Eckpunkt hängt: Mittelpunkt, Startwinkel des Bogens."""
        px, py = CORNERS[k]
        th = self.arc_angle(k, t)
        cx, cy = px + math.cos(th) * ARC_R, py + math.sin(th) * ARC_R
        return cx, cy, th + math.pi          # Bogen von der Ecke (th+pi) bis zum fernen Ende (th+2pi)

    def spin(self, t):
        """Bögen drehen bis 'voll', danach laufen sie weich aus."""
        if self.full_t is None or t <= self.full_t:
            return t
        e = min(1.0, (t - self.full_t) / 2.0)
        return self.full_t + 2.0 * (e - e * e / 2)

    def omega(self, k, t):
        if self.full_t is not None and t > self.full_t:
            return self.OMS[k] * max(0.0, 1 - (t - self.full_t) / 2.0)
        return self.OMS[k]

    def allow(self, n, t):
        """Wachstum folgt einer glatten Kurve: früh spannend, am Ende voll (geteilt wird nur bei echter Berührung)."""
        T = self.T_FULL
        return n < 1 + (self.N_FULL - 1) * (min(t, T) / T) ** 2.1 + 2

    T_FULL = 34.0

    def run(self):
        rng = self.rng
        N, r = self.CAP, self.R
        P, V = np.zeros((N, 2)), np.zeros((N, 2))
        gen = np.zeros(N, np.int64)
        last = np.full(N, -9.0)
        born = np.full(N, -9.0)          # Zeitpunkt der Teilung (für das Plattdrücken)
        nang = np.zeros(N)               # Richtung der Wand-Normalen beim Quetschen
        self.arc_squeeze = [[] for _ in range(4)]
        P[0] = (X1 - 60, Y1 - 120)
        a = math.radians(-35)
        V[0] = (math.cos(a) * self.SPEED, math.sin(a) * self.SPEED)
        n = 1
        cell = 2 * r
        x0, y0 = X0 - 10, Y0 - 10
        gw = gh = int((2 * HALF + 20) / cell) + 1
        head = np.full(gw * gh, -1, np.int64)
        nxt = np.full(N, -1, np.int64)
        hit = np.zeros(N, np.int64)
        dt = 1.0 / (FPS * self.SUB)
        t = 0.0
        self.full_t = None
        self.frames, self.events = [], []
        for fi in range(int(self.DUR * FPS)):
            for _ in range(self.SUB):
                P[:n] += V[:n] * dt
                hit[:n] = 0
                for k, (qx, qy) in enumerate(CORNERS):
                    cx_, cy_, a0 = self.arc_geom(k, t)
                    arc_hit(P, V, n, r, cx_, cy_, a0, self.omega(k, t), hit, qx, qy, ARC_R, k + 1)
                # gequetscht = vom Bogen gedrückt UND an der Wand des Vierecks
                wallnear = ((P[:n, 0] < X0 + r + 2.5) | (P[:n, 0] > X1 - r - 2.5) |
                            (P[:n, 1] < Y0 + r + 2.5) | (P[:n, 1] > Y1 - r - 2.5))
                hit[:n] = np.where(wallnear, hit[:n], 0)
                for _ in range(2):
                    collide(P, V, n, r, cell, head, nxt, gw, gh, x0, y0)
                    box_walls(P, V, n, r)
                keep_speed(V, n, self.SPEED if self.full_t is None else self.SPEED * 0.5)
                if self.full_t is None:
                    for i in np.nonzero(hit[:n])[0]:
                        if n >= N or t - last[i] < self.GAP or not self.allow(n, t):
                            continue
                        # Teilung: zwei Bälle der nächsten Farbe, nebeneinander, gleiche Bewegung leicht gespreizt
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
                        last[i] = last[n] = t
                        born[i] = born[n] = t
                        # Normale der nächstgelegenen Wand
                        dl, dr_, dt_, db = P[i, 0] - X0, X1 - P[i, 0], P[i, 1] - Y0, Y1 - P[i, 1]
                        mn = min(dl, dr_, dt_, db)
                        nang[i] = nang[n] = 0.0 if mn in (dl, dr_) else math.pi / 2
                        self.arc_squeeze[int(hit[i]) - 1].append(t)
                        self.events.append((t, "split", {"n": n + 1, "gen": int(g)}))
                        n += 1
                t += dt
            self.frames.append((P[:n].astype(np.float32).copy(), gen[:n].astype(np.int8).copy(),
                                [self.arc_geom(k, t) for k in range(4)], born[:n].astype(np.float32).copy(),
                                nang[:n].astype(np.float32).copy()))
            if self.full_t is None and n >= self.N_FULL:
                self.full_t = t
            if self.full_t is not None and t > self.full_t + self.OUTRO:
                break
        self.n_frames = len(self.frames)
        return self


# ------------------------------------------------------------------ Darstellung

RING_A, RING_B = (255, 90, 190), (90, 200, 255)
NEON, NEON_HI, NEON_D, NEON_DD = (57, 255, 60), (210, 255, 200), (10, 150, 40), (5, 70, 20)
GOLD, GOLD_HI, GOLD_D = (255, 214, 120), (255, 240, 200), (215, 160, 70)
LAV, LAV_HI = (178, 168, 235), (225, 220, 255)


def ball_sprites(r):
    out = []
    for col in GEN_COLS:
        n = int(r * 5)
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2 + 0.5
        d = np.hypot(xx, yy)
        core = np.clip(r - d + 0.7, 0, 1)
        glow = np.clip(1 - d / (r * 2.9), 0, 1) ** 2.0 * 0.85
        c = np.array(col, np.float32)
        light = np.clip(1.0 - ((xx + r * 0.35) ** 2 + (yy + r * 0.45) ** 2) / (2.2 * r * r), 0, 1)
        rgb = c * (0.82 + 0.35 * light[..., None])
        hard = np.exp(-(((xx + r * 0.36) / (r * 0.2)) ** 2 + ((yy + r * 0.42) / (r * 0.16)) ** 2))
        rgb = rgb + hard[..., None] * (255 - rgb) * 0.9
        gl = np.zeros_like(rgb)
        gl[:] = c
        rgb = rgb * core[..., None] + gl * (1 - core[..., None])
        a = np.maximum(glow, core)
        out.append(Image.fromarray(np.dstack([np.clip(rgb, 0, 255), a * 255]).astype(np.uint8), "RGBA"))
    return out


class ArcRenderer:
    def __init__(self, sim):
        self.sim = sim
        k = SS
        yy, xx = np.mgrid[0:H1 * k, 0:W1 * k].astype(np.float32) / k
        f = (yy / H1)[..., None]
        base = np.array((12, 10, 28), np.float32) * (1 - f) + np.array((26, 10, 36), np.float32) * f
        base += np.exp(-(((xx - CX) / 560) ** 2 + ((yy - CY) / 560) ** 2))[..., None] * np.array((22, 16, 44))
        inside = (xx > X0) & (xx < X1) & (yy > Y0) & (yy < Y1)
        base[inside] = base[inside] * 0.55 + np.array((4, 3, 10))
        # Neon-Rahmen mit Farbverlauf und Glow
        dist = np.minimum(np.minimum(np.abs(xx - X0), np.abs(xx - X1)), np.minimum(np.abs(yy - Y0), np.abs(yy - Y1)))
        on = (xx > X0 - 20) & (xx < X1 + 20) & (yy > Y0 - 20) & (yy < Y1 + 20)
        dist = np.where(on, dist, 999)
        ang = (np.arctan2(yy - CY, xx - CX) + math.pi) / (2 * math.pi)
        mixv = (0.5 + 0.5 * np.cos(ang * 2 * math.pi))[..., None]
        col = np.array(RING_A, np.float32) * mixv + np.array(RING_B, np.float32) * (1 - mixv)
        band = np.clip(1 - dist / 4.5, 0, 1)
        glow = np.exp(-(dist / 22) ** 2) * 0.55
        base = base * (1 - band[..., None]) + (col * 0.5 + 255 * 0.5) * band[..., None] + col * glow[..., None] * 0.7
        self.base = Image.fromarray(base.clip(0, 255).astype(np.uint8)).convert("RGBA")
        self.sprites = ball_sprites(sim.R * k)
        self.ghost = []
        for sp in self.sprites:   # zarte Kopie für die Bewegungsunschärfe
            a = np.asarray(sp).copy()
            a[..., 3] = (a[..., 3] * 0.28).astype(np.uint8)
            self.ghost.append(Image.fromarray(a, "RGBA"))
        self.splits = [e[0] for e in sim.events]

    def arcs(self, d, angs):
        """Neon-grüne Halbkreise mit Textur: dunkler Rand, heller Kern, Glanzlinie und Rillen quer zum Bogen."""
        k = SS
        for (px, py, a) in angs:
            def bb(dr):
                return [(px - ARC_R - dr) * k, (py - ARC_R - dr) * k, (px + ARC_R + dr) * k, (py + ARC_R + dr) * k]
            a0, a1 = math.degrees(a), math.degrees(a + SPAN)
            d.arc(bb(0), a0, a1, fill=NEON_D + (255,), width=int((ARC_W + 4) * k))
            d.arc(bb(0), a0, a1, fill=NEON + (255,), width=int(ARC_W * k))
            d.arc(bb(-3), a0 + 0.5, a1 - 0.5, fill=NEON_HI + (255,), width=int(4 * k))     # Glanz innen
            d.arc(bb(5), a0 + 0.5, a1 - 0.5, fill=NEON_DD + (255,), width=int(2 * k))      # Schatten außen
            for q in range(1, 18):                     # Rillen quer zum Bogen
                e = a + SPAN * q / 18
                c, s_ = math.cos(e), math.sin(e)
                r0, r1 = ARC_R - ARC_W / 2 + 2, ARC_R + ARC_W / 2 - 2
                d.line([((px + c * r0) * k, (py + s_ * r0) * k), ((px + c * r1) * k, (py + s_ * r1) * k)],
                       fill=NEON_D + (200,), width=int(2 * k))
            for e in (a, a + SPAN):                    # runde Enden
                ex, ey = px + math.cos(e) * ARC_R, py + math.sin(e) * ARC_R
                rr = ARC_W / 2 + 1
                d.ellipse([(ex - rr) * k, (ey - rr) * k, (ex + rr) * k, (ey + rr) * k], fill=NEON + (255,),
                          outline=NEON_D + (255,), width=k)
        for px, py in CORNERS:                         # Gelenkpunkte
            d.ellipse([(px - 15) * k, (py - 15) * k, (px + 15) * k, (py + 15) * k], fill=LAV + (255,))
            d.ellipse([(px - 6) * k, (py - 6) * k, (px + 6) * k, (py + 6) * k], fill=(40, 30, 70, 255))

    def frame(self, fi):
        sim, k = self.sim, SS
        t = (fi + 1) / FPS
        Pn, gens, angs = sim.frames[fi][:3]
        im = self.base.copy()
        prev = sim.frames[fi - 1][0] if fi > 0 else Pn
        m = min(len(prev), len(Pn))
        for f, sprs in ((0.33, self.ghost), (0.66, self.ghost)):
            Q = prev[:m] + (Pn[:m] - prev[:m]) * f
            for (x, y), g in zip(Q, gens[:m]):
                sp = sprs[int(g) % len(sprs)]
                im.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        for (x, y), g in zip(Pn, gens):
            sp = self.sprites[int(g) % len(self.sprites)]
            im.alpha_composite(sp, (int(x * k - sp.width / 2), int(y * k - sp.height / 2)))
        glow = Image.new("RGBA", im.size, (0, 0, 0, 0))
        self.arcs(ImageDraw.Draw(glow, "RGBA"), angs)
        g2 = glow.resize((W1 // 4, H1 // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5)).resize(im.size)
        im.alpha_composite(g2)
        im.alpha_composite(glow)
        out = im.convert("RGB").reduce(SS)
        arr = np.asarray(out, np.float32)
        g = out.resize((W1 // 6, H1 // 6), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4)).resize((W1, H1))
        arr = arr + np.clip(np.asarray(g, np.float32) - 50, 0, 255) * 0.35
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(out, "RGBA")
        last = [s for s in self.splits if s <= t]
        pop = 1 + 0.12 * max(0.0, 1 - (t - last[-1]) / 0.15) if last else 1.0
        d.text((CX, Y1 + 120), f"{len(Pn):,}", font=_font(F_BOLD, int(54 * pop)), fill=(255, 255, 255, 235),
               anchor="mm")
        return out


# ------------------------------------------------------------------ Klang: weiches Plopp, Tonhöhe je Farbe

def build_audio(sim, path):
    from .breaker import plop
    dur = sim.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + SR * 2)
    penta = [0, 2, 4, 7, 9, 12, 14, 16]
    last = -1.0
    for t, kind, d in sim.events:
        if t - last < 0.05:
            continue
        last = t
        f = 440.0 * 2 ** (penta[d["gen"] % len(penta)] / 12)
        s = plop(round(f / 10) * 10)
        i = int(t * SR)
        e = min(len(out), i + len(s))
        out[i:e] += s[: e - i] * 0.45
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
           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-maxrate", "24M", "-bufsize", "48M",
           "-pix_fmt", "yuv420p", "-g", "120", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(i0, i1):
        proc.stdin.write(_R.frame(i).tobytes())
    proc.stdin.close()
    proc.wait()
    return out


def render(sim, out_path, jobs=4):
    global _R
    _R = ArcRenderer(sim)
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


SEED_ARC = 1

if __name__ == "__main__":
    import sys
    sim = ArcBox(SEED_ARC).run()
    print("voll", sim.full_t, "Dauer", sim.n_frames / FPS,
          [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(0, 50, 5)])
    if "--preview" in sys.argv:
        R = ArcRenderer(sim)
        ts = [0.5, 3.0, 8.0, 18.0, 30.0, sim.n_frames / FPS - 0.1]
        ims = [R.frame(min(sim.n_frames - 1, int(x * FPS))) for x in ts]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/arcbox_check.png")
        tiktok_ui_mock(ims[3], "vertical").save("output/preview/arcbox_ui.png")
    else:
        render(sim, "output/a04_arc_squeeze_green_vertical.mp4")
        print("output/a04_arc_squeeze_green_vertical.mp4")
