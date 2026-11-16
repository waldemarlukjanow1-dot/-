"""Arc Squeeze – viral-optimierte Kurzversion (~17 s, nahtloser Loop):
  * 4 Start-Bälle, die erste Quetschung passiert in der ersten Sekunde (Hook)
  * schnelleres Füllen, größeres Viereck weiter oben (mehr Fläche im Feed)
  * Ton nur beim Quetschen: Plopp-Tonhöhe steigt mit der Ballzahl (aufsteigende Melodie) +
    leises Summen der Bögen, das mit der Ballzahl anschwillt
  * kurze leuchtende Spur hinter jedem Ball
  * Loop: am Ende bleiben genau die 4 Start-Bälle übrig und fliegen mit normalem Tempo in ihre Startzustände
"""
import math
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import arcbox as A
from . import arcbox2 as B
from .arcbox import X0, X1, Y0, Y1, HALF, ARC_R, FPS, W1, H1, SS
from .arcbox2 import ArcLoop, LoopRenderer, RAINBOW
from .multiply import collide as _collide


class ArcLoopShort(ArcLoop):
    R = 18.0
    N_FULL = 680
    CAP = 900
    T_FULL = 11.0
    GAP = 0.08
    SPEED = 700.0
    OM = 2 * math.pi / (120 / 60)      # eine Umdrehung = 120 Bilder (2 s)
    HOLD = 0.4
    POP = 1.1
    GLIDE = 1.0
    STARTS = None

    def run(self):
        rng = self.rng
        N, r = self.CAP, self.R
        P, V = np.zeros((N, 2)), np.zeros((N, 2))
        gen = np.zeros(N, np.int64)
        last = np.full(N, -9.0)
        born = np.full(N, -9.0)
        nang = np.zeros(N)
        gone = np.full(N, 1e9)
        self.arc_squeeze = [[] for _ in range(4)]
        SP, SV = self.STARTS
        m0 = len(SP)
        P[:m0], V[:m0] = SP, SV
        n = m0
        cell = 2 * r
        x0, y0 = X0 - 10, Y0 - 10
        gw = gh = int((2 * HALF + 20) / cell) + 1
        head = np.full(gw * gh, -1, np.int64)
        nxt = np.full(N, -1, np.int64)
        hit = np.zeros(N, np.int64)
        dt = 1.0 / (FPS * self.SUB)
        period = 2 * math.pi / self.OM
        self.full_t = self.loop_t = None
        self.surv = []
        self.glides = {}
        t = 0.0
        self.frames, self.events = [], []
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
                P[:n][~alive] = (-500.0, -500.0)
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
                for i, (b, cs, uu, L) in self.glides.items():      # geführter Flug in den Startzustand
                    T = self.loop_t - self.pick_t
                    sd = min(L, (t - self.pick_t) * L / T)
                    u = float(np.interp(sd, cs, uu))
                    P[i] = ((1 - u) ** 3 * b[0] + 3 * (1 - u) ** 2 * u * b[1] + 3 * (1 - u) * u ** 2 * b[2]
                            + u ** 3 * b[3])
                    tg = 3 * (1 - u) ** 2 * (b[1] - b[0]) + 6 * (1 - u) * u * (b[2] - b[1]) + 3 * u ** 2 * (b[3] - b[2])
                    V[i] = tg / (np.hypot(*tg) + 1e-9) * L / T
                t += dt
            if self.full_t is None and n >= self.N_FULL:
                self.full_t = t
                self.loop_t = math.ceil((t + self.HOLD + self.POP + self.GLIDE + 0.3) / period) * period
                self.cand = [i for i in range(n) if gen[i] % len(RAINBOW) == 0]
                cset = set(self.cand)
                order = [i for i in rng.permutation(n) if i not in cset]
                t0 = t + self.HOLD
                for j, i in enumerate(order):
                    gone[i] = t0 + self.POP * (j / len(order)) ** 0.8
                self.pick_t = self.loop_t - self.GLIDE
            if self.full_t is not None and not self.glides and t >= self.pick_t:
                used = set()
                for s in range(m0):                    # für jeden Start-Ball den passendsten roten Ball wählen
                    best, bs = None, 1e18
                    want = self.SPEED * self.GLIDE
                    for i in self.cand:
                        if i in used:
                            continue
                        for L in np.linspace(0.3, 1.6, 10) * want:
                            b = self.bezier(P[i].copy(), V[i].copy(), SP[s], SV[s], L)
                            cs, uu, pts = self.bez_len(b)
                            inside = ((pts[:, 0] > X0 + r) & (pts[:, 0] < X1 - r) & (pts[:, 1] > Y0 + r)
                                      & (pts[:, 1] < Y1 - r)).all()
                            sc = abs(cs[-1] - want) + (0 if inside else 1e6)
                            if sc < bs:
                                best, bs = (i, b, cs, uu, cs[-1]), sc
                    i, b, cs, uu, L = best
                    used.add(i)
                    self.glides[i] = (b, cs, uu, L)
                    self.surv.append(i)
                rest = [i for i in self.cand if i not in used]
                for j, i in enumerate(rest):
                    gone[i] = t + 0.3 * j / max(1, len(rest))
            scale = np.clip((gone[:n] - t) / 0.12, 0, 1).astype(np.float32)
            self.frames.append((P[:n].astype(np.float32).copy(), gen[:n].astype(np.int8).copy(),
                                [self.arc_geom(k, t) for k in range(4)], born[:n].astype(np.float32).copy(),
                                nang[:n].astype(np.float32).copy(), scale))
            if self.loop_t is not None and t >= self.loop_t - 0.5 / FPS:
                break
            if t > 60:
                break
        # Loop-Ende: Indizes der Überlebenden auf 0..m0-1 abbilden ist nicht nötig (Renderer zeichnet nach Position)
        self.n_frames = len(self.frames)
        return self


def find_starts(seed, tries=200):
    """Start-Bälle so wählen, dass die erste Quetschung in der ersten Sekunde passiert."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        SP, SV = [], []
        for _k in range(4):
            x = rng.uniform(X0 + 40, X1 - 40)
            y = rng.uniform(Y0 + 40, Y1 - 40)
            a = rng.uniform(0, 2 * math.pi)
            SP.append((x, y))
            SV.append((math.cos(a) * ArcLoopShort.SPEED, math.sin(a) * ArcLoopShort.SPEED))
        SP, SV = np.array(SP), np.array(SV)
        sim = ArcLoopShort(1)
        sim.STARTS = (SP, SV)
        sim.DUR_TEST = True
        # kurzer Testlauf: nur die ersten 1,2 s
        first = probe(sim, 1.2)
        if first is not None and (best is None or first < best[0]) and first > 0.35:
            best = (first, SP, SV)
            if first < 0.8:
                break
    return best


def probe(sim, T):
    r = sim.R
    SP, SV = sim.STARTS
    P, V = SP.copy(), SV.copy()
    n = len(P)
    hit = np.zeros(n, np.int64)
    dt = 1.0 / (FPS * sim.SUB)
    t = 0.0
    while t < T:
        P += V * dt
        hit[:] = 0
        for k, (qx, qy) in enumerate(A.CORNERS):
            cx_, cy_, a0 = sim.arc_geom(k, t)
            A.arc_hit(P, V, n, r, cx_, cy_, a0, sim.OM, hit, qx, qy, ARC_R, k + 1)
        wallnear = ((P[:, 0] < X0 + r + 2.5) | (P[:, 0] > X1 - r - 2.5) | (P[:, 1] < Y0 + r + 2.5) |
                    (P[:, 1] > Y1 - r - 2.5))
        if (hit.astype(bool) & wallnear).any():
            return t
        A.box_walls(P, V, n, r)
        t += dt
    return None


class ShortRenderer(LoopRenderer):
    """+ kurze leuchtende Spur hinter jedem Ball."""
    TRAIL = 4

    def frame(self, fi):
        sim, k = self.sim, SS
        Pn, gens, angs, born, nang, scale = sim.frames[fi]
        trail = Image.new("RGBA", (W1 * k, H1 * k), (0, 0, 0, 0))
        d = ImageDraw.Draw(trail, "RGBA")
        r = sim.R * k * 0.55
        for b in range(1, self.TRAIL + 1):
            F = sim.frames[(fi - b) % sim.n_frames]
            Q, Gq, Sq = F[0], F[1], F[5]
            m = min(len(Q), len(Pn))
            al = int(150 * (1 - b / (self.TRAIL + 1)))
            rr = r * (1 - 0.15 * b)
            for i in range(m):
                if Sq[i] <= 0 or scale[i] <= 0:
                    continue
                x, y = Q[i]
                c = RAINBOW[int(gens[i]) % len(RAINBOW)]
                d.ellipse([x * k - rr, y * k - rr, x * k + rr, y * k + rr], fill=c + (al,))
        self._trail = trail.filter(ImageFilter.GaussianBlur(2 * k))
        return super().frame(fi)


_orig_base_copy = None


def _patched_frame(self, fi):
    return ShortRenderer.frame(self, fi)


# Spur unter die Bälle legen: LoopRenderer.frame beginnt mit self.base.copy() -> Basis um die Spur ergänzen
class _BaseWithTrail:
    def __init__(self, base, trail):
        self.base, self.trail = base, trail

    def copy(self):
        im = self.base.copy()
        im.alpha_composite(self.trail)
        return im


class ShortRenderer2(ShortRenderer):
    def __init__(self, sim):
        super().__init__(sim)
        self._plain = self.base

    def frame(self, fi):
        sim, k = self.sim, SS
        Pn, gens, angs, born, nang, scale = sim.frames[fi]
        trail = Image.new("RGBA", (W1 * k, H1 * k), (0, 0, 0, 0))
        d = ImageDraw.Draw(trail, "RGBA")
        r = sim.R * k * 0.6
        for b in range(1, self.TRAIL + 1):
            F = sim.frames[(fi - b) % sim.n_frames]
            Q, Sq = F[0], F[5]
            m = min(len(Q), len(Pn))
            al = int(140 * (1 - b / (self.TRAIL + 1)))
            rr = r * (1 - 0.12 * b)
            for i in range(m):
                if Sq[i] <= 0 or scale[i] <= 0:
                    continue
                x, y = Q[i]
                c = RAINBOW[int(gens[i]) % len(RAINBOW)]
                d.ellipse([x * k - rr, y * k - rr, x * k + rr, y * k + rr], fill=c + (al,))
        trail = trail.filter(ImageFilter.GaussianBlur(2 * k))
        self.base = _BaseWithTrail(self._plain, trail)
        try:
            return LoopRenderer.frame(self, fi)
        finally:
            self.base = self._plain


def build_audio(sim, path):
    """Ton nur beim Quetschen: Plopp steigt mit der Ballzahl als Melodie; leises Summen schwillt mit der Ballzahl."""
    from .breaker import plop
    from .marble_render import reverb
    SR = 44100
    dur = sim.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + SR * 2)
    scale = [0, 2, 4, 7, 9]
    last = -1.0
    for t, kind, d in sim.events:
        if kind != "split" or t - last < 0.045:
            continue
        last = t
        x = min(1.0, math.log(max(d["n"], 4) / 4) / math.log(sim.N_FULL / 4))   # 0..1 über das ganze Video
        step = int(x * 14)
        f = 392.0 * 2 ** ((12 * (step // 5) + scale[step % 5]) / 12)
        s = plop(round(f / 5) * 5)
        i = int(t * SR)
        e = min(len(out), i + len(s))
        out[i:e] += s[: e - i] * 0.45
    out = out[:n]
    # Summen der Bögen: Lautstärke folgt der sichtbaren Ballzahl je Bild (am Loop-Punkt gleich wie am Anfang)
    vis = np.array([float(np.sum(f[5] > 0.5)) for f in sim.frames])
    vol = np.interp(np.arange(n) / SR * FPS, np.arange(len(vis)), np.log1p(vis) / math.log1p(sim.N_FULL))
    tt = np.arange(n) / SR
    hum = (np.sin(2 * np.pi * 98 * tt) + 0.5 * np.sin(2 * np.pi * 147 * tt)) * (0.012 + 0.03 * vol)
    tail = reverb(np.concatenate([out, np.zeros(SR)]), 0.6) * 0.08
    mix_ = out + tail[:n] + hum
    mix_[:SR] += tail[n:n + SR]
    peak = np.max(np.abs(mix_)) or 1
    mix_ = mix_ / peak * 0.6
    st = np.stack([mix_, mix_], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def render(sim, out_path):
    old_r, old_a = A.ArcRenderer, A.build_audio
    A.ArcRenderer, A.build_audio = ShortRenderer2, build_audio
    try:
        A.render(sim, out_path)
    finally:
        A.ArcRenderer, A.build_audio = old_r, old_a


if __name__ == "__main__":
    import sys
    best = find_starts(3)
    print("erste Quetschung bei", best[0])
    sim = ArcLoopShort(1)
    sim.STARTS = (best[1], best[2])
    sim.run()
    print("voll", sim.full_t, "Loop", sim.loop_t, "Bilder", sim.n_frames,
          [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(0, 20, 2)],
          "erste Teilung", sim.events[0][0] if sim.events else None)
    if "--preview" in sys.argv:
        R = ShortRenderer2(sim)
        ts = [0.2, sim.events[0][0] + 0.05, 5.0, 10.0, sim.full_t + 0.8, sim.loop_t - 0.4]
        ims = [R.frame(min(sim.n_frames - 1, int(x * FPS))) for x in ts]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, im in enumerate(ims):
            out.paste(im.resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/short_check.png")
        ims2 = [R.frame(i) for i in (sim.n_frames - 2, sim.n_frames - 1, 0, 1)]
        out = Image.new("RGB", (360 * 4, 640))
        for i, im in enumerate(ims2):
            out.paste(im.resize((360, 640)), (i * 360, 0))
        out.save("output/preview/short_seam.png")
        from tiktok.render import tiktok_ui_mock
        tiktok_ui_mock(ims[3], "vertical").save("output/preview/short_ui.png")
    else:
        render(sim, "output/a07_arc_squeeze_short_loop_vertical.mp4")
        print("output/a07_arc_squeeze_short_loop_vertical.mp4")
