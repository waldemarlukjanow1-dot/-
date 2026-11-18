"""Arc Squeeze a14 – nach dem Vorbild "4 arms, every crush splits in 3":
  * jede Zerquetschung (Crack) teilt den Ball in DREI
  * die Halbkreise drehen viel langsamer und schwergängig: je mehr Bälle gerade gegen die Wand
    gepresst werden, desto zäher (langsamer) drehen alle vier – synchron, im Uhrzeigersinn
  * graue Metall-Bögen mit Rillen, cyanfarbener Neon-Rahmen, dunkler Hintergrund
  * Farben pro Generation: Blau -> Gelb -> Lila -> Grün -> Rot
  * Dramaturgie wie im Vorbild: schneller Hook (erster Crack < 1 s), dann ein paar Sekunden
    Spannung bei 15 Bällen (nichts passiert), danach Explosion bis die Box voll ist
  * Ton wie im Vorbild: tiefes mahlendes Motor-Brummen (wird bei Last zäher/lauter),
    "Piu" beim Wandabprall, Crack + glasiges Klingeln bei jeder Zerquetschung
  * keine Schrift außer der Ballzahl, nahtloser Loop
"""
import math
import wave

import numpy as np
from PIL import ImageDraw

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox7 as G
from .arcbox import X0, X1, Y0, Y1, HALF, ARC_R, ARC_W, SPAN, FPS
from .multiply import collide as _collide

THEME = dict(bg=(7, 9, 16), bg_glow=(14, 20, 34), frame=(70, 225, 255), frame_glow=(0, 170, 255),
             pivot=(10, 14, 22), text=(235, 240, 248))
ARCS = dict(NEON=(150, 155, 165), NEON_HI=(225, 229, 236), NEON_D=(80, 84, 94), NEON_DD=(40, 42, 50))
BALLS = [(40, 120, 255), (255, 200, 20), (165, 70, 245), (40, 225, 120), (255, 70, 70)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS


class ArcTriple(C.ArcLoopShort):
    R = 14.0
    N_FULL = 950
    CAP = 1300
    SPEED = 430.0
    OM = 2 * math.pi / 10.0          # Grundtempo: 10 s pro Umdrehung (vorher 3 s)
    LOAD_K = 0.22                    # wie stark Last bremst
    OM_MIN = 0.3                     # nie langsamer als 30 % des Grundtempos
    GAP = 0.35
    HOLD = 0.5
    POP = 1.4
    GLIDE = 1.2
    T_HOOK, N_HOOK = 2.4, 15         # schneller Start bis 15 Bälle ...
    T_WAIT = 9.0                     # ... dann Spannung (keine Teilung) bis hier
    T_FULL = 44.0
    SYM = math.pi / 2                # Bögen sind um 90° symmetrisch -> Loop-Bedingung

    def __init__(self, seed=3):
        super().__init__(seed)
        self._phi = None

    def arc_angle(self, k, t):
        ph = self.OM * t if self._phi is None else self._phi
        return self.PH[k] + ph

    def allow(self, n, t):
        if t < self.T_WAIT:
            return n + 2 <= self.N_HOOK
        e = min(1.0, (t - self.T_WAIT) / (self.T_FULL - self.T_WAIT))
        return n + 2 < self.N_HOOK + (self.N_FULL - self.N_HOOK) * e ** 1.9 + 3

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
        self.full_t = self.loop_t = None
        self.surv, self.glides = [], {}
        self.frames, self.events, self.bounces = [], [], []
        self.omega_log, self.load_log = [], []
        t, phi, load_s = 0.0, 0.0, 0.0
        om = self.OM
        plan = None
        while True:
            for _ in range(self.SUB):
                # --- Drehtempo: zäh unter Last, nach "voll" exakt geplant für den Loop
                if plan is None:
                    om = max(self.OM * self.OM_MIN, self.OM / (1 + self.LOAD_K * load_s))
                else:
                    tf, om_f, om_c, tl = plan
                    if t < tf + 1.0:
                        e = (t - tf) / 1.0
                        s = e * e * (3 - 2 * e)
                        om = om_f + (om_c - om_f) * s
                    elif t < tl - 1.5:
                        om = om_c
                    else:
                        e = min(1.0, (t - (tl - 1.5)) / 1.5)
                        s = e * e * (3 - 2 * e)
                        om = om_c + (self.OM - om_c) * s
                self._phi = phi
                P[:n] += V[:n] * dt
                hit[:n] = 0
                for k, (qx, qy) in enumerate(A.CORNERS):
                    cx_, cy_, a0 = self.arc_geom(k, t)
                    A.arc_hit(P, V, n, r, cx_, cy_, a0, om, hit, qx, qy, ARC_R, k + 1)
                wallnear = ((P[:n, 0] < X0 + r + 2.5) | (P[:n, 0] > X1 - r - 2.5) |
                            (P[:n, 1] < Y0 + r + 2.5) | (P[:n, 1] > Y1 - r - 2.5))
                hit[:n] = np.where(wallnear, hit[:n], 0)
                load = float(np.count_nonzero(hit[:n]))
                load_s += (load - load_s) * min(1.0, dt / 0.12)
                alive = gone[:n] > t
                v_before = V[:n].copy()
                for _ in range(2):
                    _collide(P, V, n, r, cell, head, nxt, gw, gh, x0, y0)
                    A.box_walls(P, V, n, r)
                # Wandabprall erkennen (Normalkomponente dreht sich direkt an der Wand um)
                bx = (((P[:n, 0] < X0 + r + 1) & (v_before[:, 0] < 0) & (V[:n, 0] > 0)) |
                      ((P[:n, 0] > X1 - r - 1) & (v_before[:, 0] > 0) & (V[:n, 0] < 0)))
                by = (((P[:n, 1] < Y0 + r + 1) & (v_before[:, 1] < 0) & (V[:n, 1] > 0)) |
                      ((P[:n, 1] > Y1 - r - 1) & (v_before[:, 1] > 0) & (V[:n, 1] < 0)))
                nb = int(np.count_nonzero((bx | by) & alive))
                if nb:
                    self.bounces.append((t, nb))
                A.keep_speed(V, n, self.SPEED)
                P[:n][~alive] = (-500.0, -500.0)
                if self.full_t is None:
                    for i in np.nonzero(hit[:n])[0]:
                        if n + 2 >= N or t - last[i] < self.GAP or not self.allow(n, t):
                            continue
                        g = gen[i] + 1
                        vx, vy = V[i]
                        ang = math.atan2(vy, vx)
                        sp = math.hypot(vx, vy)
                        px_, py_ = -math.sin(ang) * r * 1.0, math.cos(ang) * r * 1.0
                        ci = P[i].copy()
                        for j, (o, da) in enumerate(((0.0, 0.0), (1.0, 0.38), (-1.0, -0.38))):
                            idx = i if j == 0 else n + j - 1
                            P[idx] = (ci[0] + px_ * o, ci[1] + py_ * o)
                            V[idx] = (math.cos(ang + da) * sp, math.sin(ang + da) * sp)
                            gen[idx] = g
                            last[idx] = born[idx] = t
                        dl, dr_, dt_, db = ci[0] - X0, X1 - ci[0], ci[1] - Y0, Y1 - ci[1]
                        mn = min(dl, dr_, dt_, db)
                        nang[i] = nang[n] = nang[n + 1] = 0.0 if mn in (dl, dr_) else math.pi / 2
                        self.arc_squeeze[int(hit[i]) - 1].append(t)
                        self.events.append((t, "split", {"n": n + 2, "gen": int(g)}))
                        n += 2
                    A.box_walls(P, V, n, r)
                for i, (b, cs, uu, L) in self.glides.items():
                    T = self.loop_t - self.pick_t
                    sd = min(L, (t - self.pick_t) * L / T)
                    u = float(np.interp(sd, cs, uu))
                    P[i] = ((1 - u) ** 3 * b[0] + 3 * (1 - u) ** 2 * u * b[1] + 3 * (1 - u) * u ** 2 * b[2]
                            + u ** 3 * b[3])
                    tg = 3 * (1 - u) ** 2 * (b[1] - b[0]) + 6 * (1 - u) * u * (b[2] - b[1]) + 3 * u ** 2 * (b[3] - b[2])
                    V[i] = tg / (np.hypot(*tg) + 1e-9) * L / T
                phi += om * dt
                t += dt
            self._phi = phi
            self.omega_log.append(om)
            self.load_log.append(load_s)
            if self.full_t is None and n >= self.N_FULL:
                self.full_t = t
                # Loop planen: Phase muss bei loop_t ein Vielfaches von 90° sein, Tempo endet beim Grundtempo
                tmin = self.HOLD + self.POP + self.GLIDE + 0.6
                m = math.floor(phi / self.SYM) + 1
                while True:
                    dphi = m * self.SYM - phi
                    tt = (dphi - 0.5 * om - 0.75 * self.OM) / self.OM + 1.25
                    if tt >= tmin:
                        break
                    m += 1
                tl = round((t + tt) * FPS) / FPS
                tt = tl - t
                om_c = (dphi - 0.5 * om - 0.75 * self.OM) / (tt - 1.25)
                plan = (t, om, om_c, tl)
                self.loop_t = tl
                self.cand = [i for i in range(n) if gen[i] % len(B.RAINBOW) == 0]
                cset = set(self.cand)
                order = [i for i in rng.permutation(n) if i not in cset]
                self.pick_t = self.loop_t - self.GLIDE
                t0 = self.pick_t - self.POP - 0.1
                for j, i in enumerate(order):
                    gone[i] = t0 + self.POP * (j / len(order)) ** 0.8
            if self.full_t is not None and not self.glides and t >= self.pick_t:
                used = set()
                want = self.SPEED * self.GLIDE
                for s in range(m0):
                    best, bs = None, 1e18
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
            if t > 64:
                break
        self.end_phase_err = (phi % self.SYM, self.SYM - phi % self.SYM)
        self.n_frames = len(self.frames)
        return self


def find_starts(seed, lo=0.45, hi=0.95, tries=3000):
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        SP = np.array([(rng.uniform(C.X0 + 70, C.X1 - 70), rng.uniform(C.Y0 + 70, C.Y1 - 70)) for _k in range(3)])
        if min(np.hypot(*(SP[a] - SP[b])) for a in range(3) for b in range(a + 1, 3)) < 120:
            continue
        a = rng.uniform(0, 2 * math.pi, 3)
        SV = np.stack([np.cos(a), np.sin(a)], 1) * ArcTriple.SPEED
        sim = ArcTriple(1)
        sim.STARTS = (SP, SV)
        first = G.probe(sim, hi + 0.2)
        if first is not None and lo <= first <= hi:
            return first, SP, SV
    raise RuntimeError("kein Start gefunden")


def build(seed=5):
    apply_theme()
    first, SP, SV = find_starts(seed)
    sim = ArcTriple(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


class MetalRenderer(C.ShortRenderer2):
    """graue, schwere Metall-Bögen mit Querrillen (wie Schienen), kaum Leuchten."""

    def arc_layer(self, d, angs, alpha, pulse):
        k = A.SS
        for idx, (px, py, a) in enumerate(angs):
            p = pulse[idx] * 0.35

            def bb(dr):
                return [(px - ARC_R - dr) * k, (py - ARC_R - dr) * k, (px + ARC_R + dr) * k, (py + ARC_R + dr) * k]

            def mix(c, w=p):
                return tuple(int(v + (255 - v) * w) for v in c)
            a0, a1 = math.degrees(a), math.degrees(a + SPAN)
            d.arc(bb(0), a0, a1, fill=(28, 30, 36, 255), width=int((ARC_W + 6) * k))
            d.arc(bb(0), a0, a1, fill=mix((128, 133, 143)) + (255,), width=int(ARC_W * k))
            d.arc(bb(-ARC_W * 0.22), a0 + 0.3, a1 - 0.3, fill=mix((205, 210, 218)) + (255,), width=int(3 * k))
            d.arc(bb(ARC_W * 0.28), a0 + 0.3, a1 - 0.3, fill=(88, 92, 102, 255), width=int(2 * k))
            for q in range(1, 40):
                e = a + SPAN * q / 40
                c, s_ = math.cos(e), math.sin(e)
                r0, r1 = ARC_R - ARC_W / 2 + 1.5, ARC_R + ARC_W / 2 - 1.5
                d.line([((px + c * r0) * k, (py + s_ * r0) * k), ((px + c * r1) * k, (py + s_ * r1) * k)],
                       fill=(62, 66, 76, 255), width=int(1.6 * k))
            for e in (a, a + SPAN):
                ex, ey = px + math.cos(e) * ARC_R, py + math.sin(e) * ARC_R
                rr = ARC_W / 2 + 2
                d.ellipse([(ex - rr) * k, (ey - rr) * k, (ex + rr) * k, (ey + rr) * k], fill=mix((150, 155, 165)) + (255,),
                          outline=(28, 30, 36, 255), width=2 * k)
                d.ellipse([(ex - 3) * k, (ey - 3) * k, (ex + 3) * k, (ey + 3) * k], fill=(40, 43, 50, 255))


# ------------------------------------------------------------------ Ton
SR = 44100


def _crack(rng):
    """Crack (kurzer breitbandiger Knack) + glasiges Klingeln mit Teiltönen ~520/1250/2150 Hz."""
    n = int(SR * 0.55)
    tt = np.arange(n) / SR
    f = rng.uniform(0.94, 1.08)
    noise = rng.standard_normal(n)
    click = noise * np.exp(-tt / 0.004) * 0.9
    hiss = np.diff(np.concatenate([[0], noise])) * np.exp(-tt / 0.03) * 0.35
    ring = (np.sin(2 * np.pi * 520 * f * tt) * np.exp(-tt / 0.22) * 0.55 +
            np.sin(2 * np.pi * 1250 * f * tt + 0.7) * np.exp(-tt / 0.15) * 0.32 +
            np.sin(2 * np.pi * 2150 * f * tt + 1.3) * np.exp(-tt / 0.11) * 0.22 +
            np.sin(2 * np.pi * (3400 * f * tt + 6 * np.sin(2 * np.pi * 11 * tt))) * np.exp(-tt / 0.08) * 0.12)
    ring *= np.minimum(1, tt / 0.002)
    return click + hiss + ring


def _chirp():
    """'Piu' beim Wandabprall: schnell fallender Ton 6 kHz -> 180 Hz mit Obertönen."""
    n = int(SR * 0.12)
    tt = np.arange(n) / SR
    f = 180 * (6000 / 180) ** (1 - tt / 0.12)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
    return s * np.exp(-tt / 0.045) * np.minimum(1, tt / 0.001)


def audio(sim, path):
    rng = np.random.default_rng(7)
    dur = sim.n_frames / FPS
    n = int(SR * dur)
    pad = SR
    out = np.zeros(n + pad)
    # 1) mahlendes Motor-Brummen: Pulsfolge ~38 Hz, Tempo folgt den Bögen, bei Last tiefer und lauter
    om = np.interp(np.arange(n + pad) / SR, (np.arange(len(sim.omega_log)) + 1) / FPS,
                   np.array(sim.omega_log), period=dur)
    ld = np.interp(np.arange(n + pad) / SR, (np.arange(len(sim.load_log)) + 1) / FPS,
                   np.array(sim.load_log), period=dur)
    rate = 26 + 14 * om / sim.OM
    ph = np.cumsum(rate) / SR
    buzz = np.clip(np.sin(2 * np.pi * ph), 0, 1) ** 6
    buzz += 0.5 * np.sin(2 * np.pi * np.cumsum(rate * 1.5) / SR)
    from scipy.signal import lfilter
    al = math.exp(-2 * math.pi * 180 / SR)
    for _ in range(3):                                 # Tiefpass ~180 Hz
        buzz = lfilter([1 - al], [1, -al], buzz)
    buzz -= buzz.mean()
    buzz /= np.abs(buzz).max() + 1e-9
    rum = buzz * (0.16 + 0.12 * np.clip(ld / 8, 0, 1))
    xf = int(0.3 * SR)
    w = np.linspace(0, 1, xf)
    rum[:xf] = rum[:xf] * w + rum[n:n + xf] * (1 - w)     # Brummen nahtlos über die Loop-Naht
    # 2) Piu beim Wandabprall (höchstens ~9 pro Sekunde, wird bei vielen Bällen leiser)
    ch = _chirp()
    last = -1.0
    for t, nb in sim.bounces:
        if t >= dur or t - last < 0.11:
            continue
        last = t
        fi = min(sim.n_frames - 1, int(t * FPS))
        cnt = int(np.sum(sim.frames[fi][5] > 0.5))
        g = 0.22 / (1 + cnt / 120)
        i = int(t * SR)
        out[i:i + len(ch)] += ch * g
    # 3) Crack + Klingeln bei JEDER Zerquetschung (überlappt bei vielen)
    for t, kind, _ in sim.events:
        if kind != "split" or t >= dur:
            continue
        s = _crack(rng)
        i = int(t * SR)
        out[i:i + len(s)] += s * 0.5
    tail = out[n:].copy()                               # Nachklang am Ende -> an den Anfang
    out = out[:n]
    out[:len(tail)] += tail
    out += rum[:n]
    out = np.tanh(out * 1.3) * 0.9
    st = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as wv:
        wv.setnchannels(2)
        wv.setsampwidth(2)
        wv.setframerate(SR)
        wv.writeframes((st * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    import sys
    from PIL import Image
    sim, first = build()
    sp = [e for e in sim.events if e[1] == "split"]
    print("erster Crack", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t, "Cracks", len(sp),
          "Phasenfehler", sim.end_phase_err,
          "Bälle", [int(np.sum(sim.frames[min(sim.n_frames - 1, q * FPS)][5] > 0.5)) for q in range(0, 64, 3)],
          "Tempo min/max", round(min(sim.omega_log) / sim.OM, 2), round(max(sim.omega_log) / sim.OM, 2),
          "Gen-Verteilung", np.bincount(sim.frames[int(sim.full_t * FPS)][1].astype(int)))
    if "--preview" in sys.argv:
        R = MetalRenderer(sim)
        ts = [0.3, first + 0.2, 5.0, 20.0, 38.0, sim.loop_t - 0.3]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, x in enumerate(ts):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a14_check.png")
        audio(sim, "output/preview/a14_audio.wav")
    else:
        old_r, old_a = A.ArcRenderer, A.build_audio
        A.ArcRenderer, A.build_audio = MetalRenderer, audio
        try:
            A.render(sim, "output/a14_arc_crush_split3_vertical.mp4")
        finally:
            A.ArcRenderer, A.build_audio = old_r, old_a
        print("output/a14_arc_crush_split3_vertical.mp4")
