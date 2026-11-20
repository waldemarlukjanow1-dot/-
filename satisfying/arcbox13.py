"""Arc Squeeze a22 – "Bomben": spannenderes Prinzip.
  * Grundsystem wie a14: jede Zerquetschung teilt in 3, schwergängige Arme, synchron im Uhrzeigersinn
  * NEU: ab und zu entsteht beim Teilen eine BOMBE (schwarzer Ball mit blinkendem roten Licht, leises Zischen).
    Wird eine Bombe an der Wand zerquetscht, explodiert sie: alle Bälle in der Nähe platzen (Welle von innen
    nach außen), die weiter entfernten werden weggeschleudert, der Zähler fällt. Bomben im Radius gehen mit
    hoch -> Kettenreaktion. Spannung: schafft es die Box trotzdem, voll zu werden?
  * in den letzten Sekunden vor "voll" keine neuen Bomben -> die Box wird sicher voll, dann nahtloser Loop
  * Design "Warnzone": fast schwarzer Hintergrund, Rahmen in Warngelb, Arme mit gelb-schwarzen Warnstreifen,
    Bälle in kühlen Bonbonfarben (Mint, Himmelblau, Flieder, Rosa, Weiß)
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox9 as H
from .arcbox import ARC_R, ARC_W, SPAN, FPS, W1, H1

THEME = dict(bg=(13, 12, 14), bg_glow=(26, 24, 22), frame=(255, 205, 0), frame_glow=(255, 150, 0),
             pivot=(20, 18, 14), text=(255, 236, 170))
ARCS = dict(NEON=(200, 160, 20), NEON_HI=(255, 230, 120), NEON_D=(90, 70, 0), NEON_DD=(40, 30, 0))
BALLS = [(90, 245, 190), (90, 190, 255), (190, 150, 255), (255, 140, 200), (245, 248, 255)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS
    B.BLOOM = 0.3


class BombBox(H.ArcTriple):
    N_FULL = 900
    T_FULL = 45.0
    FORCE_FULL = 54.0
    BOMB_FROM = 11.0          # ab hier können Bomben entstehen
    BOMB_UNTIL = 37.0         # danach keine neuen Bomben mehr (Finale: Box wird voll)
    BOMB_AT = [12.5, 19.0, 24.5, 24.9, 30.0, 30.4, 30.8]   # ab diesen Zeiten wird die nächste Teilung eine Bombe
    BOMB_MAX = 3
    CAP = 3200
    R_KILL = 160.0
    FUSE = 1.5                # so lange ist eine neue Bombe noch nicht scharf
    LIFE = 5.5                # Zeitzünder
    R_PUSH = 260.0

    def run(self):
        self.bombs = set()          # aktive Bomben (Index)
        self.pending = []           # (Zeit, Index) verzögerte Kettenreaktion
        self.bomb_log = []
        self.booms = []             # (t, x, y, getötet)
        self.bomb_q = list(self.BOMB_AT)
        self.bomb_born = {}
        return super().run()

    def excluded(self):
        return set(self.bombs)

    def after_split(self, i, a, b, t):
        if (self.bomb_q and t >= self.bomb_q[0] and t <= self.BOMB_UNTIL
                and len(self.bombs) < self.BOMB_MAX):
            self.bomb_q.pop(0)
            self.bombs.add(b)
            self.bomb_born[b] = t

    def pre_crush(self, i, t):
        if i in self.bombs:
            if t - self.bomb_born.get(i, -9) >= self.FUSE:
                self._explode(i, t)
            return True
        return False

    def _explode(self, b, t):
        P, V, gen, gone, last, born = self._ctx
        n = self._n
        if b not in self.bombs:
            return
        self.bombs.discard(b)
        c = P[b].copy()
        gone[b] = t
        alive = np.nonzero(gone[:n] > t)[0]
        d = np.hypot(P[alive, 0] - c[0], P[alive, 1] - c[1])
        killed = 0
        for j, dj in zip(alive, d):
            if j in self.glides:
                continue
            if dj < self.R_KILL:
                if j in self.bombs:                    # Kettenreaktion
                    self.pending.append((t + 0.12 + dj / 1500, j))
                else:
                    gone[j] = t + 0.03 + dj / 650      # Platz-Welle von innen nach außen
                    killed += 1
            elif dj < self.R_PUSH:
                u = (P[j] - c) / (dj + 1e-9)
                V[j] += u * 900 * (1 - dj / self.R_PUSH)
        self.booms.append((t, float(c[0]), float(c[1]), killed))
        self.events.append((t, "boom", {"killed": killed}))

    def substep(self, t):
        for b in [b for b in self.bombs if t - self.bomb_born.get(b, t) >= self.LIFE]:
            self._explode(b, t)                 # Zeitzünder: spätestens nach LIFE Sekunden
        if self.pending:
            due = [p for p in self.pending if p[0] <= t]
            self.pending = [p for p in self.pending if p[0] > t]
            for _, j in due:
                self._explode(j, t)

    def on_frame(self, t, n):
        gone = self._ctx[3]
        if self.full_t is not None:            # beim Loop-Ende platzen Bomben einfach mit
            self.bombs = {b for b in self.bombs if gone[b] > t + 0.12}
        self.bomb_log.append(np.array(sorted(b for b in self.bombs if gone[b] > t), np.int64))


class HazardRenderer(H.MetalRenderer):
    def arc_layer(self, d, angs, alpha, pulse):
        k = A.SS
        for idx, (px, py, a) in enumerate(angs):
            p = pulse[idx] * 0.3

            def bb(dr):
                return [(px - ARC_R - dr) * k, (py - ARC_R - dr) * k, (px + ARC_R + dr) * k, (py + ARC_R + dr) * k]

            def mix(c):
                return tuple(int(v + (255 - v) * p) for v in c) + (255,)
            a0, a1 = math.degrees(a), math.degrees(a + SPAN)
            d.arc(bb(0), a0, a1, fill=(18, 16, 12, 255), width=int((ARC_W + 6) * k))
            seg = 22
            for q in range(seg):                       # gelb-schwarze Warnstreifen
                s0 = a0 + (a1 - a0) * q / seg
                s1 = a0 + (a1 - a0) * (q + 1) / seg
                col = (255, 200, 0) if q % 2 == 0 else (64, 60, 54)
                d.arc(bb(0), s0, s1 + 0.2, fill=mix(col), width=int(ARC_W * k))
            d.arc(bb(-ARC_W * 0.3), a0 + 0.3, a1 - 0.3, fill=(255, 255, 255, 60), width=int(2 * k))
            for e in (a, a + SPAN):
                ex, ey = px + math.cos(e) * ARC_R, py + math.sin(e) * ARC_R
                rr = ARC_W / 2 + 2
                d.ellipse([(ex - rr) * k, (ey - rr) * k, (ex + rr) * k, (ey + rr) * k], fill=(60, 58, 54, 255),
                          outline=(18, 16, 12, 255), width=2 * k)
                d.ellipse([(ex - 3.5) * k, (ey - 3.5) * k, (ex + 3.5) * k, (ey + 3.5) * k], fill=(255, 200, 0, 255))

    def frame(self, fi):
        sim = self.sim
        t = (fi + 1) / FPS
        im = super().frame(fi)
        Pn = sim.frames[fi][0]
        scale = sim.frames[fi][5]
        d = ImageDraw.Draw(im, "RGBA")
        r = sim.R * 1.55
        for b in sim.bomb_log[fi]:
            if b >= len(Pn) or scale[b] < 1:
                continue
            x, y = Pn[b]
            d.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6], fill=(255, 50, 30, 80))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(22, 22, 26, 255), outline=(70, 70, 78, 255), width=2)
            d.ellipse([x - r * 0.55, y - r * 0.6, x - r * 0.1, y - r * 0.2], fill=(120, 120, 130, 170))
            age = t - sim.bomb_born.get(int(b), t)
            ph = 2 * math.pi * (2.0 * age + 0.9 * age * age)         # blinkt immer schneller
            on = math.sin(ph) > 0
            lc = (255, 40, 30, 255) if on else (90, 20, 15, 255)
            if on:
                d.ellipse([x - 14, y - 14, x + 14, y + 14], fill=(255, 60, 40, 70))
            d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=lc)
        # Wackeln nach Explosionen
        sh = 0.0
        for tb, _, _, kl in sim.booms:
            e = t - tb
            if 0 <= e < 0.35:
                sh = max(sh, (1 - e / 0.35) ** 2 * min(14.0, 5 + kl * 0.25))
        if sh > 0.5:
            ox = int(round(math.sin(t * 97) * sh))
            oy = int(round(math.cos(t * 83) * sh))
            bg = Image.new("RGB", im.size, THEME["bg"])
            bg.paste(im, (ox, oy))
            im = bg
        return im


def _boom():
    SR = H.SR
    n = int(SR * 0.9)
    tt = np.arange(n) / SR
    rng = np.random.default_rng(3)
    f = 95 * np.exp(-tt / 0.18) + 38
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.35)
    noise = rng.standard_normal(n)
    from scipy.signal import lfilter
    al = math.exp(-2 * math.pi * 900 / SR)
    rumble = lfilter([1 - al], [1, -al], noise) * np.exp(-tt / 0.25) * 3.0
    crack = noise * np.exp(-tt / 0.012) * 0.6
    return (body * 0.9 + rumble + crack) * np.minimum(1, tt / 0.002)


def audio(sim, path):
    import wave
    H.audio(sim, path)                           # Brummen, Piu, Crack
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2)[:, 0].astype(np.float64) / 32768
    a = np.arctanh(np.clip(a / 0.9, -0.999, 0.999)) / 1.3   # Soft-Clip rückgängig, neu mischen
    SR = H.SR
    n = len(a)
    dur = n / SR
    out = np.zeros(n + SR)
    out[:n] = a
    bm = _boom()
    for t, kind, dd in sim.events:
        if kind == "boom" and t < dur:
            i = int(t * SR)
            out[i:i + len(bm)] += bm * (0.55 + 0.25 * min(1.0, dd["killed"] / 40))
            # Platz-Geräusche der weggesprengten Bälle (kurze Knackser, verteilt über die Welle)
            rng = np.random.default_rng(int(t * 1000))
            for _ in range(min(25, dd["killed"])):
                j = i + int(rng.uniform(0.03, 0.2) * SR)
                L = int(0.025 * SR)
                out[j:j + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (0.004 * SR)) * 0.12
    # leises Zischen der Lunte, solange Bomben im Spiel sind
    nb = np.array([len(x) for x in sim.bomb_log], np.float64)
    env = np.interp(np.arange(n) / SR, (np.arange(len(nb)) + 1) / FPS, np.minimum(nb, 2) / 2)
    from scipy.signal import lfilter
    hs = np.random.default_rng(5).standard_normal(n)
    hs = lfilter([1, -0.97], [1], hs)
    crackle = (np.random.default_rng(6).random(n) < 0.0015) * np.random.default_rng(7).standard_normal(n) * 2
    out[:n] += (hs * 0.02 + crackle * 0.06) * env
    tail = out[n:]
    out = out[:n]
    out[:len(tail)] += tail
    out = np.tanh(out * 1.3) * 0.9
    st = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def build(seed=17):
    apply_theme()
    first, SP, SV = H.find_starts(seed, cls=BombBox)
    sim = BombBox(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


if __name__ == "__main__":
    import sys
    sim, first = build()
    vis = [int(np.sum(sim.frames[min(sim.n_frames - 1, q * FPS)][5] > 0.5)) for q in range(0, 60, 3)]
    print("erster Crack", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t, "Bälle", vis)
    print("Explosionen", [(round(t, 1), k) for t, _, _, k in sim.booms])
    if "--preview" in sys.argv:
        R = HazardRenderer(sim)
        tb = sim.booms[len(sim.booms) // 2][0] if sim.booms else 25.0
        bt = next((i / FPS for i, b in enumerate(sim.bomb_log) if len(b)), 20.0)
        ts = [0.5, bt + 0.5, tb - 0.05, tb + 0.12, sim.full_t, sim.loop_t - 0.3]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, x in enumerate(ts):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a22_check.png")
    else:
        A.ArcRenderer, A.build_audio = HazardRenderer, audio
        A.render(sim, "output/a22_bomben_vertical.mp4")
        print("output/a22_bomben_vertical.mp4")
