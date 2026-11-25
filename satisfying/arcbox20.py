"""Arc Squeeze a30 – "Viral-Version": gleiches Prinzip (Halbkreise quetschen an der Wand, jede Quetschung verdoppelt),
aber mit den Hebeln, die Watchtime / Rewatches treiben:
  1. Hook in der 1. Sekunde: erste Quetschung nach ~1,2 s statt 4 s (kein Wegscrollen am Anfang)
  2. Steigende Tonhöhe: jedes "FAHH" wird mit wachsender Ballzahl höher -> hörbarer Spannungsaufbau bis zum Finale
  3. Kürzer & dichter: voll nach ~38 s (Video ~43 s) -> höhere Abschlussrate, Loop zählt als Rewatch
  4. Belohnung am Ende: wenn das Quadrat voll ist, läuft eine Glanz-Welle über alle Bälle (kein Ring, kein Text)
  5. Neuer Look: Mitternachtsblau, violetter Rahmen, Glas-Röhren-Halbkreise mit Verlauf Cyan -> Pink, Jelly-Bälle
Halbkreise versetzt, alle im Uhrzeigersinn; nur die Ballzahl als Schrift; nahtloser Loop.
"""
import math
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox7 as G
from . import arcbox18 as H18
from .arcbox import FPS

THEME = dict(bg=(5, 7, 22), bg_glow=(16, 22, 66), frame=(185, 130, 255), frame_glow=(140, 70, 255),
             pivot=(30, 16, 60), text=(240, 235, 255))
ARCS = dict(NEON=(120, 160, 255), NEON_HI=(225, 235, 255), NEON_D=(40, 60, 150), NEON_DD=(15, 20, 60))
BALLS = [(255, 70, 120), (255, 190, 50), (60, 230, 160), (60, 200, 255), (190, 120, 255), (255, 255, 255),
         (255, 130, 60), (120, 255, 90)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS
    B.BLOOM = 0.28
    H18.BALLS[:] = BALLS
    H18.C0, H18.C1 = np.array((0, 230, 255), float), np.array((255, 70, 200), float)


class ViralRenderer(H18.AuroraRenderer):
    WAVE = 1.1                                          # Dauer der Glanz-Welle

    def frame(self, fi):
        im = super().frame(fi)
        sim = self.sim
        if sim.full_t is None:
            return im
        t = (fi + 1) / FPS
        e = (t - sim.full_t - 0.1) / self.WAVE
        if not 0 <= e <= 1:
            return im
        Pn, scale = sim.frames[fi][0], sim.frames[fi][5]
        xw = C.X0 - 120 + (C.X1 - C.X0 + 240) * e        # Welle läuft diagonal von links oben nach rechts unten
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov, "RGBA")
        r = sim.R
        for i in np.nonzero(scale > 0.5)[0]:
            x, y = Pn[i]
            dd = (x + (y - C.Y0) * 0.6) - xw
            w = math.exp(-(dd / 70) ** 2)
            if w < 0.04:
                continue
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, int(200 * w)))
        im = im.convert("RGBA")
        im.alpha_composite(ov)
        return im.convert("RGB")


def audio(sim, path, sample="assets/fahh.wav", vol=0.6):
    """FAHH bei jeder Quetschung; Tonhöhe steigt mit der Ballzahl (0,85x -> 1,6x)."""
    SR = 44100
    with wave.open(sample) as w:
        smp = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    smp = smp * np.minimum(1, np.arange(len(smp)) / (SR * 0.003))
    dur = sim.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + len(smp) * 2)
    cache = {}
    for t, kind, dd in sim.events:
        if kind != "split" or t >= dur:
            continue
        q = min(1.0, dd["n"] / sim.N_FULL)
        f = round(0.85 + 0.75 * q ** 0.75, 2)
        s = cache.get(f)
        if s is None:
            L = int(len(smp) / f)
            s = np.interp(np.arange(L) * f, np.arange(len(smp)), smp)
            cache[f] = s
        i = int(t * SR)
        out[i:i + len(s)] += s * vol
    tail = out[n:]
    out = out[:n]
    out[:len(tail)] += tail
    out = np.tanh(out * 1.2) * 0.9
    st = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def build():
    apply_theme()
    W = G.ArcLoopWait
    W.PH = (0.3, 0.3 + 1.9, 0.3 + 2.9, 0.3 + 4.6)       # versetzt, alle im Uhrzeigersinn
    W.R = 12.0
    W.N_FULL = 1300
    W.CAP = 1900
    W.GAP = 0.2
    W.T_FULL = 38.0
    W.HOLD = 1.4                                         # Zeit für die Glanz-Welle, bevor es abbaut
    first, SP, SV = G.find_starts(67, lo=1.0, hi=1.5)
    sim = W(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


if __name__ == "__main__":
    sim, first = build()
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"),
          "Bälle", [len(sim.frames[min(sim.n_frames - 1, q * FPS)][0]) for q in range(0, 50, 4)])
    if "--preview" in sys.argv:
        R = ViralRenderer(sim)
        ts = [1.0, 12.0, 25.0, sim.full_t + 0.4, sim.full_t + 0.7, sim.loop_t - 0.3]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, x in enumerate(ts):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a30_check.png")
    else:
        A.ArcRenderer, A.build_audio = ViralRenderer, audio
        A.render(sim, "output/a30_viral_fahh_vertical.mp4")
        print("output/a30_viral_fahh_vertical.mp4")
