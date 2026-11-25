"""Arc Squeeze a31 – wie die Viral-Version a30 (Hook ~1 s, ~45 s, Glanz-Welle am Ende, nahtloser Loop),
aber mit angenehmem PLOPP statt FAHH und neuen Farben "Sakura Night".
Plopp: weicher, runder Tropfen-Ton (kurzer Sinus mit schnellem Tonhöhenfall), bei jeder Quetschung;
Tonhöhe steigt mit der Ballzahl entlang einer Pentatonik (klingt immer harmonisch, auch wenn sich viele überlagern)."""
import math
import sys
import wave

import numpy as np
from PIL import Image

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox7 as G
from . import arcbox18 as H18
from . import arcbox20 as V
from .arcbox import FPS

THEME = dict(bg=(14, 7, 20), bg_glow=(46, 22, 58), frame=(255, 185, 215), frame_glow=(255, 110, 170),
             pivot=(50, 20, 40), text=(255, 238, 246))
ARCS = dict(NEON=(240, 170, 230), NEON_HI=(255, 235, 248), NEON_D=(120, 60, 130), NEON_DD=(50, 20, 60))
BALLS = [(255, 140, 180), (190, 160, 255), (130, 240, 200), (140, 200, 255), (255, 235, 130), (255, 190, 150),
         (255, 255, 255), (230, 120, 255)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS
    B.BLOOM = 0.25
    H18.BALLS[:] = BALLS
    H18.C0, H18.C1 = np.array((255, 215, 235), float), np.array((160, 120, 255), float)


SR = 44100
PENTA = [0, 2, 4, 7, 9]                     # Dur-Pentatonik


def _plop(freq, rng):
    n = int(SR * 0.16)
    t = np.arange(n) / SR
    f = freq * (1 + 0.9 * np.exp(-t / 0.012))           # kurzer "Tropfen"-Abfall der Tonhöhe
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.18 * np.sin(2 * ph)
    env = (1 - np.exp(-t / 0.002)) * np.exp(-t / 0.05)
    return s * env


def audio(sim, path):
    rng = np.random.default_rng(5)
    dur = sim.n_frames / FPS
    n = int(SR * dur)
    out = np.zeros(n + SR)
    cache = {}
    for t, kind, dd in sim.events:
        if kind != "split" or t >= dur:
            continue
        q = min(1.0, dd["n"] / sim.N_FULL)
        step = int(q ** 0.8 * 14) + int(rng.integers(0, 3))  # steigt mit der Ballzahl, leicht variiert
        semi = 12 * (step // 5) + PENTA[step % 5]
        freq = 330 * 2 ** (semi / 12)
        s = cache.get(semi)
        if s is None:
            s = cache[semi] = _plop(freq, rng)
        i = int(t * SR)
        g = 0.55 / (1 + q * 3)                              # bei vielen Plopps leiser -> bleibt angenehm
        out[i:i + len(s)] += s * g
    tail = out[n:]
    out = out[:n]
    out[:len(tail)] += tail
    out = np.tanh(out * 1.1) * 0.85
    st = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


def build():
    apply_theme()
    W = G.ArcLoopWait
    W.PH = (0.3, 0.3 + 1.4, 0.3 + 3.4, 0.3 + 4.3)
    W.R = 12.0
    W.N_FULL = 1300
    W.CAP = 1900
    W.GAP = 0.2
    W.T_FULL = 38.0
    W.HOLD = 1.4
    first, SP, SV = G.find_starts(71, lo=1.0, hi=1.5)
    sim = W(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


if __name__ == "__main__":
    sim, first = build()
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"))
    if "--preview" in sys.argv:
        R = V.ViralRenderer(sim)
        out = Image.new("RGB", (1080, 640))
        for i, x in enumerate((1.0, 25.0, sim.full_t + 0.5)):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), (i * 360, 0))
        out.save("output/preview/a31_check.png")
        audio(sim, "output/preview/a31.wav")
    else:
        A.ArcRenderer, A.build_audio = V.ViralRenderer, audio
        A.render(sim, "output/a31_plopp_sakura_vertical.mp4")
        print("output/a31_plopp_sakura_vertical.mp4")
