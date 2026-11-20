"""Arc Squeeze a23 – wieder das bewährte System (wie a14/a21):
4 schwergängige Metall-Halbkreise im Uhrzeigersinn, jede Quetschung an der Wand teilt den Ball in 3
(neue Bälle entstehen nur dort, wo gequetscht wurde), Hook -> Spannung -> Explosion, nahtloser Loop.
Neu: Quetsch-Sound = "HUSS!" (Nutzer-Sample), Bälle als glänzende Jellys, neue Farben (Kirsch-Look)."""
import numpy as np
from PIL import Image

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox9 as H
from .arcbox import FPS, SS
from .jelly import jelly_sprite

THEME = dict(bg=(16, 5, 12), bg_glow=(52, 12, 32), frame=(255, 95, 135), frame_glow=(255, 40, 100),
             pivot=(30, 8, 20), text=(255, 230, 238))
BALLS = [(235, 20, 60), (255, 150, 20), (150, 230, 40), (40, 120, 255), (150, 60, 230), (0, 210, 200)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in H.ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS
    B.BLOOM = 0.22


class JellyArcRenderer(H.MetalRenderer):
    TRAIL = 2

    def __init__(self, sim):
        super().__init__(sim)
        sps = [jelly_sprite(sim.R * SS * 1.04, c) for c in BALLS]
        ghs = []
        for sp in sps:
            a = np.asarray(sp).copy()
            a[..., 3] = (a[..., 3] * 0.22).astype(np.uint8)
            ghs.append(Image.fromarray(a, "RGBA"))
        for g in self.sprites:
            self.sprites[g] = sps
            self.ghost[g] = ghs


_BANK = []


def _huss(rng):
    import wave
    if not _BANK:
        for j in range(1, 7):
            with wave.open(f"assets/huss_{j}.wav") as w:
                _BANK.append(np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768)
    return _BANK[rng.integers(len(_BANK))] * 1.5


def audio(sim, path):
    H._crack = _huss                     # bei jeder Quetschung "HUSS!" statt Crack
    H.audio(sim, path)


def build(seed=23):
    apply_theme()
    first, SP, SV = H.find_starts(seed)
    sim = H.ArcTriple(3)
    sim.STARTS = (SP, SV)
    sim.run()
    return sim, first


if __name__ == "__main__":
    import sys
    sim, first = build()
    print("erster Crack", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Bälle", [int(np.sum(sim.frames[min(sim.n_frames - 1, q * FPS)][5] > 0.5)) for q in range(0, 60, 4)])
    if "--preview" in sys.argv:
        R = JellyArcRenderer(sim)
        ts = [0.5, 8.0, 22.0, 38.0, sim.full_t + 1, sim.loop_t - 0.3]
        out = Image.new("RGB", (360 * 3, 640 * 2))
        for i, x in enumerate(ts):
            out.paste(R.frame(min(sim.n_frames - 1, int(x * FPS))).resize((360, 640)), ((i % 3) * 360, (i // 3) * 640))
        out.save("output/preview/a23_check.png")
        R.frame(int(30 * FPS)).crop((250, 600, 750, 1100)).save("output/preview/a23_zoom.png")
    else:
        A.ArcRenderer, A.build_audio = JellyArcRenderer, audio
        A.render(sim, "output/a23_jelly_huss_vertical.mp4")
        print("output/a23_jelly_huss_vertical.mp4")
