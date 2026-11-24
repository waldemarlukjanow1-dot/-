"""Arc Squeeze a28 – Halbkreis-Video (System wie a13/a25/a27) mit:
  * glänzenden Jelly-Bällen (statt Formen)
  * neuer Halbkreis-Optik: dicke, glatte Glas-Röhre mit Farbverlauf (Limette -> Cyan) entlang des Bogens,
    weicher Glanzstreifen, leuchtende Kugel-Enden – auffälliger, aber ruhig/smooth
  * neue Farben "Aurora": dunkles Petrol, eisweißer Rahmen, Jellys in Koralle, Pfirsich, Flieder, Aqua, Zitrone, Rosé
  * Sound: "FAHH" bei jeder Quetschung (assets/fahh.wav), überlappend; Halbkreise versetzt, alle im Uhrzeigersinn
"""
import math
import os
import sys
import wave

import numpy as np
from PIL import Image

from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox6 as F
from . import arcbox7 as G
from .arcbox import ARC_R, ARC_W, SPAN, SS
from .jelly import jelly_sprite

THEME = dict(bg=(3, 15, 18), bg_glow=(8, 46, 50), frame=(215, 245, 255), frame_glow=(120, 220, 255),
             pivot=(10, 40, 44), text=(225, 250, 255))
ARCS = dict(NEON=(90, 240, 160), NEON_HI=(220, 255, 240), NEON_D=(0, 110, 100), NEON_DD=(0, 50, 50))
BALLS = [(255, 95, 80), (255, 175, 110), (185, 140, 255), (40, 215, 230), (250, 235, 70), (255, 120, 170),
         (120, 245, 140), (90, 150, 255)]
C0, C1 = np.array((175, 255, 60), float), np.array((0, 215, 255), float)    # Verlauf entlang des Bogens


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS
    B.BLOOM = 0.25


class AuroraRenderer(C.ShortRenderer2):
    TRAIL = 3

    def __init__(self, sim):
        super().__init__(sim)
        k = SS
        sps = [jelly_sprite(sim.R * k * 1.04, c) for c in BALLS]
        ghs = []
        for sp in sps:
            a = np.asarray(sp).copy()
            a[..., 3] = (a[..., 3] * 0.22).astype(np.uint8)
            ghs.append(Image.fromarray(a, "RGBA"))
        for g in self.GLOW_LEVELS:
            self.sprites[g] = sps
            self.ghost[g] = ghs

    def arc_layer(self, d, angs, alpha, pulse):
        k = SS
        seg = 36
        W = ARC_W + 6                                   # optisch etwas dicker, rund und weich
        for idx, (px, py, a) in enumerate(angs):
            p = pulse[idx] * 0.35

            def bb(dr):
                return [(px - ARC_R - dr) * k, (py - ARC_R - dr) * k, (px + ARC_R + dr) * k, (py + ARC_R + dr) * k]

            def col(u, light=0.0, dark=0.0):
                c = C0 + (C1 - C0) * u
                c = c + (255 - c) * min(1.0, light + p)
                c = c * (1 - dark)
                return tuple(int(v) for v in np.clip(c, 0, 255)) + (255,)
            a0, a1 = math.degrees(a), math.degrees(a + SPAN)
            for q in range(seg):                        # dunkler Rand, Körper, Glanz – jeweils mit Verlauf
                u = (q + 0.5) / seg
                s0 = a0 + (a1 - a0) * q / seg
                s1 = a0 + (a1 - a0) * (q + 1) / seg + 0.4
                d.arc(bb(0), s0, s1, fill=col(u, dark=0.55), width=int((W + 4) * k))
            for q in range(seg):
                u = (q + 0.5) / seg
                s0 = a0 + (a1 - a0) * q / seg
                s1 = a0 + (a1 - a0) * (q + 1) / seg + 0.4
                d.arc(bb(0), s0, s1, fill=col(u), width=int(W * k))
                d.arc(bb(-W * 0.18), s0, s1, fill=col(u, light=0.45), width=int(W * 0.32 * k))
                d.arc(bb(-W * 0.26), s0, s1, fill=col(u, light=0.85), width=max(1, int(2.2 * k)))
                d.arc(bb(W * 0.3), s0, s1, fill=col(u, dark=0.25), width=max(1, int(2.5 * k)))
            for e, u in ((a, 0.0), (a + SPAN, 1.0)):     # leuchtende Kugel-Enden
                ex, ey = px + math.cos(e) * ARC_R, py + math.sin(e) * ARC_R
                for rr, c in ((W * 0.75, col(u, dark=0.55)), (W * 0.62, col(u)), (W * 0.3, col(u, light=0.7))):
                    d.ellipse([(ex - rr) * k, (ey - rr) * k, (ex + rr) * k, (ey + rr) * k], fill=c)
                d.ellipse([(ex - W * 0.38) * k, (ey - W * 0.42) * k, (ex - W * 0.08) * k, (ey - W * 0.18) * k],
                          fill=(255, 255, 255, 220))


SOUND = "assets/fahh.wav"


def audio(sim, path):
    if os.path.exists(SOUND):
        F.audio(sim, path, sample=SOUND, vol=0.6)
    else:                                               # Platzhalter: Stille, Sound wird später eingesetzt
        n = int(44100 * sim.n_frames / A.FPS)
        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(np.zeros((n, 2), np.int16).tobytes())


def build():
    G.apply_theme = apply_theme
    G.ArcLoopWait.PH = (0.3, 0.3 + 2.0, 0.3 + 3.3, 0.3 + 5.0)       # versetzt, alle im Uhrzeigersinn
    return G.build(seed=59)


if __name__ == "__main__":
    sim, first = build()
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"))
    if "--preview" in sys.argv:
        R = AuroraRenderer(sim)
        out = Image.new("RGB", (1080, 640))
        for i, x in enumerate((2.0, 15.0, 30.0)):
            out.paste(R.frame(int(x * A.FPS)).resize((360, 640)), (i * 360, 0))
        out.save("output/preview/a28_check.png")
        R.frame(int(20 * A.FPS)).crop((150, 300, 900, 1100)).save("output/preview/a28_zoom.png")
    elif "--audio-only" in sys.argv:                    # nur Ton neu bauen und ins fertige Video einsetzen
        import subprocess
        audio(sim, "output/preview/a28.wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "output/a28_aurora_jelly_vertical.mp4", "-i",
                        "output/preview/a28.wav", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "192k", "-af", "loudnorm=I=-17:TP=-2:LRA=11", "-shortest", "-movflags", "+faststart",
                        "output/a28_aurora_jelly_fahh_vertical.mp4"], check=True)
        print("output/a28_aurora_jelly_fahh_vertical.mp4")
    else:
        A.ArcRenderer, A.build_audio = AuroraRenderer, audio
        A.render(sim, "output/a28_aurora_jelly_vertical.mp4")
        print("output/a28_aurora_jelly_vertical.mp4")
