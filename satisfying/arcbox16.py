"""Arc Squeeze a25 (wie a13, Halbkreise mit unterschiedlicher Startstellung): Deep-Meow-Sound bei jeder Quetschung (nur dort, überlappend erlaubt).
Spannungsstart wie a12 (1 Ball, erste Quetschung nach ~4 s).
Farbwelt: Tiefsee-Blau, korallroter Rahmen, türkise Neon-Bögen, Bälle in Sonnenuntergangsfarben."""
from functools import partial
from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox6 as F
from . import arcbox7 as G

THEME = dict(bg=(2, 10, 24), bg_glow=(6, 34, 60), frame=(255, 140, 120), frame_glow=(255, 80, 60),
             pivot=(40, 12, 10), text=(255, 236, 228))
ARCS = dict(NEON=(0, 230, 210), NEON_HI=(190, 255, 245), NEON_D=(0, 130, 120), NEON_DD=(0, 55, 50))
BALLS = [(255, 90, 60), (255, 170, 50), (255, 225, 90), (255, 110, 150), (200, 80, 200), (255, 245, 230),
         (255, 60, 90), (255, 200, 140)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS


if __name__ == "__main__":
    import sys
    from PIL import Image
    G.apply_theme = apply_theme
    # alle im Uhrzeigersinn, gleiche Geschwindigkeit, aber jede startet in einer anderen Stellung
    G.ArcLoopWait.PH = (0.3, 0.3 + 2.45, 0.3 + 3.55, 0.3 + 5.35)
    sim, first = G.build(seed=47)
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"),
          "erste", [round(e[0], 2) for e in sim.events if e[1] == "split"][:6])
    if "--preview" in sys.argv:
        R = C.ShortRenderer2(sim)
        ts = [2.0, first + 0.3, 25.0]
        out = Image.new("RGB", (1080, 640))
        for i, x in enumerate(ts):
            out.paste(R.frame(int(x * A.FPS)).resize((360, 640)), (i * 360, 0))
        out.save("output/preview/a25_check.png")
    else:
        old_r, old_a = A.ArcRenderer, A.build_audio
        A.ArcRenderer, A.build_audio = C.ShortRenderer2, partial(F.audio, sample="assets/meow.wav", vol=0.55)
        try:
            A.render(sim, "output/a25_arc_squeeze_meow_versetzt_vertical.mp4")
        finally:
            A.ArcRenderer, A.build_audio = old_r, old_a
        print("output/a25_arc_squeeze_meow_versetzt_vertical.mp4")
