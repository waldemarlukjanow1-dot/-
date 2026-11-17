"""Arc Squeeze a13: Deep-Meow-Sound bei jeder Quetschung (nur dort, überlappend erlaubt).
Spannungsstart wie a12 (1 Ball, erste Quetschung nach ~4 s).
Farbwelt: Mitternachtsblau, Pfirsich-Rahmen, gelbe Bögen, Bälle in Pastell/Bonbon (Katzen-Vibe)."""
from functools import partial
from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox6 as F
from . import arcbox7 as G

THEME = dict(bg=(5, 8, 24), bg_glow=(16, 24, 60), frame=(255, 215, 190), frame_glow=(255, 140, 110),
             pivot=(40, 25, 20), text=(255, 235, 220))
ARCS = dict(NEON=(255, 225, 30), NEON_HI=(255, 250, 200), NEON_D=(190, 140, 0), NEON_DD=(80, 55, 0))
BALLS = [(255, 150, 190), (255, 190, 120), (150, 220, 255), (200, 160, 255), (140, 255, 190), (255, 255, 255),
         (255, 110, 120), (120, 170, 255)]


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
    sim, first = G.build(seed=33)
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"),
          "erste", [round(e[0], 2) for e in sim.events if e[1] == "split"][:6])
    if "--preview" in sys.argv:
        R = C.ShortRenderer2(sim)
        ts = [2.0, first + 0.3, 25.0]
        out = Image.new("RGB", (1080, 640))
        for i, x in enumerate(ts):
            out.paste(R.frame(int(x * A.FPS)).resize((360, 640)), (i * 360, 0))
        out.save("output/preview/a13_check.png")
    else:
        old_r, old_a = A.ArcRenderer, A.build_audio
        A.ArcRenderer, A.build_audio = C.ShortRenderer2, partial(F.audio, sample="assets/meow.wav", vol=0.55)
        try:
            A.render(sim, "output/a13_arc_squeeze_meow_vertical.mp4")
        finally:
            A.ArcRenderer, A.build_audio = old_r, old_a
        print("output/a13_arc_squeeze_meow_vertical.mp4")
