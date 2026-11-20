"""Arc Squeeze a24 (wie a13): Deep-Meow-Sound bei jeder Quetschung (nur dort, überlappend erlaubt).
Spannungsstart wie a12 (1 Ball, erste Quetschung nach ~4 s).
Farbwelt: tiefes Schwarz, Mint-Rahmen, orange Neon-Bögen, Bälle in Pink, Cyan, Zitrone, Flieder, Weiß, Limette."""
from functools import partial
from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox6 as F
from . import arcbox7 as G

THEME = dict(bg=(8, 8, 10), bg_glow=(22, 26, 26), frame=(160, 255, 215), frame_glow=(40, 230, 160),
             pivot=(10, 30, 22), text=(230, 255, 245))
ARCS = dict(NEON=(255, 110, 20), NEON_HI=(255, 215, 170), NEON_D=(180, 60, 0), NEON_DD=(80, 25, 0))
BALLS = [(255, 60, 160), (40, 220, 255), (255, 240, 60), (190, 150, 255), (250, 250, 255), (150, 255, 60),
         (255, 120, 90), (60, 140, 255)]


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
    sim, first = G.build(seed=41)
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"),
          "erste", [round(e[0], 2) for e in sim.events if e[1] == "split"][:6])
    if "--preview" in sys.argv:
        R = C.ShortRenderer2(sim)
        ts = [2.0, first + 0.3, 25.0]
        out = Image.new("RGB", (1080, 640))
        for i, x in enumerate(ts):
            out.paste(R.frame(int(x * A.FPS)).resize((360, 640)), (i * 360, 0))
        out.save("output/preview/a24_check.png")
    else:
        old_r, old_a = A.ArcRenderer, A.build_audio
        A.ArcRenderer, A.build_audio = C.ShortRenderer2, partial(F.audio, sample="assets/meow.wav", vol=0.55)
        try:
            A.render(sim, "output/a24_arc_squeeze_meow_vertical.mp4")
        finally:
            A.ArcRenderer, A.build_audio = old_r, old_a
        print("output/a24_arc_squeeze_meow_vertical.mp4")
