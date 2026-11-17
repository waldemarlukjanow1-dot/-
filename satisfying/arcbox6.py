"""Arc Squeeze Rizz #2: Ton NUR bei der Quetschung – dafür bei jeder einzelnen (darf sich überschneiden, Chaos),
neue Farbwelt: fast schwarzes Violett, lila Rahmen, elektrisch-blaue Bögen, warme Bälle."""
from . import arcbox as A
from . import arcbox2 as B
from . import arcbox3 as C
from . import arcbox4 as D
from . import arcbox5 as E

THEME = dict(bg=(10, 4, 18), bg_glow=(26, 10, 40), frame=(220, 200, 255), frame_glow=(150, 90, 255),
             pivot=(30, 12, 50), text=(240, 232, 255))
ARCS = dict(NEON=(40, 160, 255), NEON_HI=(200, 230, 255), NEON_D=(10, 70, 170), NEON_DD=(5, 30, 80))
BALLS = [(255, 70, 50), (255, 150, 40), (255, 210, 60), (255, 120, 150), (230, 80, 255), (255, 245, 235),
         (120, 255, 200), (255, 95, 95)]


def apply_theme():
    B.THEME.update(THEME)
    for k_, v in ARCS.items():
        setattr(B, k_, v)
    B.RAINBOW[:] = BALLS
    C.RAINBOW[:] = BALLS


def audio(sim, path, sample="assets/rizz.wav", vol=0.6):
    """Rizz NUR bei Quetschungen, bei jeder einzelnen, ohne Mindestabstand (Überschneiden = Chaos gewollt).
    Feste Lautstärke + Soft-Limiter: die ersten einzelnen Quetschungen bleiben voll hörbar,
    das Chaos sättigt weich. Ausklang am Ende wird an den Anfang gefaltet (nahtloser Loop)."""
    import wave
    import numpy as np
    SR = 44100
    with wave.open(sample) as w:
        smp = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64) / 32768
    smp = smp * np.minimum(1, np.arange(len(smp)) / (SR * 0.003))
    dur = sim.n_frames / A.FPS
    n = int(SR * dur)
    out = np.zeros(n + len(smp))
    for t, kind, _ in sim.events:
        if kind == "split" and t < dur:
            i = int(t * SR)
            out[i:i + len(smp)] += smp * vol
    tail = out[n:]
    out = out[:n]
    out[:len(tail)] += tail                      # Loop-Naht: Ausklang landet am Anfang
    out = np.tanh(out * 1.2) * 0.9
    st = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    E.apply_theme = apply_theme
    sim, first = E.build(seed=11)
    print("erste Quetschung", round(first, 2), "voll", sim.full_t, "Loop", sim.loop_t,
          "Quetschungen", sum(1 for e in sim.events if e[1] == "split"))
    old_r, old_a = A.ArcRenderer, A.build_audio
    A.ArcRenderer, A.build_audio = C.ShortRenderer2, audio
    try:
        A.render(sim, "output/a11_arc_squeeze_rizz_chaos_vertical.mp4")
    finally:
        A.ArcRenderer, A.build_audio = old_r, old_a
    print("output/a11_arc_squeeze_rizz_chaos_vertical.mp4")
