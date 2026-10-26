"""Fertig abgestimmte Satisfying-Episoden. Seeds wurden per Suche gewählt (siehe make_satisfying.py --search)."""
from .sim import Elimination, GrowingBall, Multiply, RingEscape, StringArt

EPISODES = {
    "s01_ring_battle": dict(
        kind="ring", title="WELCHE FARBE GEWINNT?",
        make=lambda: RingEscape(237),
        teaser=("TEIL 2", "8 FARBEN – 40 RINGE"),
        caption="Welche Farbe hast du gewählt? 👇 #satisfying #simulation #physics #fyp #fürdich",
    ),
    "s02_wachsender_ball": dict(
        kind="grow", title="FÜLLT DER BALL DEN KREIS?",
        make=lambda: GrowingBall(1, growth=3.0),
        teaser=("TEIL 2", "2 BÄLLE GLEICHZEITIG"),
        caption="Wie viele Aufpralle hast du geschätzt? 🤔 #satisfying #oddlysatisfying #physics #fyp",
    ),
    "s03_last_one_standing": dict(
        kind="elim", title="WER BLEIBT ALS LETZTES DRIN?",
        make=lambda: Elimination(98, gap_deg=14),
        teaser=("TEIL 2", "10 FARBEN – 1 GEWINNER"),
        caption="Welche Farbe war deine? 👇 #satisfying #marblerace #simulation #fyp #fürdich",
    ),
    "b01_ball_explosion": dict(
        kind="multiply", title="1 BALL WIRD ZU 250",
        make=lambda: Multiply(7, gap_deg=40),
        teaser=("TEIL 2", "1 BALL WIRD ZU 1000"),
        caption="Wie lange hast du geschätzt? 🤯 #ballbounce #satisfying #simulation #fyp #fürdich",
    ),
    "b02_string_art": dict(
        kind="string", title="WAS ZEICHNET DER BALL?",
        make=lambda: StringArt(1, speed=(1000, 1500)),
        teaser=("TEIL 2", "2 BÄLLE – 200 LINIEN"),
        caption="Wie sieht es aus? 😍 #ballbounce #satisfying #stringart #fyp #fürdich",
    ),
    "b03_ring_escape_solo": dict(
        kind="solo", title="SCHAFFT ER ALLE 20 RINGE?",
        make=lambda: RingEscape(4, names=("WEISS",), n_rings=20, r_in=90, gap_deg=45),
        teaser=("TEIL 2", "50 RINGE – SCHAFFT ER ES?"),
        caption="Hättest du gedacht, dass er es schafft? 😮 #ballbounce #satisfying #fyp #fürdich",
    ),
}


def _outro(sim, sec=1.8):
    sim.outro = sec
    return sim


# Hypnose-Versionen: kein Text, schnelleres Tempo, starke Effekte (Renderer: satisfying/hypno.py)
HYPNO = {
    "h01_ring_escape": dict(
        kind="ring",
        make=lambda: _outro(RingEscape(35, names=("WEISS",), n_rings=50, r_in=60, gap_deg=65, speed=(900, 1300), jitter=0.35)),
        caption="Bis zum Ende schauen 🌀 #satisfying #ballbounce #oddlysatisfying #fyp #fürdich",
    ),
    "h02_ball_explosion": dict(
        kind="multiply",
        make=lambda: _outro(Multiply(8, gap_deg=32, n0=6, target=1000, speed=(700, 1100), r=7)),
        caption="Warte aufs Ende 🤯 #satisfying #ballbounce #simulation #fyp #fürdich",
    ),
    "h03_string_art": dict(
        kind="string",
        make=lambda: _outro(StringArt(7, target=200, speed=(1000, 1500), n_balls=2)),
        caption="So hypnotisch 😵‍💫 #satisfying #stringart #ballbounce #fyp #fürdich",
    ),
}
