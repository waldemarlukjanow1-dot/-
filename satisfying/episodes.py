"""Fertig abgestimmte Satisfying-Episoden. Seeds wurden per Suche gewählt (siehe make_satisfying.py --search)."""
from .sim import Elimination, GrowingBall, RingEscape

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
}
