"""Fertig abgestimmte Satisfying-Episoden. Seeds wurden per Suche gewählt (siehe make_satisfying.py --search)."""
from .flow import FlowSim
from .maze import MazeSim
from .marble import MarbleSim
from .orbit import OrbitSim
from .minimal import SpiralSim, StringSim
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


# Flow-Videos: zufällige Sprungphysik, Muster-Schweife, ruhiges Ende (Renderer: satisfying/flow.py)
FLOW = {
    "f01_neon_schwarm": dict(
        mode="trail", instrument="bell",
        make=lambda: FlowSim(11, n_start=2, n_max=9, spawn_every=4.5, slow_at=44.0, r=21.0),
        caption="Welches Muster ist deins? ✨ #satisfying #ballbounce #oddlysatisfying #fyp #fürdich",
    ),
    "f02_neon_malerei": dict(
        mode="paint", instrument="pluck",
        make=lambda: FlowSim(23, n_start=4, n_max=4, slow_at=45.0, r=13.0, speed=(650, 1150), speedup=0.3,
                             styles=["ribbon", "dots", "rings", "zigzag"]),
        caption="Die Bälle malen ein Kunstwerk 🎨 #satisfying #ballbounce #art #fyp #fürdich",
    ),
}


# Parcours: Bälle müssen sich durch 5 Ebenen mit Schlitzen/Toren/Hindernissen ins Ziel arbeiten (satisfying/maze.py)
MAZE = {
    "m01_der_weg_nach_unten": dict(
        make=lambda: MazeSim(242, gap_k=1.45),
        caption="Schafft es der letzte Ball? 😮‍💨 #satisfying #ballbounce #marblerun #fyp #fürdich",
    ),
}


# Große Murmelbahn: Bounce-Kreis -> Bahn mit Speedbändern, Trampolin, Wippe, Looping, Katapult -> Ziel (marble.py)
MARBLE = {
    "r01_murmelbahn": dict(
        make=lambda: MarbleSim(256, outro=7.0),
        caption="Schaffen es alle drei ins Ziel? 🥹 #satisfying #marblerun #ballbounce #asmr #fyp #fürdich",
    ),
}


# Kreis-Kette: Ball-Bounce mit System – Leuchtpunkte füllen, Kreis öffnet sich, Bälle verdoppeln sich (orbit.py)
ORBIT = {
    "o01_kreis_kette": dict(
        make=lambda: OrbitSim(12),
        caption="Jeder Kreis verdoppelt die Bälle 🔵➡️🔵🔵 #satisfying #ballbounce #oddlysatisfying #asmr #fyp #fürdich",
    ),
}


# Minimalistische Viral-Formate (minimal.py): jeder Aufprall = nächster Ton einer eigenen Melodie
MINIMAL = {
    "v01_spiral_ringe": dict(
        kind="spiral", make=lambda: SpiralSim(40, n=60, half_deg=36, twist=0.12, omega=1.2),
        caption="Schafft er alle 60 Ringe? 🔵 #bouncingball #satisfying #circle #ring #fyp #fürdich",
    ),
    "v02_faeden": dict(
        kind="strings", make=lambda: StringSim(3, speed=(900, 1250)), octave=0,
        caption="Jeder Aufprall = eine Linie 🎵 #bouncingball #satisfying #circle #fyp #fürdich",
    ),
}
