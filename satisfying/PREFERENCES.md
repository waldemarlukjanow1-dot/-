# Wünsche für die nächsten Videos (Stand: nach m07)

- **Kein Schluss-Sound:** Wenn der Kreis voll ist, gibt es keinen extra Ton mehr.
- **Keine Ring-Effekte:** Beim Zerquetschen/Verdoppeln entstehen keine Kreise/Ringe mehr.
- **Katapult rein physikalisch:** Die Bälle bekommen keine gesetzte Geschwindigkeit mehr. Sie bewegen sich nur,
  weil der Kolben sie anschiebt (Reaktion auf die Kolbenbewegung). In `press3.py`: `SCRIPTED_LAUNCH = False`.
- Weiterhin: Zähler unter dem Kreis, Ton = weiches Plopp (Zerquetschen) + tiefes Plopp (Presse/Katapult),
  keine Überschrift, Kreis soll am Ende bis zum Rand voll sein, Spannung früh aufbauen.
- **Arc Box (Vorbild "The Bouncing Ball"):** Jeder Halbkreis hängt mit EINEM Ende an einer Ecke des Vierecks und
  dreht sich um diese Ecke. Der Halbkreis quetscht die Bälle an die Wand des Vierecks – nur so eingeklemmte Bälle
  verdoppeln sich (pro Ball), die zwei bekommen die nächste Farbe.
- **Arc Box ab a04:** alle Halbkreise drehen im Uhrzeigersinn, neon-grün mit Textur; Ton nur beim Quetschen an
  der Wand (Plopp pro verdoppeltem Ball), 60 fps mit leichter Bewegungsunschärfe, große kräftige Neon-Bälle.
- **Arc Squeeze optimiert (a05, `arcbox2.py`):** fast schwarzer Hintergrund, schlichter weiß-grüner Rahmen,
  Glossy-Bälle mit Rand + Schatten, Glow nimmt mit Ballzahl ab, Regenbogen-Farben je Generation, Bogen-Schweif,
  Bogen leuchtet beim Quetschen kurz auf, Squash beim Teilen, Mikro-Zoom bei 100/250/500, Ende: Bögen blenden aus,
  Rahmen leuchtet einmal auf.
- **Loop (a06):** alle Halbkreise drehen gleichzeitig im Uhrzeigersinn mit gleicher Geschwindigkeit; Video endet
  nahtlos im Startbild (Kaskade bis 1 Ball, letzter Ball fliegt mit normalem Tempo in die Startposition).
- **Viral-Kurzversion (a07, `arcbox3.py`):** ~18 s Loop, 4 Start-Bälle (rot), erste Quetschung < 1 s,
  größeres Viereck weiter oben, Plopp-Tonhöhe steigt mit Ballzahl (Melodie), leises Summen der Bögen,
  kurze leuchtende Spur hinter den Bällen.
- **Ab a10:** Länge 45–62 s, ruhiger aber nicht zu langer Start (2 Bälle, erste Quetschung ~1 s),
  eigener Sound bei jeder Quetschung (assets/rizz.wav, max. ~3 pro Sekunde), jedes Video neue Farben.
