# TikTok-Video-Generator

## Satisfying-Simulationen (aktuelle Nische)

Neon-Physiksimulationen mit Tönen bei jedem Aufprall, 52–55 s, fertig zum Hochladen.

```
python3 make_satisfying.py                         # alle Episoden, Hoch- + Querformat
python3 make_satisfying.py s01_ring_battle -f vertical
python3 make_satisfying.py --preview               # Standbilder inkl. simulierter TikTok-Oberfläche
python3 make_satisfying.py --search ring           # spannende Seeds für ein neues Ring-Battle finden
```

| Episode | Idee | Retention-Trick |
|---|---|---|
| `s01_ring_battle` | 4 Farben in 22 rotierenden Ringen, wer durch eine Lücke fliegt, zerschlägt den Ring (+1) | Seed mit 5 Führungswechseln, Comeback-Sieg mit 1 Punkt Vorsprung; „NOCH X RINGE“-Zähler |
| `s02_wachsender_ball` | Ball wächst bei jedem Aufprall bis er den Kreis füllt | Fortschrittsbalken in %, Melodie wird zum Ende immer schneller, Explosion |
| `s03_last_one_standing` | 6 Farben, Arena mit drehender Lücke, wer rausfällt ist raus | Ausscheiden ca. alle 6–11 s, „FINALE: X GEGEN Y“ |

Jedes Video: Frage-Hook ab Frame 1, „Schreib deine Farbe in die Kommentare“, Gewinner + Konfetti,
dann Cliffhanger „TEIL 2 – Folge mir“. Die Simulation ist deterministisch (Seed): das Ergebnis ist
vorab bekannt und so gewählt, dass es bis zum Schluss spannend bleibt.
Beschreibungstexte für TikTok stehen in `satisfying/episodes.py` (`caption`).

Code: `satisfying/sim.py` (Physik), `satisfying/scene.py` (Neon-Rendering, Layouts/Safe-Zones),
`satisfying/sound.py` (Marimba-Melodie Am–F–C–G, Glas, Fanfare).

---

## Tür-Rätsel (ältere Nische)

Erstellt 45–60-Sekunden-Rätselvideos im Stil „Welche Tür wählst du?“, fertig zum Hochladen.

```
python3 make_video.py                         # alle Episoden, Hochformat + Querformat
python3 make_video.py ep01_loewen -f vertical # eine Episode, ein Format
python3 make_video.py --preview               # nur Standbilder inkl. simulierter TikTok-Oberfläche
```

Ausgabe: `output/<episode>_vertical.mp4` (1080×1920, 9:16) und `output/<episode>_landscape.mp4` (1920×1080, 16:9).
Benötigt: Python 3, numpy, Pillow, ffmpeg.

## Was ich aus deinen Beispielvideos übernommen habe

| Beispiel | Übernommenes Element |
|---|---|
| Bunte Türen in der Wüste | Sofort im ersten Frame 3 klar unterscheidbare, farbige Türen |
| Feuer/Eis/Gas-Türen + „Nur 1% kennen die Lösung“ | Hook-Titel oben die ganze Zeit sichtbar, gefährliche Räume hinter den Türen |
| Slenderman + Countdown 4-3-2 | Countdown-Ring mit Ticken + Herzschlag, Wort-für-Wort-Untertitel |

## Aufbau jedes Videos (Retention-Struktur)

1. **0–3 s Hook:** „Nur 1 % …“ + Türen sind sofort da (kein Intro, kein Logo).
2. **Spannungsaufbau:** jede Tür öffnet sich einzeln mit Zoom + Whoosh → man will die nächste Tür sehen.
3. **Frage + 7-s-Countdown:** „Pausiere und schreib deine Tür in die Kommentare“ → Kommentare pushen den Algorithmus.
4. **Pattern Interrupt:** „Die meisten wählen Tür 1 … FALSCH!“ (Stempel + Buzzer).
5. **Auflösung** mit Riser-Sound, grünem Haken und Erklärung.
6. **Cliffhanger:** „TEIL 2 – nur 0,1 % schaffen es. Folge mir!“ → Grund zu folgen und das nächste Video zu suchen.
   Das Video endet hart (kein Ausblenden), damit der Loop nahtlos wieder beim Hook startet.

## Safe-Zones (nichts wird verdeckt)

* **Hochformat 1080×1920:** Inhalt nur in x 60–920, y 230–1420. Rechts bleibt die Spalte für
  Profilbild/Like/Kommentar/Teilen frei, oben „Folge ich | Für dich“, unten Username + Beschreibung.
  Auch der Kamera-Zoom ist so begrenzt, dass keine Tür unter die Buttons rutscht.
* **Querformat 1920×1080:** rechts 270 px frei (dort ragt die Button-Leiste ins Video).
* `--preview` legt eine nachgebaute TikTok-Oberfläche über die Frames (`*_ui.png`) zur Kontrolle.

> **Tipp zum Format:** TikTok ist ein Hochformat-Feed. 16:9-Videos werden dort als schmaler Streifen
> mit schwarzen Balken gezeigt und wirken viel kleiner. Für Reichweite die `_vertical.mp4` hochladen.

## Beim Hochladen

* **Sprachausgabe:** In TikTok Text hinzufügen → „Text-zu-Sprache“ (deutsche Stimme) – das Skript steht
  in den Untertiteln bzw. in `episodes/*.py` (`text`). Musik/SFX sind bereits im Video.
* **Beschreibung (Beispiel):** `Welche Tür hättest du gewählt? 🚪👇 #rätsel #logikrätsel #denksport #fyp #fürdich`
* **Erster Kommentar (selbst anpinnen):** „Ich hätte Tür __ gewählt 😅“ – startet die Diskussion.
* Teile im Abstand von 1–2 Tagen posten, damit der „Teil 2“-Cliffhanger funktioniert.

## Neue Episode erstellen

Kopiere eine Datei in `episodes/` und passe sie an. Wichtige Felder pro Segment:

| Feld | Wirkung |
|---|---|
| `text` | Untertitel (Wort für Wort); Dauer wird automatisch aus der Wortanzahl berechnet |
| `dur` | Dauer manuell setzen (Sekunden) |
| `open` / `close` | Türen öffnen/schließen (Index ab 0) |
| `label` | Beschriftung unter Tür einblenden |
| `focus` | Kamera zoomt leicht auf diese Tür |
| `highlight` | Türen gelb pulsierend umranden |
| `mark` | `{0: "x"}` rotes Kreuz, `{2: "check"}` grüner Haken |
| `stamp` | großer Text, z. B. `["FALSCH!", [235, 40, 40]]` |
| `countdown` | Countdown in Sekunden (+ `hint`) |
| `riser` | Spannungs-Sound, der zum Segmentstart aufsteigt |
| `teaser` | Cliffhanger-Bildschirm „TEIL X“ |

Türen: `color`, `inside` (`fire`, `gas`, `water`, `dark`, `light`, `void`), optional `icon`/`guard` (Emoji),
`sign` (Schild auf der Tür), `label`, `label_color`.
