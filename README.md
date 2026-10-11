# Smart Crossing

**Live ansehen: https://smartcrossing.pages.dev**

Eine Kreuzung ohne Ampeln, ohne Spuren und ohne Vorfahrtsregeln, über die nur selbstfahrende
Autos fahren. Jedes Auto hat ein kleines neuronales Netz, sieht seine Umgebung (Fahrbahnrand,
die nächsten Autos und deren Dachlampe) und lernt selbst, zu lenken, Gas zu geben, zu bremsen
und mit seiner Lampe zu "sprechen". Was die Lampe bedeutet, ist nicht vorgegeben.

Fragen, die das Projekt untersucht:

- Wie viele Autos pro Minute schaffen lernende Autos über eine regelfreie Kreuzung – und wie
  oft krachen sie dabei?
- Einigen sie sich von selbst auf Rechts- oder Linksverkehr? *(Ja – welche Seite, hängt vom
  Zufall beim Trainingsstart ab.)*
- Entsteht über die Lampen eine Art Sprache? *(Bisher nein: die Lampen senden etwas, aber
  niemand hört hin.)*
- Was passiert, wenn unterschiedlich trainierte Autos gemeinsam fahren?

Die Website erklärt Schritt für Schritt, was ein Auto sieht, was es tun kann, wie sein
"Gehirn" aussieht und wie es lernt. Alles läuft direkt im Browser.

## Selbst ausprobieren

Voraussetzungen: Python 3.13 und [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run pytest -q                                                      # Tests
poetry run python -m smartcrossing.cli bench --driver reservation --demand 0.15   # Kennzahlen
poetry run python -m smartcrossing.cli render --driver light --out runs/light     # Bilder + GIF
poetry run python site/build.py                                          # Website bauen
cd site/public && python3 -m http.server 8000                            # Website lokal ansehen
```

## Aufbau

| Pfad | Inhalt |
|---|---|
| `smartcrossing/sim.py` | Simulation der Kreuzung (numpy, viele Kreuzungen parallel) |
| `smartcrossing/ppo.py`, `es.py` | Training: PPO und Evolution Strategies |
| `smartcrossing/baselines.py` | Vergleichsfahrer mit Spuren: Ampel und zentraler Verkehrsrechner |
| `site/` | die Website; `site/src/js/sim.js` ist die Browser-Version der Simulation |
| `docs/PLAN.md` | Ziele, Entscheidungen, Architektur |
| `docs/experiments.md` | Protokoll aller Trainingsläufe |

Ein Hobbyprojekt, entwickelt größtenteils mit [Claude Code](https://claude.com/claude-code).
