# Smart Crossing

<p align="center"><a href="https://smartcrossing.pages.dev"><img src="results/readme/crossing.gif" width="420" alt="Selbst lernende Autos überqueren eine Kreuzung ohne Ampeln"></a></p>
<p align="center"><em>Ein Netz steuert alle Autos, ohne Ampel und Spuren (Farbe = Abbiegerichtung, doppelte Geschwindigkeit) – <a href="https://smartcrossing.pages.dev">live im Browser ausprobieren</a>.</em></p>

**Live ansehen: https://smartcrossing.pages.dev**

Eine Kreuzung ohne Ampeln, ohne Spuren und ohne Vorfahrtsregeln, über die nur selbstfahrende
Autos fahren. Jedes Auto hat ein kleines neuronales Netz, sieht seine Umgebung (Fahrbahnrand,
die nächsten Autos und deren Dachlampe) und lernt selbst, zu lenken, Gas zu geben, zu bremsen
und mit seiner Lampe zu "sprechen". Was die Lampe bedeutet, ist nicht vorgegeben.

Fragen, die das Projekt untersucht:

- Wie viele Autos pro Minute schaffen lernende Autos über eine regelfreie Kreuzung – und wie
  oft krachen sie dabei?
- Wie fahren sie, wenn sie auf der rechten Straßenhälfte starten (Rechtsverkehr ist vorgegeben)?
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
poetry run python -m smartcrossing.cli evalset --driver policy:results/round4_best.npz --out runs/eval.jsonl   # Standard-Auswertung
poetry run python site/build.py                                          # Website bauen
cd site/public && python3 -m http.server 8000                            # Website lokal ansehen
```

## Aufbau

| Pfad | Inhalt |
|---|---|
| `smartcrossing/sim.py` | Simulation der Kreuzung (numpy, viele Kreuzungen parallel) |
| `smartcrossing/ppo.py` | Training mit PPO (optional auf der GPU: `poetry install --with gpu`, dann `--rollout-device cuda`; siehe `ppo_torch.py`, `sim_torch.py`) |
| `smartcrossing/baselines.py` | Vergleichsfahrer mit Spuren: Ampel und zentraler Verkehrsrechner |
| `smartcrossing/policy.py`, `policy_set.py` | das Gehirn: einfaches Netz (MLP) bzw. Netz mit Mini-Netz pro Nachbar + Aufmerksamkeit; `policy_compat.py` lädt alte Gehirne |
| `smartcrossing/bc.py` | Nachahmen: ein neues Gehirn lernt erst vom alten (`ppo --init-bc`) |
| `smartcrossing/gates.py` | Anhaltetest: hält ein Auto vor einem Hindernis an? |
| `smartcrossing/evalset.py` | Standard-Auswertung (`cli evalset`): mehrere Verkehrsdichten und Zufalls-Seeds + Anhaltetest |
| `site/` | die Website; `site/src/js/sim.js` ist die Browser-Version der Simulation |
| `docs/PLAN.md` | Ziele, Entscheidungen, Architektur |
| `docs/experiments.md` | Protokoll aller Trainingsläufe |

Ein Hobbyprojekt, entwickelt größtenteils mit [Claude Code](https://claude.com/claude-code).
