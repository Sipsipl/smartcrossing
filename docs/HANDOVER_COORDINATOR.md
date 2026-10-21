# Handover: Coordinator für Smart Crossing (lokal, Windows)

Du übernimmst als Coordinator/Orchestrator das Hobbyprojekt **Smart Crossing**
(https://github.com/Sipsipl/smartcrossing, live: https://smartcrossing.pages.dev). Die
vorherige Coordinator-Session lief in der Cloud; du läufst jetzt lokal auf dem Windows-PC des
Owners, damit Kontext frisch ist und GPU-Training möglich wird.

## Erst lesen (alles Projektwissen steht im Repo)
1. `CLAUDE.md` – Befehle, Regeln.
2. `docs/PLAN.md` – Ziele, feste Entscheidungen des Owners, Architektur, Status, Roadmap,
   „Decided for the next training round“. **Die festen Entscheidungen sind verbindlich.**
3. `docs/experiments.md` – Protokoll aller Läufe; nur die letzten Abschnitte im Detail lesen.
4. `README.md`, `docs/DEPLOY.md` bei Bedarf.

## Schritt 0: lokal einrichten (Windows; erledigt, Hardware/Versionen in `CLAUDE.md`)
- `git pull` auf `main` (dort liegt der aktuelle Stand). Neuer Arbeits-Branch für deine Arbeit.
- Herausfinden, was da ist: Python-Versionen (pyenv-win: `pyenv versions`; Projekt braucht 3.13,
  `.python-version`), Poetry (`poetry --version`), Node (für den JS-Paritätstest), Git, CUDA/GPU
  (`nvidia-smi`). Fehlendes mit dem Owner klären, bevor du etwas global installierst.
- `poetry install`, dann `poetry run pytest -q` muss grün sein. Auf Windows auf
  Pfad-/Encoding-/Multiprocessing-Eigenheiten achten (`spawn` statt `fork`; Trainings nutzen
  `multiprocessing`).
- `runs/` (Trainingsläufe) existiert lokal nicht; die Website baut aus `results/`
  (`poetry run python site/build.py`).

## Arbeitsweise (vom Owner so gewünscht und bewährt)
- **Owner möglichst selten fragen.** Du planst, delegierst an Subagents, prüfst selbst.
  Rückfragen nur bei echten Entscheidungen – dann einfach verständlich, **mit
  Antwortmöglichkeiten und markierter Empfehlung**.
- **Kontext klein halten:** Arbeit an Subagents geben (klare Dateizuständigkeit, keine zwei
  Agents in denselben Dateien), knappe Berichte (Zahlen, Pfade, keine Code-Dumps) anfordern.
  Subagents schauen sich Bilder/Screenshots selbst an; du nur stichprobenartig.
- **Standard-Eval = `cli evalset`** (4 Nachfragen × 3 Verkehrs-Seeds × 8 Envs × 120 s + Anhaltetest, JSONL wie results/round4_eval.jsonl).
- **Modellauswahl = `cli select` mit Gate-Filter** (alle Checkpoints eines Runs: Anhaltetest >= 0.95 als harter Filter, dann bester Durchsatz bei 0.5 unter Crash-Limit bei 0.3; `--every N`, `--workers 4`, `--copy-to`).
- **Selbst evaluieren:** `pytest`, `cli bench` (JSON-Zeile), `cli render` (sheet.png,
  spacetime.png), `analyze.py`, Playwright-Screenshots der Website. Nie Erfolg behaupten ohne
  Beleg; Misserfolge ehrlich berichten.
- **Vor jedem Training:** Dauerabschätzung + 2-3 Optionen (kurz/mittel/lang), der Owner entscheidet. Der PC des Owners ist ein Spiele-PC: Auf Zuruf sofort stoppen, vor langen Läufen warnen.
- **Refactoring:** regelmäßig, vom Coordinator zwischen den Runden angesetzt (erst Audit, dann Umbau, Bench identisch).
- **Token sparen:** einfache Subagent-Aufgaben mit Sonnet.
- **Training kurz:** während der Entwicklung ≤ ~1 h pro Runde; lange Läufe nur vor einem Merge
  nach `main`. Kein Training ohne Grund starten; der Owner kann „kein Training“ vorgeben.
- **Belohnung nur für Ergebnisse** (ankommen, Unfall/Abkommen nach Aufpralltempo, Zeit,
  Warten). Keine Stilvorgaben (keine Strafe fürs Wackeln, kein Spurbonus). Lernende Autos
  bekommen nie Spur-/Regel-/Zielinformation anderer Autos.
- **Git:** auf dem Arbeits-Branch committen und pushen (Zwischenstände ruhig als WIP).
  `main` ist geschützt/live: Merge nur per PR und erst nach OK des Owners; CI (`pytest`) muss
  grün sein. Branch-Vorschau: `https://<branch-alias>.smartcrossing.pages.dev` (Cloudflare kürzt
  den Alias auf 28 Zeichen; genaue URL im Cloudflare-Dashboard bzw. im PR-Check).
- **Kommunikation:** Deutsch, direkt, nüchtern, ehrlich statt gefällig. Uhrzeiten in Berliner
  Zeit. Der Owner ist visuell: Ergebnisse gern als Bild/Screenshot zeigen.
- **Website-Texte:** Deutsch, locker, für Laien; keine internen Kürzel (z. B. „ppo17“); Zahlen im
  Fließtext immer aus den Messdaten, nie von Hand; keine Redundanz.
- **Subagents in der Cloud:** Wenn dir dafür ein Werkzeug zur Verfügung steht (Cloud-Sessions
  von claude.ai/code), darfst du Aufgaben ohne GPU-Bedarf dorthin geben (z. B. Website,
  Analyse, Refactoring). GPU-Arbeit bleibt lokal. Falls nicht verfügbar: lokale Subagents.

## Workflow-Tipps aus der bisherigen Zusammenarbeit
- **Subagent-Aufträge selbsttragend schreiben:** Ziel, zu lesende Dateien, **Dateizuständigkeit**
  („du besitzt X; fass Y nicht an“), Abbruch-/Zeitlimit, „nicht committen“ (du committest),
  und ein festes Berichtsformat: max. 150–300 Wörter, Zahlen + Pfade, kein Code.
- **Abnahmebedingungen mitgeben**, nicht nur Aufgaben: z. B. „pytest grün“, „Bench vorher/nachher
  identisch“ (Refactoring), „Screenshots selbst ansehen“, „keine Konsolenfehler“.
- **Simulationsänderungen seriell:** `sim.py` + `site/src/js/sim.js` + Paritätstest hängen
  zusammen; immer nur ein Agent darin. Website-Arbeit parallel nur in `site/`-Dateien.
- **Erst lesen, dann ändern:** Bei unklarer Lage zuerst einen read-only Prüf-Agent (Audit,
  Messung, Ausfall-Aufschlüsselung) und auf Basis der Liste handeln. Messen vor Optimieren.
- **Robuste Zahlen:** Bench über mehrere Verkehrs-Seeds; ein einzelner Seed täuscht (±10–20
  Unfälle/1000). Immer mit Ampel/Verkehrsrechner und dem bisherigen besten Modell vergleichen.
- **Seeds nach Zweck trennen:** Auswahl-Seeds 1000/2000/3000 (`select.py`, `evalset.py`, Training) – alles auf der Website (Bench, Fundamental, Wolke) nutzt die ungesehenen Berichts-Seeds 4000/5000/6000 (`deadlock.REPORT_SEEDS`), sonst schmeichelt die Auswahl dem gewählten Gehirn.
- **Trainings:** vor dem Start Iterationen/min messen und Dauer hochrechnen; Zeitlimit im Auftrag;
  CPU-Last im Blick (Training + Website-Build gleichzeitig überbucht die Kerne).
- **Berichte kritisch prüfen:** Subagent-Berichte und externe Handover sind Behauptungen. Gegen
  Repo/Daten prüfen (eine frühere Analyse beruhte auf einem veralteten Commit). Stichprobe:
  ein Bild ansehen, eine Zahl nachrechnen, Tests selbst laufen lassen.
- **Häufig committen** (WIP auf dem Branch), damit Abbrüche (Ausgabenlimit, Neustart) nichts
  kosten. Abgebrochene Agents mit kurzem Statusupdate fortsetzen statt neu starten.
- **Gedächtnis ist das Repo:** nach jeder Runde `docs/experiments.md` (was, warum, Ergebnis) und
  bei Entscheidungen `docs/PLAN.md` aktualisieren lassen. Neue Owner-Regeln sofort dort festhalten.
- **Dem Owner berichten:** kurz, Tabelle mit Vergleichswerten, ehrliche Einordnung (was hat
  nicht geklappt), dann ein konkreter Vorschlag. Kein Fachjargon ohne Erklärung.

## Nächste Schritte (Vorschlag, mit dem Owner abstimmen)
Erledigt (2026-10-11): lokale Einrichtung, GPU-Training (Roadmap C), Runde 1 in der neuen Welt
(Wracks, Bremslicht, 1 Lampe, Einfahren rechts), Unfall-Diagnose, erster Aufräum-Schritt.
1. **Runde 2** (Design in PLAN.md): Varianten A/B/C (Exploration, Hindernis-Szenarien, Unfallkosten
   nach eigener Aufprallgeschwindigkeit), gleich lang, gepaart verglichen; Anhaltetest als Messung.
2. **Roadmap B:** Set-/Attention-Encoder über alle Nachbarn statt der 4 nächsten, optional GRU;
   dazu Vergleich breiteres Netz.
3. **Roadmap D (Forschungsfrage):** Normen-Experimente mit ≥ 20 Seeds pro Konfiguration
   (Fahrseite, Spuren, Vorrangmuster, Lampenbedeutung, Ablationen, getrennte Populationen),
   sobald die Autos ordentlich fahren. GPU macht viele Seeds erst praktikabel.

## Bekannte Stolpersteine
- Das monatliche Ausgabenlimit kann Subagents mitten in der Arbeit abbrechen; Zwischenstände
  deshalb oft committen und abgebrochene Agents mit Kontext fortsetzen.
- Windows: ein lokaler Webserver (`python -m http.server` in `site/public`, z. B. von einem
  Browser-Check) sperrt das Verzeichnis; `site/build.py` bricht dann mitten im Löschen ab. Vor dem
  Build solche Server beenden und vor jedem Commit `git status` prüfen.
- CUDA-Tests laufen nur mit `SC_TEST_CUDA=1` (nie während eines Trainings; Standardlauf überspringt sie).
- Mehrere CPU-lastige Jobs gleichzeitig (Training + Website-Build) überbuchen die Kerne.
- Alte Checkpoints laufen über `smartcrossing/policy_compat.py`; bei Änderungen an
  Beobachtung/Aktion prüfen, dass die Website-Gehirne weiter laden (`site/build.py`, Tests).
