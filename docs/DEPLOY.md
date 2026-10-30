# Website veröffentlichen (Cloudflare Pages, kostenlos)

Die Website ist rein statisch: HTML, CSS, JavaScript und eine Datendatei. Die Simulation
läuft komplett im Browser der Besucher, es gibt keinen Server und keine Datenbank.

## Was liegt wo?

| Ordner | Inhalt | ins Git? |
|---|---|---|
| `site/src/` | die handgeschriebenen Seiten (HTML/CSS/JS) | ja |
| `site/build.py`, `site/sitebuild/` | Build (Einstiegspunkt + Paket): kopiert `site/src/` (inkl. Simulation `js/sim.js`), schreibt `js/sim-config.js` (Zahlen aus `smartcrossing/sim.py`), wählt Gehirne aus dem Schnappschuss `results/site/` plus `runs/` (zusammengeführt), misst sie, schreibt die Daten | ja |
| `results/site/` | kompakter Schnappschuss von allem, was der Build aus `runs/` nimmt (benutzte Gehirne, Trainingskette, Lernkurven); schreibt der Build selbst | ja |
| `site/obs_labels.py` | deutsche Beschriftungen der Sensorwerte | ja |
| `site/data/bench_cache.json` | Zwischenspeicher der Messungen (damit ein Neubau schnell geht) | ja |
| `site/public/` | **die fertige Website** – genau dieser Ordner wird veröffentlicht | **ja** |

Der Code liegt auf <https://github.com/Sipsipl/smartcrossing>; jede Seite verlinkt ihn in der Fußzeile.

Warum wird `site/public/` mit eingecheckt? Der Build braucht Python + Poetry und die Gehirne
(aus `runs/`, gitignored, oder dem eingecheckten Schnappschuss `results/site/`; ein frischer
`git clone` + `poetry install` + `poetry run python site/build.py` baut also ohne `runs/`).
Cloudflare hat das nicht. Also
bauen wir lokal und Cloudflare liefert den fertigen Ordner einfach aus – ohne Build-Schritt.
(`dist/` und `build/` sind in `.gitignore` ausgeschlossen, darum heißt der Ordner `public/`.)

## Einmalig einrichten

1. Auf <https://dash.cloudflare.com> anmelden (der Account deines anderen Projekts geht).
2. Links **Workers & Pages** → **Create** (bzw. „Erstellen“) → Reiter **Pages** →
   **Connect to Git** („Mit Git verbinden“).
3. GitHub verbinden, falls noch nicht geschehen, und das Repository **smartcrossing**
   auswählen. Ein privates Repository ist kein Problem (Cloudflare fragt nach Zugriff nur
   auf ausgewählte Repositories – dieses eine reicht).
4. Einstellungen:
   - **Project name**: z. B. `smartcrossing` → die Adresse wird `smartcrossing.pages.dev`
   - **Production branch**: `main`
   - **Framework preset**: `None`
   - **Build command**: leer lassen
   - **Build output directory**: `site/public`
   - **Root directory**: leer lassen (Repository-Wurzel)
5. **Save and Deploy**. Nach ca. einer Minute ist die Seite unter
   `https://<projektname>.pages.dev` erreichbar.

Optional: Unter **Custom domains** eine eigene (Sub-)Domain eintragen, z. B.
`kreuzung.deinedomain.de`, wenn die Domain schon bei Cloudflare liegt.

Kosten: keine. Der kostenlose Tarif reicht für eine statische Hobbyseite bei Weitem
(unbegrenzte Aufrufe statischer Dateien, 500 Deployments pro Monat).

## Aktualisieren

Immer wenn neue Trainingsergebnisse da sind oder Texte geändert wurden:

```bash
poetry install                      # einmalig
poetry run python site/build.py     # baut site/public/ neu
git add site/ && git commit -m "Website aktualisieren" && git push
```

Cloudflare merkt den Push auf `main` und veröffentlicht automatisch neu
(<https://smartcrossing.pages.dev>). Pushes auf andere Branches bekommen eine eigene
Vorschau-Adresse (`<branch>.<projekt>.pages.dev`) – praktisch, um etwas vorher anzusehen.
Achtung: Cloudflare kürzt den Branch-Namen dafür auf **28 Zeichen** (Sonderzeichen wie `/`
werden zu `-`). Beispiel: Branch `claude/youthful-ritchie-fc1lad` →
<https://claude-youthful-ritchie-fc1l.smartcrossing.pages.dev>.

Nützliche Optionen des Build-Skripts:

```bash
poetry run python site/build.py --main runs/r1_x3/current.npz    # bestimmtes Gehirn als Hauptgehirn nehmen
poetry run python site/build.py --source snapshot                # nur results/site/ benutzen (wie ein frischer Clone)
poetry run python site/build.py --previous ''                    # kein "vorheriges Gehirn" auf Vergleich
poetry run python site/build.py --bench                          # alle Messungen neu (statt Cache), ~5 min
poetry run python site/build.py --timeline 8                     # mehr Zwischenstände in der Zeitreise
poetry run python site/build.py --world wide                     # nur Gehirne der breiten Kreuzung (14 m)
```

`--world` wählt, welche Kreuzung die Seite zeigt: `any` (Standard, keine Einschränkung),
`narrow` (die alte 7-m-Kreuzung) oder `wide` (14 m, Läufe `runs/wide_*`). Ampel und Verkehrsrechner werden immer auf der
Kreuzung des besten Gehirns gemessen und aufgezeichnet.

Der Build nimmt außerdem für die Seite „Vergleich“ kurze Fahrten auf (`public/data/replays.js`,
ca. 0,4 MB): Ampel, Verkehrsrechner und das beste Gehirn fahren je 90 s mit genau demselben
Verkehr (wenig / mittel / viel). Die Aufnahme läuft bei jedem Build neu (ca. 20 s), passt sich
also automatisch an die aktuelle Simulation und das aktuell beste Gehirn an.

Ohne Optionen ist das Hauptgehirn `DEFAULT_MAIN` in `site/sitebuild/main.py` (aktuell `r9_E_s1/ckpt_1550`,
= `results/round9_best.npz`); `--previous` (Standard `r8_M_s1/ckpt_1398` = `results/round8_best.npz`) ist das Gehirn davor, das
auf „Vergleich“ daneben steht. Quelle (`--source auto`): Schnappschuss `results/site/` und
Trainingsläufe aus `runs/` werden zusammengeführt (ein Lauf in `runs/` ersetzt den gleichnamigen im
Schnappschuss); der Build schreibt den Schnappschuss danach neu. Die Zeitreise folgt der Kette der
Trainingsläufe, die aufeinander aufbauen (`"init"` in `args.json`). Ändert sich die
Simulation (`smartcrossing/sim.py`) oder eine Vergleichsfahrweise (`smartcrossing/baselines.py`), wird
automatisch neu gemessen: der Schlüssel des Bench-Caches enthält einen Hash beider Dateien.

Lokal ansehen, bevor du pushst:

```bash
cd site/public && python3 -m http.server 8000      # dann http://localhost:8000 öffnen
```

## Wenn etwas schiefgeht

- **Build bricht mit „OWN_FEATURES has … labels but sim.OBS_DIM is …“ ab**: Die Sensoren der
  Simulation haben sich geändert. Dann in `site/obs_labels.py` die Liste
  `OWN_FEATURES` (deutsche Beschriftungen) anpassen – das ist Absicht, damit die Seite nie
  falsch beschriftet.
- **„skip runs/…npz“** im Build: Diese Datei ist kein Gehirn (z. B. ein Trainingszustand)
  oder passt nicht zur Simulation; sie wird einfach ausgelassen.
- **Seite zeigt alte Daten**: Browser-Cache; die Dateien werden mit einer Versionsnummer
  geladen, ein normales Neuladen reicht nach einem neuen Deployment.
