# Website veröffentlichen (Cloudflare Pages, kostenlos)

Die Website ist rein statisch: HTML, CSS, JavaScript und eine Datendatei. Die Simulation
läuft komplett im Browser der Besucher, es gibt keinen Server und keine Datenbank.

## Was liegt wo?

| Ordner | Inhalt | ins Git? |
|---|---|---|
| `site/src/` | die handgeschriebenen Seiten (HTML/CSS/JS) | ja |
| `site/build.py` | Build-Skript: kopiert `site/src/`, `viewer/sim.js`, wählt Gehirne aus `runs/`, misst sie, schreibt die Daten | ja |
| `site/data/bench_cache.json` | Zwischenspeicher der Messungen (damit ein Neubau schnell geht) | ja |
| `site/public/` | **die fertige Website** – genau dieser Ordner wird veröffentlicht | **ja** |

Warum wird `site/public/` mit eingecheckt? Der Build braucht die Trainingsläufe in `runs/`
(gitignored, nur lokal vorhanden) sowie Python + Poetry. Cloudflare hat beides nicht. Also
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

Cloudflare merkt den Push auf `main` und veröffentlicht automatisch neu. Pushes auf andere
Branches bekommen eine eigene Vorschau-Adresse (`<branch>.<projekt>.pages.dev`) – praktisch,
um etwas vorher anzusehen.

Nützliche Optionen des Build-Skripts:

```bash
poetry run python site/build.py --main runs/es5/final_g310.npz   # bestimmtes Gehirn als "bestes" nehmen
poetry run python site/build.py --bench                          # alle Messungen neu (statt Cache), ~5 min
poetry run python site/build.py --timeline 8                     # mehr Zwischenstände in der Zeitreise
```

Ohne Optionen nimmt der Build pro Methode (Evolution `runs/es*`, PPO `runs/ppo*`) den
Checkpoint, der bei mittlerem Verkehr am besten fährt. Die Zeitreise folgt der Kette der
Trainingsläufe, die aufeinander aufbauen (`"init"` in `args.json`). Ändert sich die
Simulation (`smartcrossing/sim.py`), wird automatisch neu gemessen.

Lokal ansehen, bevor du pushst:

```bash
cd site/public && python3 -m http.server 8000      # dann http://localhost:8000 öffnen
```

## Wenn etwas schiefgeht

- **Build bricht mit „OWN_FEATURES has … labels but sim.OBS_DIM is …“ ab**: Die Sensoren der
  Simulation haben sich geändert. Dann in `smartcrossing/export_viewer.py` die Liste
  `OWN_FEATURES` (deutsche Beschriftungen) anpassen – das ist Absicht, damit die Seite nie
  falsch beschriftet.
- **„skip runs/…npz“** im Build: Diese Datei ist kein Gehirn (z. B. ein Trainingszustand)
  oder passt nicht zur Simulation; sie wird einfach ausgelassen.
- **Seite zeigt alte Daten**: Browser-Cache; die Dateien werden mit einer Versionsnummer
  geladen, ein normales Neuladen reicht nach einem neuen Deployment.
