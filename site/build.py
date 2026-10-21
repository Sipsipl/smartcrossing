"""Build the static website into site/public/ (commit the result; Cloudflare Pages
serves that folder as is, see docs/DEPLOY.md).

    poetry run python site/build.py                       # snapshot results/site/ + training runs in runs/
    poetry run python site/build.py --main runs/es5/final_g310.npz --bench

What it does:
1. copies the hand-written pages and scripts from site/src/ (incl. the simulator js/sim.js) and
   writes js/sim-config.js (SimConfig defaults and sim constants, from sim.js_constants),
2. picks policies from runs/*/ (.npz): the best checkpoint per method (ES: runs/es*,
   PPO: runs/ppo*), a "Zeitreise" timeline along the warm-start chain (args.json "init")
   of the best run, and the other runs as side notes,
3. summarises the training logs (runs/*/log.jsonl) into compact learning curves,
4. benchmarks the reference drivers and the chosen policies in the Python sim
   (throughput vs demand). Results are cached in site/data/bench_cache.json, keyed by
   policy weights, physics and the sim source, so rebuilding is fast and reproducible,
5. writes everything the pages need into public/data/site-data.js (no fetch needed).

runs/ is gitignored and usually holds only the latest runs, so by default (--source auto) the build
merges: the committed snapshot plus the training runs found in runs/ (a run in runs/ replaces the
snapshot's run of the same name). Training runs are runs/es*, runs/ppo*, runs/wide* and runs/r<N>_*
(training rounds, PPO); prof_*, smoke_* and other folders are ignored. Warm starts are followed via
args.json "init" (a run's state.npz), "init_policy" (a transplanted checkpoint, matched by weights) and
"init_bc" (the MLP teacher a brain was behaviour-cloned from, matched by weights). Brains are MLPs
(policy "layers") or set encoders (smartcrossing/policy_set.py: policy "arch" + flat "params").
A build from runs/ (or a merge) also writes a compact snapshot of everything it used
to results/site/ (committed): the chosen checkpoints (brains/<run>/<name>.npz, exact weights,
so the bench cache keys stay valid), the warm-start chain info and the thinned learning curves
(manifest.json). Without runs/ (a fresh clone) the build reads that snapshot instead
(--source), so `poetry run python site/build.py` works from a checkout. The committed
site/public/ is what gets deployed.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "site")]

from obs_labels import interface_key, obs_layout  # noqa: E402
from smartcrossing.policy import default_in_scale, init_params, layer_sizes, unpack  # noqa: E402
from smartcrossing.policy_compat import INTERFACE_KEYS, interface_cfg, load_adapted_policy, physics_of  # noqa: E402
from smartcrossing.sim import BRAKE_ACC, SimConfig, js_constants  # noqa: E402

SITE = ROOT / "site"
SRC = SITE / "src"
OUT = SITE / "public"
CACHE = SITE / "data" / "bench_cache.json"
RUNS = ROOT / "runs"
SNAPSHOT = ROOT / "results" / "site"  # committed copy of what the build uses from runs/ (see docstring)
SOURCE = {"kind": "runs", "manifest": None, "runs": set()}  # set in main(): "runs", "snapshot" or "merge"
FOUND = {}  # discovered checkpoints (set in main(); init_of matches transplanted warm starts against it)
MP_START = "fork" if "fork" in mp.get_all_start_methods() else "spawn"  # Windows has no fork

DEMANDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]  # up to the start page slider maximum (0.5 /s/arm = 120 cars/min)
# the site's main brain (wide crossing) and the end of the older line whose training history is
# shown before it (narrow crossing); override with --main / --history
# Checkpoints are named <run>/<file stem> (found in runs/ or in the snapshot); a path to an
# .npz anywhere (e.g. results/wide_s1ft2_ckpt2650.npz) works too if its weights match one.
DEFAULT_MAIN = ["r5_K_s1/ckpt_0475"]  # = results/round5_best.npz (round 5: r4 brain trained on, picked by stop-gate selection; not the run's last checkpoint)
DEFAULT_PREVIOUS = "r4_S_s1/ckpt_0205"  # = results/round4_best.npz, the main brain before round 5
DEFAULT_HISTORY = "ppo17/ckpt_5250"
# second brain for mixing on the start page; old brains (2 lamps, no brake light) cannot share a sim with the
# main brain, so none for now (was wide_s0ft/ckpt_2850 = results/wide_s0ft_ckpt2850.npz)
DEFAULT_MIX = ""
TIMELINE_DEMANDS = [0.05, 0.15]
BENCH_ENVS, BENCH_SECONDS = 8, 120.0
BENCH_SEEDS = (1000, 2000, 3000)  # traffic seeds, the same for all drivers (paired); results are averaged
# Plain German names for known runs: run id -> (short label for chart markers, label, one-line
# explanation). Shown on "Wie lernen sie?" instead of the raw ids; unmapped runs fall back to the id.
RUN_INFO = {
    "es1": ("E1", "Evolution, Versuch 1: Stehenbleiben",
            "Irrweg: Die Autos lernten, einfach stehen zu bleiben – Stehen kostete nichts, Unfälle schon."),
    "es2": ("E2", "Evolution, Versuch 2: Stehen kostet jetzt",
            "Erster Erfolg: Autos fahren zum Ziel, aber Rechtsabbiegen klappt nie (alle landen im Bordstein)."),
    "es3": ("E3", "Evolution, Versuch 3: mehr ausprobieren",
            "Größere Zufallsänderungen: kaum Fortschritt, die Rechtskurve bleibt das Problem."),
    "es4": ("E4", "Evolution, Versuch 4: Einfahrt blockieren",
            "Irrweg: Die Autos blockierten die Einfahrt – wer gar nicht erst reinfährt, kann nicht verunglücken."),
    "es5": ("E5", "Evolution, Versuch 5: Weg entlang der Straße messen",
            "Fortschritt wird entlang der Straße gemessen, wartende Autos kosten Punkte: jetzt klappen alle drei Richtungen meistens."),
    "ppo1a": ("P1", "PPO Versuch 1: Schleichen",
              "Irrweg: Die Autos kriechen nur noch und kommen kaum an."),
    "ppo1b": ("P1b", "PPO Versuch 1b: schneller lernen",
              "Irrweg: Größere Lernschritte – die Autos bleiben trotzdem stehen."),
    "ppo2": ("P2", "PPO Versuch 2: falsche Ausfahrt",
             "Irrweg: Über die Hälfte nimmt einfach die bequemste Ausfahrt, weil die falsche Ausfahrt billiger war als Abbiegen."),
    "ppo3": ("P3", "PPO Versuch 3: härtere Strafen",
             "Irrweg: Mehr Strafe für Unfälle – die Autos nehmen weiter die falsche Ausfahrt."),
    "ppo4": ("P4", "PPO Versuch 4: mehr Autos pro Runde",
             "Irrweg: Mehr Simulationen pro Runde – dasselbe Problem mit der falschen Ausfahrt."),
    "ppo5": ("P5", "PPO Versuch 5: falsche Ausfahrt teurer",
             "Falsche Ausfahrt kostet jetzt mehr – dafür landen die meisten im Bordstein."),
    "ppo6": ("P6", "Mehr Punkte für Fortschritt",
             "Durchbruch: doppelt so viele Punkte für jeden Meter Richtung Ausfahrt – die meisten kommen an."),
    "ppo7": ("P7", "PPO Versuch 7: neu von vorn",
             "Gleiche Einstellungen wie „Ruhigeres Ausprobieren“, aber von Null an gelernt; endet am selben Punkt."),
    "ppo8": ("P8", "Ruhigeres Ausprobieren",
             "Weniger zufälliges Zappeln beim Üben. Gut bei wenig Verkehr, aber geübt wurde nur mit fast leeren Straßen."),
    "ppo9": ("P9", "Dichter Verkehr beim Üben",
             "Üben mit viel mehr Autos gleichzeitig: Begegnungen werden häufiger, die Unfälle gehen deutlich zurück."),
    "ppo10": ("P10", "PPO Versuch 10: Unfälle doppelt so teuer",
              "Kein Gewinn, früh abgebrochen."),
    "ppo11": ("P11", "PPO Versuch 11: weiter gucken (60 m)",
              "Die Autos sehen andere aus 60 statt 30 m – musste sich erst umgewöhnen, kein klarer Gewinn."),
    "ppo12": ("P12", "PPO Versuch 12: Unfälle 2,5-mal so teuer",
              "Etwas mehr Bremsen, aber nicht weniger Unfälle."),
    "ppo13": ("P13", "Feinschliff mit kleineren Lernschritten",
              "Kleinere Lernschritte: langsam, aber stetig weniger Unfälle."),
    "ppo14": ("P14", "PPO Versuch 14: 60 m Sicht, neu von vorn",
              "Irrweg: Die weit entfernten Autos überfluten das Gehirn mit großen Zahlen – alle landen im Bordstein."),
    "ppo15": ("P15", "PPO Versuch 15: 60 m Sicht, richtig skaliert",
              "Lernt genauso gut wie mit 30 m – die Sichtweite war nicht das Problem."),
    "ppo16": ("P16", "PPO Versuch 16: Wunschtempo statt Gaspedal",
              "Das Gehirn wählt eine Zielgeschwindigkeit statt Gas/Bremse. Anfangs vorsichtiger, am Ende gleich gut."),
    "ppo17": ("P17", "Noch feiner, mehr Weitblick",
              "Noch kleinere Lernschritte und längerer Blick auf Folgen: der beste Stand auf der alten, schmalen Kreuzung."),
    "wide_s1": ("B1", "Breite Kreuzung: neu gelernt",
                "Neue Kreuzung: Straßen doppelt so breit, Reifen rutschen bei zu schnellen Kurven. Von Null an gelernt."),
    "wide_s1r": ("B2", "Breite Kreuzung: mit Rückwärtsgang",
                 "Die Autos dürfen jetzt auch rückwärts fahren – sie tun es aber fast nie."),
    "wide_s1ft": ("B3", "Breite Kreuzung: Feinschliff",
                  "Kleinere Lernschritte: weniger Unfälle."),
    "wide_s1ft2": ("B4", "Breite Kreuzung: länger feinschleifen",
                   "Noch eine Stunde Feinschliff: nur noch etwas besser – das Hauptgehirn vor Runde 1."),
    "r1_x3": ("R1", "Runde 1: neue Regeln, altes Gehirn lernt weiter",
              "Neue Regeln: Wracks mit Warnblinker, Bremslicht, eine Lampe, Einfahrt nur rechts. Das alte Gehirn lernt weiter, Unfälle kosten jetzt dreimal so viel."),
    "r1_main_x3": ("R1+", "Runde 1: länger weitergelernt",
                   "Rund 40 % weniger Unfälle als das Gehirn davor, bei dichtem Verkehr aber noch viele – das Hauptgehirn vor Runde 2."),
    "r2_B_s1": ("R2", "Runde 2: mit Hindernissen geübt",
                "Im Training standen Wracks und parkende Autos im Weg: Es bremst viel öfter dahinter ab und reiht sich ein, statt hineinzufahren – das Hauptgehirn vor Runde 4."),
    "r2_A_s1": ("R2a", "Runde 2, Test: nur mehr Ausprobieren",
                "Ohne Hindernisse im Training: schon deutlich weniger Unfälle als vorher, aber längst nicht so sicher vor Hindernissen."),
    "r2_A_s2": ("R2a'", "Runde 2, Test: nur mehr Ausprobieren, zweiter Lauf",
                "Anderer Zufallsstart, fast dasselbe Ergebnis."),
    "r2_C_s1": ("R2c", "Runde 2, Test: Unfälle je nach eigenem Tempo bestraft",
                "Mit Hindernissen und tempoabhängiger Strafe: etwas schlechter als ohne die Strafe (bei einem Lauf nicht sicher)."),
    "r3_B_long": ("R3", "Runde 3: Hindernis-Gehirn länger trainiert",
                  "Weitergelernt: kaum weniger Unfälle – mehr vom Gleichen half nicht."),
    "r2_B_s2": ("R2'", "Runde 2: mit Hindernissen, zweiter Lauf",
                "Gleiches Rezept, anderer Zufallsstart: deutlich mehr Unfälle – der erste Lauf hatte auch Glück."),
    "r2_C_s2": ("R2c'", "Runde 2, Test: tempoabhängige Strafe, zweiter Lauf",
                "Bestätigt: mit tempoabhängiger Strafe mehr Unfälle bei dichtem Verkehr."),
    "r4_S_s1": ("R4", "Runde 4: ein Mini-Netz für jeden Nachbarn + Aufmerksamkeit",
                "Schaut auf bis zu 8 Nachbarn, jeder geht durch dasselbe kleine Netz, dann entscheidet eine „Aufmerksamkeit“, wer gerade zählt. "
                "Erst hat es das vorige Gehirn nachgeahmt, dann 12 Minuten weitergelernt – das Hauptgehirn bis Runde 5."),
    "r4_S_s2": ("R4'", "Runde 4: Mini-Netz + Aufmerksamkeit, zweiter Lauf",
                "Anderer Zufallsstart, ähnlich gut – bei sehr dichtem Verkehr bleiben aber mehr Autos mitten auf der Kreuzung stecken."),
    "r4_M_s1": ("R4m", "Runde 4, Vergleich: bisheriges Netz weitertrainiert",
                "Gleich lange (12 Minuten) mit dem bisherigen Netz (64 × 64) weitergelernt: nicht besser als vorher, bei dichtem Verkehr eher mehr Unfälle."),
    "r4_M_s2": ("R4m'", "Runde 4, Vergleich: bisheriges Netz, zweiter Lauf",
                "Anderer Zufallsstart, ähnlich."),
    "r4_W_s1": ("R4w", "Runde 4, Vergleich: doppelt so breites Netz",
                "Mehr Neuronen (128 × 128) helfen auch – aber nicht so viel wie das Mini-Netz pro Nachbar."),
    "r4_W_s2": ("R4w'", "Runde 4, Vergleich: breites Netz, zweiter Lauf",
                "Anderer Zufallsstart, ähnlich."),
    "r5_K_s1": ("R5", "Runde 5: weitergelernt (Kontrolle)",
                "Das aktuelle Gehirn: das Gehirn aus Runde 4 hat unverändert weitertrainiert. Gewählt ist ein Zwischenstand (Schritt 475) mit wenigen Unfällen. "
                "Ein stehendes Hindernis umfährt es zu etwa 99 % sicher, hält aber nur in rund 70 % der Fälle davor an – im weiteren Training ließ das Anhalten nach."),
    "r5_K_s2": ("R5'", "Runde 5: weitergelernt, zweiter Lauf",
                "Anderer Zufallsstart: bei dichtem Verkehr noch mehr Autos durch, dafür bleiben die Unfälle bei sehr dichtem Verkehr hoch."),
    "r5_D_s1": ("R5d", "Runde 5, Test: Training mit dichtem Verkehr",
                "Mehr Verkehr im Training: weniger Unfälle, dafür mehr Autos, die lange festsitzen – und nicht mehr Durchsatz als vorher."),
    "r5_D_s2": ("R5d'", "Runde 5, Test: dichter Verkehr, zweiter Lauf", "Anderer Zufallsstart, ähnlich."),
    "r5_DW_s1": ("R5w", "Runde 5, Test: dichter Verkehr + Warten kostet",
                 "Zusätzlich wird Warten bestraft: weniger Unfälle als vorher, aber weiter viel Stau und kaum mehr Autos durch."),
    "r5_DW_s2": ("R5w'", "Runde 5, Test: dichter Verkehr + Warten kostet, zweiter Lauf", "Anderer Zufallsstart, ähnlich."),
    "r5_DWX_s1": ("R5x", "Runde 5, Test: dichter Verkehr, Warten kostet, Unfälle sechsmal so teuer",
                  "Noch weniger Unfälle, dafür Stau statt Crash (nur ein kürzerer Lauf)."),
    "r1_x1": ("R1a", "Runde 1, Test: Unfälle so teuer wie bisher",
              "Kurzer Test: ähnlich gut wie mit dreifacher Strafe – der Unterschied ist Zufall."),
    "r1_x10": ("R1b", "Runde 1, Test: Unfälle zehnmal so teuer",
               "Kurzer Test: zu viel Strafe – weniger Autos kommen durch, und Unfälle gibt es nicht weniger."),
    "wide_s0": ("B1'", "Breite Kreuzung, zweiter Lauf",
                "Gleiches Rezept, anderer Zufallsstart (noch ohne Rückwärtsgang)."),
    "wide_s0r": ("B2'", "Zweiter Lauf: mit Rückwärtsgang",
                 "Anderer Zufallsstart, mit Rückwärtsgang weitergelernt."),
    "wide_s0ft": ("B3'", "Zweiter Lauf: Feinschliff",
                  "Anderer Zufallsstart, genauso fein geschliffen: fährt auch rechts, ein paar Unfälle mehr."),
    "wide_noman_s0": ("B0", "Ohne Abbiege-Wunsch",
                      "Die Autos erfahren nicht, ob sie links, rechts oder geradeaus wollen – nur, wo das Ziel liegt. Lernt langsamer."),
}

# One short line per phase of the main training chain, shown next to the numbered markers
# ①②③ of the learning curve ("what changed -> what happened"). Plain words, no run ids.
STEP_INFO = {
    "es2": "Start mit Zufallsgehirn; Stehenbleiben kostet Punkte → die Autos fahren los, aber Rechtsabbiegen endet im Bordstein",
    "es3": "Größere Zufallsänderungen → kaum Fortschritt",
    "es5": "Weg wird entlang der Straße gemessen, Warten kostet → alle drei Richtungen klappen meistens",
    "ppo6": "Start bei Null; Punkte für jeden Meter Richtung Ziel → die meisten kommen an",
    "ppo8": "Ruhiger ausprobieren, weniger Zappeln → gut, aber nur auf fast leeren Straßen geübt",
    "ppo9": "Üben mit viel dichterem Verkehr → Kurve fällt erst, weil es schwerer wird; dann weniger Unfälle",
    "ppo13": "Kleinere Lernschritte → stetig besser; gegen Ende noch mehr Verkehr, daher wieder etwas mehr Unfälle",
    "ppo17": "Noch kleinere Schritte, mehr Blick auf die Folgen → bester Stand auf der alten, schmalen Kreuzung",
    "wide_s1": "Neue Kreuzung: Straße doppelt so breit, Reifen rutschen in zu schnellen Kurven → von Null an neu gelernt, die Kurve beginnt wieder unten",
    "wide_s1r": "Rückwärtsgang dazu → kaum benutzt, es geht weiter bergauf",
    "wide_s1ft": "Kleinere Lernschritte → weniger Unfälle",
    "wide_s1ft2": "Länger feinschleifen → nur noch wenig besser: das Hauptgehirn bis Runde 1",
    "r1_x3": "Neue Regeln: Wracks mit Warnblinker, Bremslicht, eine Lampe, Einfahrt nur rechts; Unfälle kosten dreimal so viel, geübt wird mit dichterem Verkehr → die Kurve fällt erst, dann lernt das alte Gehirn um",
    "r1_main_x3": "Länger weitergelernt → rund 40 % weniger Unfälle als vorher: das Hauptgehirn bis Runde 2",
    "r2_B_s1": "Dazu Hindernisse im Training (Wracks, parkende Autos) → bremst viel öfter dahinter ab und reiht sich ein; noch Unfälle mitten auf der Kreuzung bei dichtem Verkehr",
    "r4_S_s1": "Neues Gehirn: ein gemeinsames Mini-Netz für jeden der bis zu 8 Nachbarn plus Aufmerksamkeit; erst das vorige Gehirn nachgeahmt, dann weitergelernt → im Vergleich (gleich lange trainiert wie das bisherige und ein doppelt so breites Netz) die wenigsten Unfälle",
    "r5_K_s1": "Dasselbe Gehirn weitertrainiert, ein Zwischenstand mit wenigen Unfällen und hohem Durchsatz → bei dichtem Verkehr mehr Autos durch und weniger Unfälle; vor einem stehenden Hindernis hält es aber nur in etwa 70 % der Fälle an, sonst fährt es herum",
}


# English names / notes of the runs: run id -> (label, note); same keys as RUN_INFO.
RUN_INFO_EN = {
    "es1": ("Evolution, attempt 1: standing still",
            "Dead end: the cars learned to just stand still – standing cost nothing, crashes did."),
    "es2": ("Evolution, attempt 2: standing now costs",
            "First success: cars drive to their goal, but turning right never works (they all end up on the curb)."),
    "es3": ("Evolution, attempt 3: trying more",
            "Bigger random changes: hardly any progress, the right turn stays the problem."),
    "es4": ("Evolution, attempt 4: blocking the entrance",
            "Dead end: the cars blocked the entrance – a car that never enters can't crash."),
    "es5": ("Evolution, attempt 5: measuring progress along the road",
            "Progress is measured along the road and waiting cars lose points: now all three directions mostly work."),
    "ppo1a": ("PPO attempt 1: creeping",
              "Dead end: the cars only crawl along and hardly ever arrive."),
    "ppo1b": ("PPO attempt 1b: learning faster",
              "Dead end: bigger learning steps – the cars still stand still."),
    "ppo2": ("PPO attempt 2: wrong exit",
             "Dead end: over half simply take the most convenient exit, because the wrong exit was cheaper than turning."),
    "ppo3": ("PPO attempt 3: tougher penalties",
             "Dead end: bigger penalty for crashes – the cars still take the wrong exit."),
    "ppo4": ("PPO attempt 4: more cars per round",
             "Dead end: more simulations per round – the same wrong-exit problem."),
    "ppo5": ("PPO attempt 5: wrong exit costs more",
             "The wrong exit costs more now – but most cars end up on the curb instead."),
    "ppo6": ("More points for progress",
             "Breakthrough: twice the points for every metre towards the exit – most cars arrive."),
    "ppo7": ("PPO attempt 7: starting fresh",
             "Same settings as “Calmer experimenting”, but learned from scratch; ends at the same point."),
    "ppo8": ("Calmer experimenting",
             "Less random twitching while practising. Good in light traffic, but it only practised on nearly empty roads."),
    "ppo9": ("Dense traffic while practising",
             "Practising with many more cars at once: encounters get more frequent, crashes drop clearly."),
    "ppo10": ("PPO attempt 10: crashes twice as costly",
              "No gain, stopped early."),
    "ppo11": ("PPO attempt 11: looking further (60 m)",
              "The cars see others from 60 instead of 30 m – needed time to adjust, no clear gain."),
    "ppo12": ("PPO attempt 12: crashes 2.5 times as costly",
              "Slightly more braking, but no fewer crashes."),
    "ppo13": ("Fine-tuning with smaller learning steps",
              "Smaller learning steps: slowly but steadily fewer crashes."),
    "ppo14": ("PPO attempt 14: 60 m view, from scratch",
              "Dead end: the far-away cars flood the brain with big numbers – everyone ends up on the curb."),
    "ppo15": ("PPO attempt 15: 60 m view, scaled properly",
              "Learns just as well as with 30 m – the viewing distance was not the problem."),
    "ppo16": ("PPO attempt 16: target speed instead of gas pedal",
              "The brain picks a target speed instead of gas/brake. More careful at first, equally good in the end."),
    "ppo17": ("Even finer, more foresight",
              "Even smaller learning steps and a longer look at the consequences: the best result on the old, narrow crossing."),
    "wide_s1": ("Wide crossing: learned anew",
                "New crossing: roads twice as wide, tyres slip in corners taken too fast. Learned from scratch."),
    "wide_s1r": ("Wide crossing: with reverse gear",
                 "The cars may now also drive backwards – but they hardly ever do."),
    "wide_s1ft": ("Wide crossing: fine-tuning",
                  "Smaller learning steps: fewer crashes."),
    "wide_s1ft2": ("Wide crossing: fine-tuning for longer",
                   "Another hour of fine-tuning: only a little better – the main brain before round 1."),
    "r1_x3": ("Round 1: new rules, old brain keeps learning",
              "New rules: wrecks with hazard lights, brake light, one lamp, entering only from the right. The old brain keeps learning, crashes now cost three times as much."),
    "r1_main_x3": ("Round 1: learned for longer",
                   "About 40% fewer crashes than the brain before, but still many in dense traffic – the main brain before round 2."),
    "r2_B_s1": ("Round 2: practised with obstacles",
                "Wrecks and parked cars stood in the way during training: it brakes behind them far more often and queues up instead of driving into them – the main brain before round 4."),
    "r2_A_s1": ("Round 2, test: just more experimenting",
                "Without obstacles in training: already clearly fewer crashes than before, but far less safe around obstacles."),
    "r2_A_s2": ("Round 2, test: just more experimenting, second run",
                "Different random start, almost the same result."),
    "r2_C_s1": ("Round 2, test: crashes penalised by own speed",
                "With obstacles and a speed-dependent penalty: slightly worse than without the penalty (not certain from one run)."),
    "r3_B_long": ("Round 3: obstacle brain trained for longer",
                  "Kept learning: hardly any fewer crashes – more of the same didn't help."),
    "r2_B_s2": ("Round 2: with obstacles, second run",
                "Same recipe, different random start: clearly more crashes – the first run was also lucky."),
    "r2_C_s2": ("Round 2, test: speed-dependent penalty, second run",
                "Confirmed: with the speed-dependent penalty there are more crashes in dense traffic."),
    "r4_S_s1": ("Round 4: a mini-network for every neighbour + attention",
                "Looks at up to 8 neighbours, each goes through the same small network, then an “attention” decides who matters right now. "
                "It first imitated the previous brain, then kept learning for 12 minutes – the main brain up to round 5."),
    "r4_S_s2": ("Round 4: mini-network + attention, second run",
                "Different random start, similarly good – but in very dense traffic more cars get stuck in the middle of the crossing."),
    "r4_M_s1": ("Round 4, comparison: previous network trained further",
                "Trained further for the same time (12 minutes) with the previous network (64 × 64): not better than before, rather more crashes in dense traffic."),
    "r4_M_s2": ("Round 4, comparison: previous network, second run",
                "Different random start, similar."),
    "r4_W_s1": ("Round 4, comparison: network twice as wide",
                "More neurons (128 × 128) help too – but not as much as the mini-network per neighbour."),
    "r4_W_s2": ("Round 4, comparison: wide network, second run",
                "Different random start, similar."),
    "r5_K_s1": ("Round 5: trained on (control)",
                "The current brain: the round 4 brain trained on, unchanged. An intermediate step (step 475) is chosen, with few crashes. "
                "It gets past a standing obstacle safely in about 99% of cases, but stops in front of it in only about 70% – stopping faded with further training."),
    "r5_K_s2": ("Round 5: trained on, second run",
                "Different random start: even more cars get through in dense traffic, but crashes stay high in very dense traffic."),
    "r5_D_s1": ("Round 5, test: training with dense traffic",
                "More traffic in training: fewer crashes, but more cars stuck for a long time – and no more throughput than before."),
    "r5_D_s2": ("Round 5, test: dense traffic, second run", "Different random start, similar."),
    "r5_DW_s1": ("Round 5, test: dense traffic + waiting costs",
                 "Waiting is penalised on top: fewer crashes than before, but still a lot of jamming and hardly more cars through."),
    "r5_DW_s2": ("Round 5, test: dense traffic + waiting costs, second run", "Different random start, similar."),
    "r5_DWX_s1": ("Round 5, test: dense traffic, waiting costs, crashes six times as costly",
                  "Even fewer crashes, but jams instead of crashes (a single, shorter run)."),
    "r1_x1": ("Round 1, test: crashes as costly as before",
              "Short test: about as good as with triple penalty – the difference is chance."),
    "r1_x10": ("Round 1, test: crashes ten times as costly",
               "Short test: too much penalty – fewer cars get through, and crashes aren't any rarer."),
    "wide_s0": ("Wide crossing, second run",
                "Same recipe, different random start (still without reverse gear)."),
    "wide_s0r": ("Second run: with reverse gear",
                 "Different random start, kept learning with reverse gear."),
    "wide_s0ft": ("Second run: fine-tuning",
                  "Different random start, fine-tuned the same way: also drives on the right, a few more crashes."),
    "wide_noman_s0": ("Without turn wish",
                      "The cars aren't told whether they want to go left, right or straight – only where the goal is. Learns more slowly."),
}

# English version of STEP_INFO (same keys).
STEP_INFO_EN = {
    "es2": "Start with a random brain; standing still costs points → the cars set off, but turning right ends on the curb",
    "es3": "Bigger random changes → hardly any progress",
    "es5": "Progress is measured along the road, waiting costs → all three directions mostly work",
    "ppo6": "Start from zero; points for every metre towards the goal → most cars arrive",
    "ppo8": "Calmer experimenting, less twitching → good, but only practised on nearly empty roads",
    "ppo9": "Practising with much denser traffic → the curve drops at first because it gets harder; then fewer crashes",
    "ppo13": "Smaller learning steps → steadily better; towards the end even more traffic, so slightly more crashes again",
    "ppo17": "Even smaller steps, more look at the consequences → best result on the old, narrow crossing",
    "wide_s1": "New crossing: road twice as wide, tyres slip in corners taken too fast → learned from scratch, the curve starts low again",
    "wide_s1r": "Reverse gear added → hardly used, it keeps climbing",
    "wide_s1ft": "Smaller learning steps → fewer crashes",
    "wide_s1ft2": "Fine-tuning for longer → only a little better: the main brain up to round 1",
    "r1_x3": "New rules: wrecks with hazard lights, brake light, one lamp, entering only from the right; crashes cost three times as much, practising with denser traffic → the curve drops at first, then the old brain relearns",
    "r1_main_x3": "Kept learning for longer → about 40% fewer crashes than before: the main brain up to round 2",
    "r2_B_s1": "Plus obstacles in training (wrecks, parked cars) → brakes behind them far more often and queues up; still crashes in the middle of the crossing in dense traffic",
    "r4_S_s1": "New brain: one shared mini-network for each of the up to 8 neighbours plus attention; first imitated the previous brain, then kept learning → fewest crashes in the comparison (trained for the same time as the previous one and a network twice as wide)",
    "r5_K_s1": "The same brain trained on, an intermediate step with few crashes and high throughput → more cars through and fewer crashes in dense traffic; but in front of a standing obstacle it stops in only about 70% of cases, otherwise it drives around",
}


def run_info(run: str | None) -> dict:
    """Names / notes of a run for the pages: German in label / note / step, English in *_en."""
    short, label, note = RUN_INFO.get(run, (run, run, None)) if run else (None, None, None)
    label_en, note_en = RUN_INFO_EN.get(run, (label, None)) if run else (None, None)
    return {"short": short, "label": label, "label_en": label_en, "note": note, "note_en": note_en,
            "step": STEP_INFO.get(run) if run else None, "step_en": STEP_INFO_EN.get(run) if run else None}


# nav entries: (file, German label); the English labels are js/i18n-en.js nav.<file>
PAGES = [
    ("index.html", "Start"),
    ("sehen.html", "Was sieht ein Auto?"),
    ("tun.html", "Was kann es tun?"),
    ("gehirn.html", "Das Gehirn"),
    ("lernen.html", "Wie lernen sie?"),
    ("vergleich.html", "Vergleich"),
]


# ------------------------------------------------------------------ checkpoints
def method_of(run: str) -> str | None:
    if run.startswith("es"):
        return "es"
    # wide_*: PPO on the wide crossing; r<N>_*: training round N (PPO, warm-started)
    if run.startswith("ppo") or run.startswith("wide") or re.match(r"r\d+_", run):
        return "ppo"
    return None


def from_runs(run: str) -> bool:
    """True if `run` is read from runs/ (source runs, or merge and the run exists in runs/)."""
    return SOURCE["kind"] == "runs" or (SOURCE["kind"] == "merge" and run in SOURCE["runs"])


def world_of(physics: dict) -> str:
    """'narrow' (original 7 m road) or 'wide' (14 m road) from a checkpoint's physics."""
    return "narrow" if physics.get("W", 3.5) < 5 else "wide"


def ckpt_meta(path: Path) -> dict:
    with np.load(path) as z:
        extra = json.loads(str(z["extra"])) if "extra" in z.files else {}
        cfg = json.loads(str(z["cfg"])) if "cfg" in z.files else {}
    gen = None
    for k in ("gen", "iter", "update", "step", "epoch"):
        if isinstance(extra.get(k), (int, float)):
            gen = int(extra[k])
            break
    if gen is None:  # e.g. final_g310.npz
        m = re.search(r"(?:g|gen|it|iter|u)(\d+)", path.stem)
        gen = int(m.group(1)) if m else None
    return {"gen": gen, "cfg": cfg, "extra": extra}


class Brain:
    """One brain of the site: a policy dict in the export_json / load_adapted_policy format, either an MLP
    (``layers``: W, b per layer) or a set encoder (smartcrossing/policy_set.py: ``arch`` + flat ``params``).
    All set / MLP branching of the build lives here."""

    def __init__(self, d: dict):
        self.d = d

    @property
    def is_set(self) -> bool:
        return bool(self.d.get("arch")) and self.d["arch"].get("type") == "set"

    def vector(self) -> np.ndarray:
        """The flat parameter vector (policy.py layout)."""
        if self.is_set:
            return np.asarray(self.d["params"], dtype=float)
        return np.concatenate([np.concatenate([np.ravel(np.asarray(l["W"], dtype=float)), np.asarray(l["b"], dtype=float)])
                               for l in self.d["layers"]])

    def hash(self) -> str:
        return hashlib.sha1(self.vector().tobytes() + json.dumps(self.d["in_scale"]).encode()).hexdigest()[:16]

    def payload(self) -> tuple:
        """Bench / replay payload for policy_driver: (params, sizes, in_scale[, arch])."""
        return (self.vector().tolist(), self.d["sizes"], self.d["in_scale"], *([self.d["arch"]] if self.is_set else []))

    def rounded(self, digits: int = 5) -> dict:
        """Shrink the JSON: weights rounded to `digits` decimals."""
        r = lambda a: np.round(np.asarray(a, dtype=float), digits).tolist()  # noqa: E731
        d = self.d
        if self.is_set:
            return {**d, "in_scale": r(d["in_scale"]), "params": r(d["params"])}
        return {**d, "in_scale": r(d["in_scale"]), "layers": [{"W": r(l["W"]), "b": r(l["b"])} for l in d["layers"]]}

    def snapshot(self) -> tuple[dict, dict]:
        """For results/site/brains/*.npz: (info fields n_layers + policy without weights,
        arrays in_scale + params (set encoder) or W<i> / b<i> (MLP))."""
        d = self.d
        info = {"n_layers": 0 if self.is_set else len(d["layers"]),
                "policy": {k: v for k, v in d.items() if k not in ("layers", "in_scale", "params")}}
        arrays = {"in_scale": np.asarray(d["in_scale"], dtype=float)}
        if self.is_set:  # set encoder: flat params, the arch stays in info["policy"]
            arrays["params"] = np.asarray(d["params"], dtype=float)
        for i, l in enumerate(d.get("layers", [])):
            arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(l["W"], dtype=float), np.asarray(l["b"], dtype=float)
        return info, arrays

    @classmethod
    def from_snapshot(cls, z, info: dict) -> "Brain":
        """Inverse of snapshot (z: the opened npz)."""
        pol = {**info["policy"], "in_scale": z["in_scale"].tolist()}
        if "params" in z.files:  # set encoder: flat params (its arch is in info["policy"])
            pol["params"] = z["params"].tolist()
        else:
            pol["layers"] = [{"W": z[f"W{i}"].tolist(), "b": z[f"b{i}"].tolist()} for i in range(info["n_layers"])]
        return cls(pol)


def policy_hash(pol: dict) -> str:
    return Brain(pol).hash()


def _entry(run: str, stem: str, file: str, method: str, gen, physics: dict, pol: dict) -> dict:
    return {"id": f"{run}/{stem}", "run": run, "file": file, "method": method, "gen": gen,
            "physics": physics, "policy": pol, "adapted": pol.get("adapted"), "hash": policy_hash(pol)}


def discover(world: str = "any") -> dict:
    """All loadable checkpoints of runs/es*, runs/ppo* and runs/wide* (or of the snapshot):
    id -> info (only those trained on `world` = narrow / wide crossing, unless 'any')."""
    found = {}
    if SOURCE["kind"] in ("snapshot", "merge"):
        for f in sorted((SNAPSHOT / "brains").glob("*/*.npz")):
            with np.load(f) as z:
                info = json.loads(str(z["info"]))
                pol = Brain.from_snapshot(z, info).d
            # snapshots written before the interface keys existed: lamp count / neighbour layout from the
            # checkpoint's own training cfg (old brains: 2 lamps, acceleration feature)
            phys = {**info["physics"], **{k: v for k, v in physics_of(pol.get("cfg") or {}).items() if k in INTERFACE_KEYS}}
            if world != "any" and world_of(phys) != world:
                continue
            if from_runs(info["run"]):
                continue
            e = _entry(info["run"], f.stem, info["file"], info["method"], info["gen"], phys, pol)
            found[e["id"]] = {**e, "snap": True}
        if SOURCE["kind"] == "snapshot":
            return found
    if not RUNS.exists():
        return found
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        method = method_of(run_dir.name)
        if not method or not from_runs(run_dir.name):
            continue
        for f in sorted(run_dir.glob("*.npz")):
            try:
                pol = load_adapted_policy(f)
            except Exception as e:  # incompatible / broken checkpoint: skip loudly
                print(f"  skip {f.relative_to(ROOT)}: {e}")
                continue
            meta = ckpt_meta(f)
            if world != "any" and world_of(physics_of(meta["cfg"])) != world:
                continue
            e = _entry(run_dir.name, f.stem, f.relative_to(ROOT).as_posix(), method, meta["gen"], physics_of(meta["cfg"]), pol)
            found[e["id"]] = e
    return found


def init_of(run: str) -> tuple[str, int] | None:
    """(run, generation) of the checkpoint `run` was warm-started from, if it is a known run."""
    if not from_runs(run):
        r = SOURCE["manifest"]["runs"].get(run, {}).get("init")
        return tuple(r) if r else None
    p = RUNS / run / "args.json"
    args = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    init = args.get("init") or ""
    # actor transplanted from a checkpoint (init_policy) or behaviour-cloned from a teacher (init_bc): find it by its weights
    src = args.get("init_policy") or args.get("init_bc")
    if not init and src:
        ip = ROOT / src
        if not ip.is_file():
            return None
        h = policy_hash(load_adapted_policy(ip))
        # a run whose first checkpoint is an exact copy of the same brain is not its origin: prefer gen > 0
        cands = [c for c in FOUND.values() if c["hash"] == h and c["run"] != run]
        c = min(cands, key=lambda c: not c["gen"], default=None)
        return (c["run"], c["gen"] or 0) if c else None
    if not init:
        return None
    ip = ROOT / init
    if not ip.exists() or not method_of(ip.parent.name):
        return None
    return ip.parent.name, ckpt_meta(ip)["gen"] or 0


def chain_of(run: str) -> list[tuple[str, int]]:
    """Warm-start chain ending in `run`: [(run, gen offset), ...] oldest first."""
    chain, gens, seen = [], {}, set()
    cur = run
    while cur and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        nxt = init_of(cur)
        if not nxt:
            break
        gens[cur] = nxt[1]
        cur = nxt[0]
    chain.reverse()
    out, offset = [], 0
    for i, r in enumerate(chain):
        if i > 0:
            offset = dict(out)[chain[i - 1]] + gens[r]
        out.append((r, offset))
    return out


# ------------------------------------------------------------------ bench
def policy_driver(sim, payload):
    """PolicyDriver for a bench / replay payload (params, sizes, in_scale[, arch]; arch None = MLP)."""
    from smartcrossing.policy import PolicyDriver

    params, sizes, in_scale, *rest = payload
    return PolicyDriver(sim, np.asarray(params), sizes, in_scale=np.asarray(in_scale), archs=rest[0] if rest else None)


def make_driver(sim, kind: str, payload):
    """Driver of a bench / replay job: a rule-based LaneDriver (kind "baseline", payload = its name) or a policy."""
    if kind == "baseline":
        from smartcrossing.baselines import LaneDriver

        return LaneDriver(sim, payload)
    return policy_driver(sim, payload)


def _bench_job(job):
    """One bench run (smartcrossing.evalset.bench_one), its KEEP_KEYS metrics."""
    from smartcrossing.evalset import bench_one, keep

    kind, payload, physics, demand, seed = job
    cfg = SimConfig(**{**physics, "demand": demand})
    return keep(bench_one(lambda sim: make_driver(sim, kind, payload), cfg, seed, BENCH_ENVS, BENCH_SECONDS))


def _bench_ck(key, physics, d, seed, sim_src):
    return f"{key}|{json.dumps(physics, sort_keys=True)}|{d}|{BENCH_ENVS}x{BENCH_SECONDS}@{seed}|{sim_src}"


def _mean_results(rs: list[dict]) -> dict:
    """Mean over the traffic seeds, field by field (None if a seed has no value)."""
    out = {}
    for k in rs[0]:
        v = [r[k] for r in rs if r[k] is not None]
        out[k] = round(sum(v) / len(v), 3) if v else None
    return out


def run_bench(entries: list[tuple[str, str, object, dict, list]], force: bool, workers: int) -> dict:
    """entries: (key, kind, payload, physics, demands) -> {key: {demand: result}}."""
    sim_src = hashlib.sha1((ROOT / "smartcrossing" / "sim.py").read_bytes()
                           + (ROOT / "smartcrossing" / "baselines.py").read_bytes()).hexdigest()[:12]
    cache, n_stored = {}, 0
    if CACHE.exists() and not force:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
        n_stored = len(cache)
        cache = {k: v for k, v in cache.items() if k.endswith(f"|{sim_src}")}  # drop results of older sim versions
    if len(cache) < n_stored:
        CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
    jobs, keys = [], []
    for key, kind, payload, physics, demands in entries:
        for d in demands:
            for seed in BENCH_SEEDS:
                ck = _bench_ck(key, physics, d, seed, sim_src)
                if ck in cache:
                    continue
                jobs.append((kind, payload, physics, d, seed))
                keys.append(ck)
    if jobs:
        print(f"  benchmarking {len(jobs)} runs ({BENCH_ENVS} envs x {BENCH_SECONDS:.0f} s each, {len(BENCH_SEEDS)} traffic seeds) ...")
        t0 = time.time()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with mp.get_context(MP_START).Pool(workers) as pool:
            for i, (ck, res) in enumerate(zip(keys, pool.imap(_bench_job, jobs, chunksize=1))):
                cache[ck] = res
                if (i + 1) % 20 == 0 or i + 1 == len(jobs):  # save progress (an interrupted build resumes)
                    CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8", newline="\n")
                    print(f"    {i + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
        print(f"  bench done in {time.time() - t0:.0f} s")
    out = {}
    for key, kind, payload, physics, demands in entries:
        out.setdefault(key, {})
        for d in demands:
            out[key][f"{d:g}"] = _mean_results([cache[_bench_ck(key, physics, d, sd, sim_src)] for sd in BENCH_SEEDS])  # as JS String(d)
    return out


# ------------------------------------------------------------------ replays ("Vergleich" side by side)
REPLAY_DEMANDS = [0.05, 0.15, 0.3]
REPLAY_SEED, REPLAY_WARMUP, REPLAY_SECONDS = 7, 15.0, 90.0
GEOMETRY_KEYS = ("L", "W", "Rc", "a_lat_max", "v_reverse")  # the "world": same for all drivers compared
Q_POS, Q_TH, Q_LAMP = 0.05, 0.01, 100.0  # quantisation: 5 cm, 0.01 rad, lamp in 1/100


def _b64(a: np.ndarray) -> str:
    import base64

    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _replay_job(job):
    """Drive one sim (1 env) and record every car as a compact track:
    [t0, man, end_step, end_kind, qx0, qy0, qth0, width, b64, brake, wreck_from] where b64 holds per
    sample (dx, dy, dth[, lamp per channel]) as int8 (width 1) or int16 (width 2), deltas of the
    quantised values (exact reconstruction), brake the sample ranges [start, end) with the
    brake lights on (acceleration < sim.BRAKE_ACC), flattened, and wreck_from the first sample at which
    the car is a wreck (-1: never; a wreck stays cfg.wreck_time, its samples continue the track).
    Plus crash marks and the queue per arm. "lamps" = lamp channels per sample (0 for the baselines)."""
    from smartcrossing.sim import DONE_NAMES, Sim

    kind, payload, physics, demand = job
    cfg = SimConfig(**{**physics, "demand": demand})
    sim = Sim(cfg, 1, seed=REPLAY_SEED)
    drv = make_driver(sim, kind, payload)
    lamps = 0 if kind == "baseline" else cfg.n_signal
    tracks, marks, queue = {}, [], []
    n_warm, n = int(round(REPLAY_WARMUP / cfg.dt)), int(round(REPLAY_SECONDS / cfg.dt))
    for k in range(-n_warm, n):
        sim.step(drv.act())
        if k < 0:
            continue
        for slot in np.nonzero(sim.done_kind[0])[0]:
            tr = tracks.get(int(sim.done_uid[0, slot]))
            if tr is not None:
                tr["end"] = (k, int(sim.done_kind[0, slot]))
        for _, ev, x, y in sim.events:
            marks.append([k, 1 if ev == "crash" else 2, int(round(x / Q_POS)), int(round(y / Q_POS))])
        for slot in np.nonzero(sim.alive[0] | sim.wreck[0])[0]:
            uid = int(sim.uid[0, slot])
            tr = tracks.get(uid)
            if tr is None:
                if sim.wreck[0, slot]:  # a wreck from before the recording started
                    continue
                tr = tracks[uid] = {"t0": k, "man": int(sim.man[0, slot]), "s": [], "b": [], "end": (None, 0), "w": -1}
            if sim.wreck[0, slot] and tr["w"] < 0:
                tr["w"] = len(tr["s"])
            tr["b"].append(bool(sim.acc[0, slot] < BRAKE_ACC))
            tr["s"].append((sim.x[0, slot], sim.y[0, slot], sim.th[0, slot], *(sim.sig[0, slot, :lamps] if lamps else ())))
        queue.append(np.minimum(sim.queue[0], 255))
    out = []
    for uid, tr in tracks.items():
        s = np.asarray(tr["s"], dtype=float)
        q = np.empty_like(s, dtype=np.int64)
        q[:, 0] = np.round(s[:, 0] / Q_POS)
        q[:, 1] = np.round(s[:, 1] / Q_POS)
        q[:, 2] = np.round(np.unwrap(s[:, 2]) / Q_TH)
        d = q.copy()
        d[1:, :3] = q[1:, :3] - q[:-1, :3]
        d[0, :3] = 0
        if lamps:
            d[:, 3:] = np.round(np.clip(s[:, 3:], -1, 1) * Q_LAMP)
        width = 1 if np.abs(d).max() <= 127 else 2
        end_k, end_kind = tr["end"]
        b = np.diff(np.concatenate([[0], np.asarray(tr["b"], dtype=np.int8), [0]]))
        runs = np.stack([np.nonzero(b == 1)[0], np.nonzero(b == -1)[0]], 1).ravel().tolist()
        out.append([tr["t0"], tr["man"], end_k if end_k is not None else -1, end_kind,
                    int(q[0, 0]), int(q[0, 1]), int(q[0, 2]), width, _b64(d.astype(np.int8 if width == 1 else "<i2")), runs,
                    tr["w"]])
    return {
        "demand": demand, "steps": n, "dt": cfg.dt, "lamps": lamps, "wreck_time": cfg.wreck_time,
        "kinds": list(DONE_NAMES), "tracks": out, "marks": marks,
        "queue": _b64(np.asarray(queue, dtype=np.uint8)),
    }


def record_replays(drivers: list[tuple[str, str, object, dict]], workers: int) -> dict:
    """drivers: (name, kind, payload, physics) -> {"q": quantisation, "runs": {name: {demand: replay}}}."""
    jobs = [(kind, payload, physics, d) for _, kind, payload, physics in drivers for d in REPLAY_DEMANDS]
    t0 = time.time()
    with mp.get_context(MP_START).Pool(workers) as pool:
        res = pool.map(_replay_job, jobs, chunksize=1)
    print(f"  replays: {len(jobs)} recorded in {time.time() - t0:.0f} s")
    runs = {}
    for (name, *_), r in zip([dr for dr in drivers for _ in REPLAY_DEMANDS], res):
        runs.setdefault(name, {})[str(r["demand"])] = r
    return {"q": {"pos": Q_POS, "th": Q_TH, "lamp": Q_LAMP}, "seed": REPLAY_SEED, "seconds": REPLAY_SECONDS,
            "demands": REPLAY_DEMANDS, "runs": runs}


def pol_entry(c: dict) -> tuple:
    return (f"pol:{c['hash']}", "policy", Brain(c["policy"]).payload(), c["physics"])


# ------------------------------------------------------------------ logs
def _flat(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _flat(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def _pick(flat: dict, patterns: list[str]):
    for pat in patterns:
        for k, v in flat.items():
            if re.fullmatch(pat, k) and isinstance(v, (int, float)):
                return float(v)
    return None


def read_curve(run: str, offset: int) -> list[dict]:
    """Learning curve of one run: x (cumulative generation / iteration), exit, crash,
    offroad (fractions of finished cars) and exit at demand 0.15 where evaluated.
    Robust to the field names: ES logs use centre.exit_ok etc."""
    if not from_runs(run):  # snapshot: stored with offset 0 (already thinned)
        rows = SOURCE["manifest"]["curves"].get(run, [])
        return [{**r, "x": round(r["x"] + offset, 4)} for r in rows]
    p = RUNS / run / "log.jsonl"
    if not p.exists():
        return []
    rows = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        f = dict(_flat(rec))
        x = _pick(f, [r"gen", r"iter(ation)?", r"update", r"step", r"epoch"])
        x = (x if x is not None else i) + offset
        row = {
            "x": x,
            "exit": _pick(f, [r"centre\.exit_ok", r"(train\.)?exit_ok(_rate)?", r".*\.exit_ok(_rate)?"]),
            "crash": _pick(f, [r"centre\.crash_rate", r"(train\.)?crash(_rate)?", r".*\.crash(_rate)?"]),
            "offroad": _pick(f, [r"centre\.offroad_rate", r"(train\.)?offroad(_rate)?", r".*\.offroad(_rate)?"]),
            "eval15": _pick(f, [r"eval\.d0\.15\.exit_ok", r"eval\..*0\.15.*exit_ok(_rate)?"]),
            "demand": _pick(f, [r"ceiling", r"demand"]),
        }
        if row["exit"] is None and row["eval15"] is None:
            continue
        if _pick(f, [r"train\.finished"]) == 0 and row["eval15"] is None:  # no car finished: no rates
            continue
        rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if v is not None})
    # thin out long logs (keep every eval point)
    if len(rows) > 400:
        step = len(rows) / 400
        keep = {int(i * step) for i in range(400)} | {i for i, r in enumerate(rows) if "eval15" in r}
        rows = [r for i, r in enumerate(rows) if i in keep]
    return rows


# ------------------------------------------------------------------ selection
def resolve(found: dict, spec: str) -> str:
    """Checkpoint id of `spec`: an id (<run>/<name>), its runs/ path, or any .npz / .json file
    whose weights equal a discovered checkpoint (e.g. results/wide_s1ft2_ckpt2650.npz)."""
    cid = next((k for k, c in found.items() if c["file"] == spec or k == spec), None)
    if cid is None and (ROOT / spec).is_file():
        h = policy_hash(load_adapted_policy(ROOT / spec))
        cid = next((k for k, c in found.items() if c["hash"] == h), None)
    if cid is None:
        raise SystemExit(f"{spec}: not a loadable checkpoint in {SOURCE['kind']} (see skip messages)")
    return cid


def run_world(run: str, found: dict) -> str | None:
    c = next((c for c in found.values() if c["run"] == run), None)
    if c is not None:
        return world_of(c["physics"])
    if SOURCE["manifest"]:
        return SOURCE["manifest"]["runs"].get(run, {}).get("world")
    return None


def write_snapshot(found: dict, used: list[str], chain_runs: list[str]):
    """results/site/: the used checkpoints (exact weights) + manifest (chains, worlds, curves)."""
    brains = SNAPSHOT / "brains"
    old = {f.relative_to(brains).as_posix() for f in brains.glob("*/*.npz")} if brains.exists() else set()
    keep = set()
    for cid in used:
        c = found[cid]
        pol = c["policy"]
        rel = f"{c['run']}/{cid.split('/', 1)[1]}.npz"
        keep.add(rel)
        if c.get("snap") and (brains / rel).exists():  # unchanged copy from the snapshot itself
            continue
        binfo, arrays = Brain(pol).snapshot()
        info = {"run": c["run"], "file": c["file"], "method": c["method"], "gen": c["gen"], "physics": c["physics"],
                **binfo}
        path = brains / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, info=json.dumps(info, sort_keys=True), **arrays)
    for rel in old - keep:
        if SOURCE["kind"] == "merge":
            continue  # merge: the snapshot only grows (nothing old is deleted)
        (brains / rel).unlink()
    for d in brains.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    runs = sorted(set(chain_runs) | {found[cid]["run"] for cid in used})
    manifest = {
        "note": "written by site/build.py from runs/ (see its docstring); merged with runs/ or read alone",
        "runs": {r: {"init": list(init_of(r)) if init_of(r) else None, "world": run_world(r, found)} for r in runs},
        "curves": {r: read_curve(r, 0) for r in sorted(set(chain_runs))},
    }
    if SOURCE["kind"] == "merge":  # keep what the snapshot knew about runs not in runs/
        for k in ("runs", "curves"):
            manifest[k] = {**{r: v for r, v in SOURCE["manifest"][k].items() if not from_runs(r)}, **manifest[k]}
    (SNAPSHOT / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    size = sum(f.stat().st_size for f in SNAPSHOT.rglob("*") if f.is_file())
    print(f"  snapshot: {len(keep)} brains + manifest in {SNAPSHOT.relative_to(ROOT)}/ ({size / 1024:.0f} KiB)")


REWARD_KEYS = ("r_exit", "r_wrong_exit", "r_crash", "r_crash_speed", "r_offroad", "r_offroad_speed", "r_timeout",
               "r_progress", "r_time", "r_queue", "crash_cost_speed", "L", "v_max", "spawn_speed", "max_age")


def reward_of(train_cfg: dict) -> dict:
    """The reward rules (and the few world numbers an example trip needs) a policy was trained with."""
    d = SimConfig()
    return {k: train_cfg.get(k, getattr(d, k)) for k in REWARD_KEYS}


def choose(found: dict, main: list[str], quick: dict) -> dict:
    """Best checkpoint per method by quick bench score (exit-weighted throughput at 0.15)."""
    best = {}
    for cid, c in found.items():
        r = quick.get(cid)
        if r is None:
            continue
        score = r["throughput_per_min"] * r["exit_ok_rate"]
        if c["method"] not in best or score > best[c["method"]][0]:
            best[c["method"]] = (score, cid)
    picked = {m: cid for m, (_, cid) in best.items()}
    for spec in main:  # explicit --main overrides
        cid = resolve(found, spec)
        picked[found[cid]["method"]] = cid
    return picked


def timeline(found: dict, best_id: str, n_max: int, keep: set = frozenset()) -> list[dict]:
    """Zeitreise points along the warm-start chain of `best_id` (at most n_max; checkpoints in
    `keep`, e.g. the previous main brain, always stay)."""
    c = found[best_id]
    chain = chain_of(c["run"])
    pts = []
    for run, off in chain:
        for cid, cc in found.items():
            if cc["run"] == run and cc["gen"] is not None:
                pts.append((off + cc["gen"], cid))
    # dedupe equal cumulative generations (current == final), prefer the chosen one
    by_gen = {}
    for g, cid in sorted(pts):
        if g not in by_gen or cid == best_id:
            by_gen[g] = cid
    pts = sorted(by_gen.items())
    best_gen = next((g for g, cid in pts if cid == best_id), None)
    if best_gen is not None:  # nothing after the chosen checkpoint
        pts = [p for p in pts if p[0] <= best_gen]
    # drop near-duplicates (closer than 3% of the whole span to the next point)
    span = max((g for g, _ in pts), default=0) or 1
    kept = []
    for p in pts:  # keep a point only if it is far enough from the previously kept one
        if not kept or p[0] - kept[-1][0] > 0.03 * span:
            kept.append(p)
        elif p == pts[-1] or p[1] == best_id or p[1] in keep:  # the final / chosen / kept checkpoint stays
            kept[-1] = p
    pts = kept
    if len(pts) > n_max:
        idx = set(np.unique(np.linspace(0, len(pts) - 1, n_max).round().astype(int)).tolist())
        for j in (j for j, p in enumerate(pts) if p[1] in keep and j not in idx):
            # replace the nearest evenly spaced point (never the first / last one) by the kept one
            near = min((i for i in idx if 0 < i < len(pts) - 1 and pts[i][1] not in keep), key=lambda i: abs(i - j), default=None)
            if near is not None:
                idx = (idx - {near}) | {j}
        pts = [pts[i] for i in sorted(idx)]
    return [{"gen": g, "id": cid} for g, cid in pts], chain


def random_policy(physics: dict | None = None) -> dict:
    """Random brain with the observation interface of `physics` (default: the current one)."""
    cfg = interface_cfg(physics) if physics else SimConfig()
    sizes = layer_sizes(cfg)
    params = init_params(sizes, 7)
    return {
        "sizes": sizes, "in_scale": default_in_scale(cfg).tolist(), "activation": "tanh", "output": "tanh",
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in unpack(params, sizes)], "cfg": {},
    }


# ------------------------------------------------------------------ pages
LOGO = (
    '<svg viewBox="0 0 24 24" aria-hidden="true"><rect width="24" height="24" rx="5" fill="#3b5640"/>'
    '<path d="M9 0h6v24H9zM0 9h24v6H0z" fill="#46484d"/><rect x="5" y="10.2" width="6" height="3.6" rx="1" fill="#f5a623"/>'
    '<rect x="10.2" y="14" width="3.6" height="6" rx="1" fill="#4287f5"/></svg>'
)


def header_html(current: str) -> str:
    """Header: logo, page links and the DE / EN toggle (js/i18n.js)."""
    items = []
    for href, label in PAGES:
        cur = ' aria-current="page"' if href == current else ""
        items.append(f'<a href="{href}"{cur} data-i18n="nav.{href[:-5]}">{label}</a>')
    nav = '<nav class="nav" aria-label="Seiten" data-i18n-attr="aria-label=nav.aria">' + "".join(items) + "</nav>"
    lang = ('<div class="lang" role="group" aria-label="Sprache / Language">'
            '<button type="button" data-lang="de" aria-pressed="true" lang="de">DE</button>'
            '<button type="button" data-lang="en" aria-pressed="false" lang="en">EN</button></div>')
    return f'<header class="top"><a class="brand" href="index.html">{LOGO}Smart Crossing</a>{nav}{lang}</header>'


REPO = "https://github.com/Sipsipl/smartcrossing"
FOOTER = (
    '<footer data-i18n="footer">Smart Crossing – ein Hobbyprojekt. Alles läuft direkt in deinem Browser, '
    'es werden keine Daten gesammelt. Stand: {built}. '
    f'<a href="{REPO}">Code auf GitHub</a></footer>'
)


def sim_config_js() -> str:
    """js/sim-config.js: the numbers sim.js takes from sim.py (loaded before sim.js)."""
    return ("// Generated by site/build.py from smartcrossing/sim.py (js_constants) - do not edit.\n"
            "globalThis.SC_SIM_CONFIG = " + json.dumps(js_constants(), separators=(",", ":")) + ";\n")


def copy_src(version: str, built: str):
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT)
    (OUT / "js" / "sim-config.js").write_text(sim_config_js(), encoding="utf-8", newline="\n")
    for html in OUT.glob("*.html"):
        t = html.read_text(encoding="utf-8")
        t = t.replace("<!-- HEADER -->", header_html(html.name)).replace("<!-- FOOTER -->", FOOTER.format(built=built))
        t = t.replace("{{VERSION}}", version)
        if re.search(r'\s(src|href)="(https?:)?//', t.replace(f'href="{REPO}"', "")):
            raise SystemExit(f"{html.name}: external request found; the site must be self-contained")
        html.write_text(t, encoding="utf-8", newline="\n")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", action="append", default=None, metavar="PATH",
                    help=f"the site's main brain (first one; drives start page, sensors, brain, Vergleich) and/or the best "
                         f"of its method (runs/<run>/x.npz); repeatable; default {DEFAULT_MAIN}")
    ap.add_argument("--history", default=DEFAULT_HISTORY, metavar="PATH",
                    help="checkpoint whose warm-start chain is shown before the main chain in the learning curve and "
                         "Zeitreise (older crossing / earlier line); '' = none")
    ap.add_argument("--bench", action="store_true", help="ignore the bench cache and re-run all benchmarks")
    ap.add_argument("--timeline", type=int, default=6, help="max checkpoints per Zeitreise")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--world", choices=["narrow", "wide", "any"], default="any",
                    help="only use checkpoints trained on this crossing (narrow = original 7 m road, wide = 14 m road)")
    ap.add_argument("--mix", default=DEFAULT_MIX, metavar="PATH", help="second brain for mixing on the start page ('' = none)")
    ap.add_argument("--previous", default=DEFAULT_PREVIOUS, metavar="PATH",
                    help="the previous main brain, shown next to the main brain on Vergleich ('' = none)")
    ap.add_argument("--source", choices=["auto", "runs", "snapshot", "merge"], default="auto",
                    help="where the checkpoints / logs come from: runs/ only, the committed snapshot results/site/ only, "
                         "or merge (snapshot + training runs in runs/); runs and merge also refresh the snapshot; "
                         "auto = merge if both exist, else whichever exists")
    a = ap.parse_args(argv)
    if a.main is None:
        a.main = list(DEFAULT_MAIN)
    return a


def setup_source(source: str) -> str:
    """Resolve --source (auto -> merge / snapshot / runs) and fill SOURCE; returns the kind."""
    kind = source
    train_runs = {p.name for p in RUNS.iterdir() if p.is_dir() and method_of(p.name)} if RUNS.exists() else set()
    if kind == "auto":
        has_snap = (SNAPSHOT / "manifest.json").exists()
        kind = ("merge" if train_runs else "snapshot") if has_snap else "runs"
    SOURCE["kind"], SOURCE["runs"] = kind, train_runs
    if kind in ("snapshot", "merge"):
        if not (SNAPSHOT / "manifest.json").exists():
            raise SystemExit(f"no runs/ and no snapshot in {SNAPSHOT.relative_to(ROOT)}/")
        SOURCE["manifest"] = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    print(f"source: {kind}" + (f" (+ runs/: {', '.join(sorted(train_runs))})" if kind == "merge" else ""))
    return kind


def collect(found: dict, a) -> dict:
    """Collect phase: pick the brains (best per method, main / history / previous / mix), the Zeitreise
    timelines, warm-start chains, learning curves and side notes; returns the selection incl. the
    policies the site shows and the bench entries they need."""
    def cid_of(spec):
        return resolve(found, spec)

    # methods without an explicit --main: quick bench of all their checkpoints at demand 0.15
    fixed = {found[cid_of(s)]["method"] for s in a.main}
    quick_entries = [(*pol_entry(c), [0.15]) for c in found.values() if c["method"] not in fixed]
    qb = run_bench(quick_entries, a.bench, a.workers)
    quick = {cid: qb[pol_entry(c)[0]]["0.15"] for cid, c in found.items() if c["method"] not in fixed}
    picked = choose(found, a.main, quick)
    main_id = cid_of(a.main[0]) if a.main else max(picked.values(), key=lambda k: quick[k]["throughput_per_min"] * quick[k]["exit_ok_rate"])
    hist_id = cid_of(a.history) if a.history else None
    prev_id = cid_of(a.previous) if a.previous else None
    print(f"  best per method: {picked}, main: {main_id}, history: {hist_id}")

    policies, bench_entries = {}, []

    def use(cid, demands):
        c = found[cid]
        if cid not in policies:
            policies[cid] = {
                "id": cid, "run": c["run"], "run_label": run_info(c["run"])["label"], "run_label_en": run_info(c["run"])["label_en"], "file": c["file"], "method": c["method"], "gen": c["gen"],
                "physics": c["physics"], "adapted": c["adapted"], "policy": Brain(c["policy"]).rounded(4),
                "bench": {},
            }
        bench_entries.append((*pol_entry(c), demands))

    timelines, curves, chains = {}, {}, {}
    for method, cid in picked.items():
        use(cid, DEMANDS)
        parts = [cid]
        if hist_id and found[hist_id]["method"] == method and hist_id != cid:
            use(hist_id, DEMANDS)
            parts.insert(0, hist_id)
        pts_all, chain_all, shift = [], [], 0
        for i, pid in enumerate(parts):
            # history first (fewer Zeitreise points), then the main chain, shifted to follow it
            pts, chain = timeline(found, pid, max(2, a.timeline // 2) if i < len(parts) - 1 else a.timeline, keep={prev_id})
            crv = {r: read_curve(r, off) for r, off in chain}
            if i > 0 and any(crv.values()):
                lo = min(row["x"] for rows in crv.values() for row in rows)
                if lo < shift:  # iteration counts restart: move the whole chain behind the history
                    chain = [(r, int(off + shift - lo)) for r, off in chain]
                    crv = {r: read_curve(r, off) for r, off in chain}
                    pts = [{**p, "gen": int(p["gen"] + shift - lo)} for p in pts]
            pts_all += pts
            chain_all += chain
            curves.setdefault(method, {}).update(crv)
            shift = 1 + max([row["x"] for rows in crv.values() for row in rows] + [p["gen"] for p in pts_all] + [0])
        chains[method] = [{"run": r, "offset": off, "world": run_world(r, found), **run_info(r)} for r, off in chain_all]
        for p in pts_all:
            use(p["id"], TIMELINE_DEMANDS)
        timelines[method] = pts_all
    # random brain as "generation 0" of the ES timeline
    if "es" in timelines:
        phys = found[picked["es"]]["physics"]
        rp = random_policy(phys)
        rid = "zufall"
        policies[rid] = {"id": rid, "run": None, "file": None, "method": "es", "gen": 0, "physics": phys,
                         "adapted": None, "policy": Brain(rp).rounded(4), "bench": {}}
        bench_entries.append(("pol:random7", "policy", Brain(rp).payload(), phys, TIMELINE_DEMANDS))
        if not timelines["es"] or timelines["es"][0]["gen"] > 0:
            timelines["es"].insert(0, {"gen": 0, "id": rid})
    # other (dead-end) runs: their best checkpoint as a side note
    others = []
    in_chain = {r["run"] for ch in chains.values() for r in ch}
    for run in sorted({c["run"] for c in found.values()} - in_chain):
        cands = [cid for cid, c in found.items() if c["run"] == run]
        cid = max(cands, key=lambda k: (found[k]["gen"] or 0))
        use(cid, TIMELINE_DEMANDS)
        others.append({"run": run, "id": cid, "method": found[cid]["method"], **run_info(run)})

    mix_id = cid_of(a.mix) if a.mix else None
    if mix_id:
        use(mix_id, DEMANDS)
    if prev_id:
        use(prev_id, DEMANDS)
    return {"picked": picked, "main_id": main_id, "hist_id": hist_id, "prev_id": prev_id, "mix_id": mix_id,
            "policies": policies, "bench_entries": bench_entries, "timelines": timelines, "curves": curves,
            "chains": chains, "others": others}


def bench(found: dict, sel: dict, cfg: SimConfig, a) -> dict:
    """Bench phase: the rule-based drivers in the main brain's world (and in each other brain's world on
    "Vergleich", e.g. the old narrow crossing, so every brain is compared on its own crossing) plus all
    selected policies; fills each policy's "bench". Returns the baseline results and worlds."""
    def world_phys(cid):
        return {**physics_of(cfg.__dict__), **{k: v for k, v in found[cid]["physics"].items() if k in GEOMETRY_KEYS}}

    base_worlds = {}  # world name -> physics; the main brain's world first
    for cid in (sel["main_id"], sel["hist_id"], sel["picked"].get("es")):
        if cid:
            base_worlds.setdefault(world_of(found[cid]["physics"]), world_phys(cid))
    if not base_worlds:
        base_worlds[world_of(physics_of(cfg.__dict__))] = physics_of(cfg.__dict__)
    main_world = next(iter(base_worlds))
    base_key = lambda drv, w: f"base:{drv}" if w == main_world else f"base:{drv}:{w}"  # noqa: E731
    entries = list(sel["bench_entries"])
    for w, phys in base_worlds.items():
        for drv in ("light", "reservation"):
            entries.append((base_key(drv, w), "baseline", drv, phys, DEMANDS))
    res = run_bench(entries, a.bench, a.workers)
    for cid, p in sel["policies"].items():
        key = f"pol:{found[cid]['hash']}" if cid in found else "pol:random7"
        p["bench"] = res[key]
    by_world = {w: {drv: res[base_key(drv, w)] for drv in ("light", "reservation")} for w in base_worlds}
    return {"baselines_by_world": by_world, "main_world": main_world, "base_phys": base_worlds[main_world]}


FUNDAMENTAL = ROOT / "results" / "deadlock_fundamental.json"
FUNDAMENTAL_DRIVERS = ("brain", "light", "reservation")  # curves of the fundamental diagram on the site
START_DEMAND_MAX = 0.5  # max of the start page demand slider
FUNDAMENTAL_FIELDS = ("demand", "density", "throughput", "density_veh_per_km", "demand_per_min", "deadlock_time_frac")


def fundamental_data():
    """Fundamental diagram (throughput vs cars on the map per driver and demand) from the deadlock study's
    results/deadlock_fundamental.json, reduced to what the site plots; None if the file is missing."""
    if not FUNDAMENTAL.exists():
        return None
    raw = json.loads(FUNDAMENTAL.read_text(encoding="utf-8"))
    drivers = {k: [{f: p[f] for f in FUNDAMENTAL_FIELDS} for p in raw["drivers"][k]["points"]]
               for k in FUNDAMENTAL_DRIVERS if k in raw["drivers"]}
    out = {"drivers": drivers, "cloud": cloud_data()}
    if "brain" in drivers:  # start value of the traffic slider = the brain's capacity: its maximum mean throughput
        cap = max(p["throughput"] for p in drivers["brain"])
        out["max_throughput"] = round(cap, 1)
        out["start_demand"] = min(round(cap / 240, 2), START_DEMAND_MAX)  # cars/min -> arrivals /s/arm, slider step 0.01
    return out


CLOUD = ROOT / "results" / "fundamental_cloud.json"


def cloud_data():
    """Scatter cloud behind the fundamental diagram (single measured 60-s windows per driver, from
    results/fundamental_cloud.json: [density, throughput] pairs); {} if the file is missing."""
    if not CLOUD.exists():
        return {}
    raw = json.loads(CLOUD.read_text(encoding="utf-8"))
    return {k: [[p[0], p[1]] for p in v["points"]] for k, v in raw["drivers"].items() if k in FUNDAMENTAL_DRIVERS}


def write_site(found: dict, sel: dict, bres: dict, cfg: SimConfig, kind: str, a):
    """Write phase: site data (observation layouts, policies, curves, baselines, ...), the side-by-side
    replays, the snapshot (runs / merge) and the public/ folder."""
    main_id, policies = sel["main_id"], sel["policies"]
    # observation labels per interface (lamp count, neighbour layout) of the brains on the site; "layout"
    # is the main brain's (the pages pick the one of their sim: SCX.layoutFor)
    ifaces = {interface_key(cfg): cfg}
    for p in policies.values():
        c = SimConfig(**{k: p["physics"][k] for k in INTERFACE_KEYS if k in p["physics"]})
        ifaces.setdefault(interface_key(c), c)
    layouts = {k: obs_layout(c) for k, c in ifaces.items()}
    main_iface = interface_key(found[main_id]["physics"]) if main_id else interface_key(cfg)
    data = {
        "built": date.today().isoformat(),
        "layout": layouts[main_iface],
        "layouts": layouts,
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "policies": policies,
        "best": sel["picked"],
        "main": found[main_id]["method"] if main_id else None,  # method of the main brain (shown first on the site)
        "main_id": main_id,
        "history": sel["hist_id"],
        "mix": sel["mix_id"],
        "previous": sel["prev_id"] if sel["prev_id"] != main_id else None,  # the main brain before the current one
        "timelines": sel["timelines"],
        "chains": sel["chains"],
        "curves": sel["curves"],
        "others": sel["others"],
        "baselines": bres["baselines_by_world"][bres["main_world"]],  # in the main brain's world
        "baselines_by_world": bres["baselines_by_world"],  # world ("wide" / "narrow") -> {light, reservation}
        "main_world": bres["main_world"],
        # reward rules the main brain was trained with (from its checkpoint's training cfg; drawn on /lernen)
        "reward": reward_of(found[main_id]["policy"].get("cfg") or {}) if main_id else None,
        "demands": DEMANDS,
        "fundamental": fundamental_data(),
        "bench_setup": {"envs": BENCH_ENVS, "seconds": BENCH_SECONDS, "seeds": len(BENCH_SEEDS)},
    }
    # side-by-side replays (same traffic) of the two rule-based drivers and the best policy
    base_phys = bres["base_phys"]
    replays = None
    if main_id:
        _, _, payload, phys = pol_entry(found[main_id])
        replays = record_replays([
            ("light", "baseline", "light", base_phys),
            ("reservation", "baseline", "reservation", base_phys),
            ("policy", "policy", payload, phys),
        ], a.workers)
        replays["policy_id"] = main_id
        replays["world"] = {k: base_phys[k] for k in ("L", "W", "Rc") if k in base_phys}
    data["replay_world"] = world_of(base_phys)
    if kind in ("runs", "merge"):
        write_snapshot(found, [cid for cid in policies if cid in found], [r["run"] for ch in sel["chains"].values() for r in ch])
    src_hash = hashlib.sha1(b"".join(f.read_bytes() for f in sorted(SRC.rglob("*")) if f.is_file())
                            + sim_config_js().encode()).hexdigest()
    version = hashlib.sha1((json.dumps([data, replays], sort_keys=True) + src_hash).encode()).hexdigest()[:8]
    copy_src(version, data["built"])
    (OUT / "data").mkdir(exist_ok=True)
    js = "// Generated by site/build.py - do not edit.\nwindow.SC_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    (OUT / "data" / "site-data.js").write_text(js, encoding="utf-8", newline="\n")
    rjs = "// Generated by site/build.py - do not edit.\nwindow.SC_REPLAYS = " + json.dumps(replays, separators=(",", ":")) + ";\n"
    (OUT / "data" / "replays.js").write_text(rjs, encoding="utf-8", newline="\n")
    print(f"  replays.js: {len(rjs) / 1024:.0f} KiB")
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"wrote {OUT.relative_to(ROOT)}/ ({size / 1024:.0f} KiB, data {len(js) / 1024:.0f} KiB, version {version})")
    for m, pts in sel["timelines"].items():
        print(f"  Zeitreise {m}: " + ", ".join(f"{p['id']}@{p['gen']}" for p in pts))


def main(argv=None):
    a = parse_args(argv)
    kind = setup_source(a.source)
    cfg = SimConfig()
    obs_layout(cfg)  # fails loudly if the sim's observation layout changed
    print("discovering checkpoints ...")
    found = discover(a.world)
    FOUND.update(found)
    print(f"  {len(found)} loadable checkpoints")
    sel = collect(found, a)
    bres = bench(found, sel, cfg, a)
    write_site(found, sel, bres, cfg, kind, a)


if __name__ == "__main__":
    main()
