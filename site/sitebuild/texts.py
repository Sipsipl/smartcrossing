"""Texts of the runs on the website ("Wie lernen sie?" / learning page), German and English side by side.

RUN_INFO: run id -> {"short": chart marker, "de": (label, note), "en": (label, note)}; shown instead of the raw
run ids (unmapped runs fall back to the id). STEP_INFO: run id -> {"de": ..., "en": ...}, one line per phase of the
main training chain next to the numbered markers of the learning curve ("what changed -> what happened").
Plain words, no run ids. ``{safe}`` / ``{stop}`` / ``{block}`` are filled with the stop-gate percentages of the main brain's
checkpoint at build time. One entry per run holds both languages, so a missing English text is impossible
(tests/test_site_i18n.py).
"""

RUN_INFO = {
    "es1": {
        "short": "E1",
        "de": ("Evolution, Versuch 1: Stehenbleiben",
               "Irrweg: Die Autos lernten, einfach stehen zu bleiben – Stehen kostete nichts, Unfälle schon."),
        "en": ("Evolution, attempt 1: standing still",
               "Dead end: the cars learned to just stand still – standing cost nothing, crashes did."),
    },
    "es2": {
        "short": "E2",
        "de": ("Evolution, Versuch 2: Stehen kostet jetzt",
               "Erster Erfolg: Autos fahren zum Ziel, aber Rechtsabbiegen klappt nie (alle landen im "
               "Bordstein)."),
        "en": ("Evolution, attempt 2: standing now costs",
               "First success: cars drive to their goal, but turning right never works (they all end up on the "
               "curb)."),
    },
    "es3": {
        "short": "E3",
        "de": ("Evolution, Versuch 3: mehr ausprobieren",
               "Größere Zufallsänderungen: kaum Fortschritt, die Rechtskurve bleibt das Problem."),
        "en": ("Evolution, attempt 3: trying more",
               "Bigger random changes: hardly any progress, the right turn stays the problem."),
    },
    "es4": {
        "short": "E4",
        "de": ("Evolution, Versuch 4: Einfahrt blockieren",
               "Irrweg: Die Autos blockierten die Einfahrt – wer gar nicht erst reinfährt, kann nicht "
               "verunglücken."),
        "en": ("Evolution, attempt 4: blocking the entrance",
               "Dead end: the cars blocked the entrance – a car that never enters can't crash."),
    },
    "es5": {
        "short": "E5",
        "de": ("Evolution, Versuch 5: Weg entlang der Straße messen",
               "Fortschritt wird entlang der Straße gemessen, wartende Autos kosten Punkte: jetzt klappen alle "
               "drei Richtungen meistens."),
        "en": ("Evolution, attempt 5: measuring progress along the road",
               "Progress is measured along the road and waiting cars lose points: now all three directions "
               "mostly work."),
    },
    "ppo1a": {
        "short": "P1",
        "de": ("PPO Versuch 1: Schleichen",
               "Irrweg: Die Autos kriechen nur noch und kommen kaum an."),
        "en": ("PPO attempt 1: creeping",
               "Dead end: the cars only crawl along and hardly ever arrive."),
    },
    "ppo1b": {
        "short": "P1b",
        "de": ("PPO Versuch 1b: schneller lernen",
               "Irrweg: Größere Lernschritte – die Autos bleiben trotzdem stehen."),
        "en": ("PPO attempt 1b: learning faster",
               "Dead end: bigger learning steps – the cars still stand still."),
    },
    "ppo2": {
        "short": "P2",
        "de": ("PPO Versuch 2: falsche Ausfahrt",
               "Irrweg: Über die Hälfte nimmt einfach die bequemste Ausfahrt, weil die falsche Ausfahrt "
               "billiger war als Abbiegen."),
        "en": ("PPO attempt 2: wrong exit",
               "Dead end: over half simply take the most convenient exit, because the wrong exit was cheaper "
               "than turning."),
    },
    "ppo3": {
        "short": "P3",
        "de": ("PPO Versuch 3: härtere Strafen",
               "Irrweg: Mehr Strafe für Unfälle – die Autos nehmen weiter die falsche Ausfahrt."),
        "en": ("PPO attempt 3: tougher penalties",
               "Dead end: bigger penalty for crashes – the cars still take the wrong exit."),
    },
    "ppo4": {
        "short": "P4",
        "de": ("PPO Versuch 4: mehr Autos pro Runde",
               "Irrweg: Mehr Simulationen pro Runde – dasselbe Problem mit der falschen Ausfahrt."),
        "en": ("PPO attempt 4: more cars per round",
               "Dead end: more simulations per round – the same wrong-exit problem."),
    },
    "ppo5": {
        "short": "P5",
        "de": ("PPO Versuch 5: falsche Ausfahrt teurer",
               "Falsche Ausfahrt kostet jetzt mehr – dafür landen die meisten im Bordstein."),
        "en": ("PPO attempt 5: wrong exit costs more",
               "The wrong exit costs more now – but most cars end up on the curb instead."),
    },
    "ppo6": {
        "short": "P6",
        "de": ("Mehr Punkte für Fortschritt",
               "Durchbruch: doppelt so viele Punkte für jeden Meter Richtung Ausfahrt – die meisten kommen an."),
        "en": ("More points for progress",
               "Breakthrough: twice the points for every metre towards the exit – most cars arrive."),
    },
    "ppo7": {
        "short": "P7",
        "de": ("PPO Versuch 7: neu von vorn",
               "Gleiche Einstellungen wie „Ruhigeres Ausprobieren“, aber von Null an gelernt; endet am selben "
               "Punkt."),
        "en": ("PPO attempt 7: starting fresh",
               "Same settings as “Calmer experimenting”, but learned from scratch; ends at the same point."),
    },
    "ppo8": {
        "short": "P8",
        "de": ("Ruhigeres Ausprobieren",
               "Weniger zufälliges Zappeln beim Üben. Gut bei wenig Verkehr, aber geübt wurde nur mit fast "
               "leeren Straßen."),
        "en": ("Calmer experimenting",
               "Less random twitching while practising. Good in light traffic, but it only practised on nearly "
               "empty roads."),
    },
    "ppo9": {
        "short": "P9",
        "de": ("Dichter Verkehr beim Üben",
               "Üben mit viel mehr Autos gleichzeitig: Begegnungen werden häufiger, die Unfälle gehen deutlich "
               "zurück."),
        "en": ("Dense traffic while practising",
               "Practising with many more cars at once: encounters get more frequent, crashes drop clearly."),
    },
    "ppo10": {
        "short": "P10",
        "de": ("PPO Versuch 10: Unfälle doppelt so teuer",
               "Kein Gewinn, früh abgebrochen."),
        "en": ("PPO attempt 10: crashes twice as costly",
               "No gain, stopped early."),
    },
    "ppo11": {
        "short": "P11",
        "de": ("PPO Versuch 11: weiter gucken (60 m)",
               "Die Autos sehen andere aus 60 statt 30 m – musste sich erst umgewöhnen, kein klarer Gewinn."),
        "en": ("PPO attempt 11: looking further (60 m)",
               "The cars see others from 60 instead of 30 m – needed time to adjust, no clear gain."),
    },
    "ppo12": {
        "short": "P12",
        "de": ("PPO Versuch 12: Unfälle 2,5-mal so teuer",
               "Etwas mehr Bremsen, aber nicht weniger Unfälle."),
        "en": ("PPO attempt 12: crashes 2.5 times as costly",
               "Slightly more braking, but no fewer crashes."),
    },
    "ppo13": {
        "short": "P13",
        "de": ("Feinschliff mit kleineren Lernschritten",
               "Kleinere Lernschritte: langsam, aber stetig weniger Unfälle."),
        "en": ("Fine-tuning with smaller learning steps",
               "Smaller learning steps: slowly but steadily fewer crashes."),
    },
    "ppo14": {
        "short": "P14",
        "de": ("PPO Versuch 14: 60 m Sicht, neu von vorn",
               "Irrweg: Die weit entfernten Autos überfluten das Gehirn mit großen Zahlen – alle landen im "
               "Bordstein."),
        "en": ("PPO attempt 14: 60 m view, from scratch",
               "Dead end: the far-away cars flood the brain with big numbers – everyone ends up on the curb."),
    },
    "ppo15": {
        "short": "P15",
        "de": ("PPO Versuch 15: 60 m Sicht, richtig skaliert",
               "Lernt genauso gut wie mit 30 m – die Sichtweite war nicht das Problem."),
        "en": ("PPO attempt 15: 60 m view, scaled properly",
               "Learns just as well as with 30 m – the viewing distance was not the problem."),
    },
    "ppo16": {
        "short": "P16",
        "de": ("PPO Versuch 16: Wunschtempo statt Gaspedal",
               "Das Gehirn wählt eine Zielgeschwindigkeit statt Gas/Bremse. Anfangs vorsichtiger, am Ende "
               "gleich gut."),
        "en": ("PPO attempt 16: target speed instead of gas pedal",
               "The brain picks a target speed instead of gas/brake. More careful at first, equally good in "
               "the end."),
    },
    "ppo17": {
        "short": "P17",
        "de": ("Noch feiner, mehr Weitblick",
               "Noch kleinere Lernschritte und längerer Blick auf Folgen: der beste Stand auf der alten, "
               "schmalen Kreuzung."),
        "en": ("Even finer, more foresight",
               "Even smaller learning steps and a longer look at the consequences: the best result on the old, "
               "narrow crossing."),
    },
    "wide_s1": {
        "short": "B1",
        "de": ("Breite Kreuzung: neu gelernt",
               "Neue Kreuzung: Straßen doppelt so breit, Reifen rutschen bei zu schnellen Kurven. Von Null an "
               "gelernt."),
        "en": ("Wide crossing: learned anew",
               "New crossing: roads twice as wide, tyres slip in corners taken too fast. Learned from scratch."),
    },
    "wide_s1r": {
        "short": "B2",
        "de": ("Breite Kreuzung: mit Rückwärtsgang",
               "Die Autos dürfen jetzt auch rückwärts fahren – sie tun es aber fast nie."),
        "en": ("Wide crossing: with reverse gear",
               "The cars may now also drive backwards – but they hardly ever do."),
    },
    "wide_s1ft": {
        "short": "B3",
        "de": ("Breite Kreuzung: Feinschliff",
               "Kleinere Lernschritte: weniger Unfälle."),
        "en": ("Wide crossing: fine-tuning",
               "Smaller learning steps: fewer crashes."),
    },
    "wide_s1ft2": {
        "short": "B4",
        "de": ("Breite Kreuzung: länger feinschleifen",
               "Noch eine Stunde Feinschliff: nur noch etwas besser – das Hauptgehirn vor Runde 1."),
        "en": ("Wide crossing: fine-tuning for longer",
               "Another hour of fine-tuning: only a little better – the main brain before round 1."),
    },
    "wide_s0": {
        "short": "B1'",
        "de": ("Breite Kreuzung, zweiter Lauf",
               "Gleiches Rezept, anderer Zufallsstart (noch ohne Rückwärtsgang)."),
        "en": ("Wide crossing, second run",
               "Same recipe, different random start (still without reverse gear)."),
    },
    "wide_s0r": {
        "short": "B2'",
        "de": ("Zweiter Lauf: mit Rückwärtsgang",
               "Anderer Zufallsstart, mit Rückwärtsgang weitergelernt."),
        "en": ("Second run: with reverse gear",
               "Different random start, kept learning with reverse gear."),
    },
    "wide_s0ft": {
        "short": "B3'",
        "de": ("Zweiter Lauf: Feinschliff",
               "Anderer Zufallsstart, genauso fein geschliffen: fährt auch rechts, ein paar Unfälle mehr."),
        "en": ("Second run: fine-tuning",
               "Different random start, fine-tuned the same way: also drives on the right, a few more crashes."),
    },
    "wide_noman_s0": {
        "short": "B0",
        "de": ("Ohne Abbiege-Wunsch",
               "Die Autos erfahren nicht, ob sie links, rechts oder geradeaus wollen – nur, wo das Ziel liegt. "
               "Lernt langsamer."),
        "en": ("Without turn wish",
               "The cars aren't told whether they want to go left, right or straight – only where the goal is. "
               "Learns more slowly."),
    },
    "r1_x3": {
        "short": "R1",
        "de": ("Runde 1: neue Regeln, altes Gehirn lernt weiter",
               "Neue Regeln: Wracks mit Warnblinker, Bremslicht, eine Lampe, Einfahrt nur rechts. Das alte "
               "Gehirn lernt weiter, Unfälle kosten jetzt dreimal so viel."),
        "en": ("Round 1: new rules, old brain keeps learning",
               "New rules: wrecks with hazard lights, brake light, one lamp, entering only from the right. The "
               "old brain keeps learning, crashes now cost three times as much."),
    },
    "r1_main_x3": {
        "short": "R1+",
        "de": ("Runde 1: länger weitergelernt",
               "Rund 40 % weniger Unfälle als das Gehirn davor, bei dichtem Verkehr aber noch viele – das "
               "Hauptgehirn vor Runde 2."),
        "en": ("Round 1: learned for longer",
               "About 40% fewer crashes than the brain before, but still many in dense traffic – the main "
               "brain before round 2."),
    },
    "r1_x1": {
        "short": "R1a",
        "de": ("Runde 1, Test: Unfälle so teuer wie bisher",
               "Kurzer Test: ähnlich gut wie mit dreifacher Strafe – der Unterschied ist Zufall."),
        "en": ("Round 1, test: crashes as costly as before",
               "Short test: about as good as with triple penalty – the difference is chance."),
    },
    "r1_x10": {
        "short": "R1b",
        "de": ("Runde 1, Test: Unfälle zehnmal so teuer",
               "Kurzer Test: zu viel Strafe – weniger Autos kommen durch, und Unfälle gibt es nicht weniger."),
        "en": ("Round 1, test: crashes ten times as costly",
               "Short test: too much penalty – fewer cars get through, and crashes aren't any rarer."),
    },
    "r2_B_s1": {
        "short": "R2",
        "de": ("Runde 2: mit Hindernissen geübt",
               "Im Training standen Wracks und parkende Autos im Weg: Es bremst viel öfter dahinter ab und "
               "reiht sich ein, statt hineinzufahren – das Hauptgehirn vor Runde 4."),
        "en": ("Round 2: practised with obstacles",
               "Wrecks and parked cars stood in the way during training: it brakes behind them far more often "
               "and queues up instead of driving into them – the main brain before round 4."),
    },
    "r2_A_s1": {
        "short": "R2a",
        "de": ("Runde 2, Test: nur mehr Ausprobieren",
               "Ohne Hindernisse im Training: schon deutlich weniger Unfälle als vorher, aber längst nicht so "
               "sicher vor Hindernissen."),
        "en": ("Round 2, test: just more experimenting",
               "Without obstacles in training: already clearly fewer crashes than before, but far less safe "
               "around obstacles."),
    },
    "r2_A_s2": {
        "short": "R2a'",
        "de": ("Runde 2, Test: nur mehr Ausprobieren, zweiter Lauf",
               "Anderer Zufallsstart, fast dasselbe Ergebnis."),
        "en": ("Round 2, test: just more experimenting, second run",
               "Different random start, almost the same result."),
    },
    "r2_C_s1": {
        "short": "R2c",
        "de": ("Runde 2, Test: Unfälle je nach eigenem Tempo bestraft",
               "Mit Hindernissen und tempoabhängiger Strafe: etwas schlechter als ohne die Strafe (bei einem "
               "Lauf nicht sicher)."),
        "en": ("Round 2, test: crashes penalised by own speed",
               "With obstacles and a speed-dependent penalty: slightly worse than without the penalty (not "
               "certain from one run)."),
    },
    "r2_B_s2": {
        "short": "R2'",
        "de": ("Runde 2: mit Hindernissen, zweiter Lauf",
               "Gleiches Rezept, anderer Zufallsstart: deutlich mehr Unfälle – der erste Lauf hatte auch Glück."),
        "en": ("Round 2: with obstacles, second run",
               "Same recipe, different random start: clearly more crashes – the first run was also lucky."),
    },
    "r2_C_s2": {
        "short": "R2c'",
        "de": ("Runde 2, Test: tempoabhängige Strafe, zweiter Lauf",
               "Bestätigt: mit tempoabhängiger Strafe mehr Unfälle bei dichtem Verkehr."),
        "en": ("Round 2, test: speed-dependent penalty, second run",
               "Confirmed: with the speed-dependent penalty there are more crashes in dense traffic."),
    },
    "r3_B_long": {
        "short": "R3",
        "de": ("Runde 3: Hindernis-Gehirn länger trainiert",
               "Weitergelernt: kaum weniger Unfälle – mehr vom Gleichen half nicht."),
        "en": ("Round 3: obstacle brain trained for longer",
               "Kept learning: hardly any fewer crashes – more of the same didn't help."),
    },
    "r4_S_s1": {
        "short": "R4",
        "de": ("Runde 4: ein Mini-Netz für jeden Nachbarn + Aufmerksamkeit",
               "Schaut auf bis zu 8 Nachbarn, jeder geht durch dasselbe kleine Netz, dann entscheidet eine "
               "„Aufmerksamkeit“, wer gerade zählt. Erst hat es das vorige Gehirn nachgeahmt, dann 12 Minuten "
               "weitergelernt – das Hauptgehirn bis Runde 5."),
        "en": ("Round 4: a mini-network for every neighbour + attention",
               "Looks at up to 8 neighbours, each goes through the same small network, then an “attention” "
               "decides who matters right now. It first imitated the previous brain, then kept learning for 12 "
               "minutes – the main brain up to round 5."),
    },
    "r4_S_s2": {
        "short": "R4'",
        "de": ("Runde 4: Mini-Netz + Aufmerksamkeit, zweiter Lauf",
               "Anderer Zufallsstart, ähnlich gut – bei sehr dichtem Verkehr bleiben aber mehr Autos mitten "
               "auf der Kreuzung stecken."),
        "en": ("Round 4: mini-network + attention, second run",
               "Different random start, similarly good – but in very dense traffic more cars get stuck in the "
               "middle of the crossing."),
    },
    "r4_M_s1": {
        "short": "R4m",
        "de": ("Runde 4, Vergleich: bisheriges Netz weitertrainiert",
               "Gleich lange (12 Minuten) mit dem bisherigen Netz (64 × 64) weitergelernt: nicht besser als "
               "vorher, bei dichtem Verkehr eher mehr Unfälle."),
        "en": ("Round 4, comparison: previous network trained further",
               "Trained further for the same time (12 minutes) with the previous network (64 × 64): not better "
               "than before, rather more crashes in dense traffic."),
    },
    "r4_M_s2": {
        "short": "R4m'",
        "de": ("Runde 4, Vergleich: bisheriges Netz, zweiter Lauf",
               "Anderer Zufallsstart, ähnlich."),
        "en": ("Round 4, comparison: previous network, second run",
               "Different random start, similar."),
    },
    "r4_W_s1": {
        "short": "R4w",
        "de": ("Runde 4, Vergleich: doppelt so breites Netz",
               "Mehr Neuronen (128 × 128) helfen auch – aber nicht so viel wie das Mini-Netz pro Nachbar."),
        "en": ("Round 4, comparison: network twice as wide",
               "More neurons (128 × 128) help too – but not as much as the mini-network per neighbour."),
    },
    "r4_W_s2": {
        "short": "R4w'",
        "de": ("Runde 4, Vergleich: breites Netz, zweiter Lauf",
               "Anderer Zufallsstart, ähnlich."),
        "en": ("Round 4, comparison: wide network, second run",
               "Different random start, similar."),
    },
    "r5_K_s1": {
        "short": "R5",
        "de": ("Runde 5: weitergelernt (Kontrolle)",
               "Das Hauptgehirn bis Runde 6: das Gehirn aus Runde 4 hat unverändert weitertrainiert. Gewählt ist "
               "ein Zwischenstand (Schritt 475) mit wenigen Unfällen. Vor einem stehenden Hindernis hielt es "
               "nur selten an – im weiteren Training ließ das Anhalten nach."),
        "en": ("Round 5: trained on (control)",
               "The main brain up to round 6: the round 4 brain trained on, unchanged. An intermediate step "
               "(step 475) is chosen, with few crashes. In front of a standing obstacle it rarely stopped – "
               "stopping faded with further training."),
    },
    "r5_K_s2": {
        "short": "R5'",
        "de": ("Runde 5: weitergelernt, zweiter Lauf",
               "Anderer Zufallsstart: bei dichtem Verkehr noch mehr Autos durch, dafür bleiben die Unfälle bei "
               "sehr dichtem Verkehr hoch."),
        "en": ("Round 5: trained on, second run",
               "Different random start: even more cars get through in dense traffic, but crashes stay high in "
               "very dense traffic."),
    },
    "r5_D_s1": {
        "short": "R5d",
        "de": ("Runde 5, Test: Training mit dichtem Verkehr",
               "Mehr Verkehr im Training: weniger Unfälle, dafür mehr Autos, die lange festsitzen – und nicht "
               "mehr Durchsatz als vorher."),
        "en": ("Round 5, test: training with dense traffic",
               "More traffic in training: fewer crashes, but more cars stuck for a long time – and no more "
               "throughput than before."),
    },
    "r5_D_s2": {
        "short": "R5d'",
        "de": ("Runde 5, Test: dichter Verkehr, zweiter Lauf",
               "Anderer Zufallsstart, ähnlich."),
        "en": ("Round 5, test: dense traffic, second run",
               "Different random start, similar."),
    },
    "r5_DW_s1": {
        "short": "R5w",
        "de": ("Runde 5, Test: dichter Verkehr + Warten kostet",
               "Zusätzlich wird Warten bestraft: weniger Unfälle als vorher, aber weiter viel Stau und kaum "
               "mehr Autos durch."),
        "en": ("Round 5, test: dense traffic + waiting costs",
               "Waiting is penalised on top: fewer crashes than before, but still a lot of jamming and hardly "
               "more cars through."),
    },
    "r5_DW_s2": {
        "short": "R5w'",
        "de": ("Runde 5, Test: dichter Verkehr + Warten kostet, zweiter Lauf",
               "Anderer Zufallsstart, ähnlich."),
        "en": ("Round 5, test: dense traffic + waiting costs, second run",
               "Different random start, similar."),
    },
    "r5_DWX_s1": {
        "short": "R5x",
        "de": ("Runde 5, Test: dichter Verkehr, Warten kostet, Unfälle sechsmal so teuer",
               "Noch weniger Unfälle, dafür Stau statt Crash (nur ein kürzerer Lauf)."),
        "en": ("Round 5, test: dense traffic, waiting costs, crashes six times as costly",
               "Even fewer crashes, but jams instead of crashes (a single, shorter run)."),
    },
    "r6_KO_s1": {
        "short": "R6k",
        "de": ("Runde 6, Test: weitergelernt, mehr Hindernis-Szenen",
               "Gleich viele Lernschritte wie die Läufe mit dichtem Verkehr, aber mit mittlerem Verkehr und "
               "häufiger Hindernissen im Training: mehr Autos durch, das Anhalten vor Hindernissen ließ trotzdem "
               "nach."),
        "en": ("Round 6, test: trained on, more obstacle scenarios",
               "The same number of learning steps as the dense-traffic runs, but with medium traffic and "
               "obstacles more often in training: more cars through, yet stopping in front of obstacles "
               "still faded."),
    },
    "r6_KO_s2": {
        "short": "R6k'",
        "de": ("Runde 6, Test: mehr Hindernis-Szenen, zweiter Lauf",
               "Anderer Zufallsstart: viel Durchsatz, am Ende aber auch wieder mehr Unfälle."),
        "en": ("Round 6, test: more obstacle scenarios, second run",
               "Different random start: high throughput, but more crashes again at the end."),
    },
    "r6_DO_s1": {
        "short": "R6d",
        "de": ("Runde 6, Test: sehr dichter Verkehr + mehr Hindernis-Szenen",
               "Geübt wird mit Verkehr bis 120 Autos pro Minute und häufiger Hindernissen: deutlich mehr "
               "Durchsatz und weniger Unfälle als mit mittlerem Verkehr."),
        "en": ("Round 6, test: very dense traffic + more obstacle scenarios",
               "Practising with traffic of up to 120 cars per minute and obstacles more often: clearly more "
               "throughput and fewer crashes than with medium traffic."),
    },
    "r6_DO_s2": {
        "short": "R6",
        "de": ("Runde 6: sehr dichter Verkehr + mehr Hindernis-Szenen, zweiter Lauf",
               "Das Hauptgehirn bis Runde 7: das Gehirn aus Runde 5 hat mit Verkehr bis 120 Autos pro Minute und "
               "häufiger Hindernissen weitertrainiert. Gewählt ist ein Zwischenstand (Schritt 750), der den "
               "Hindernis-Test besteht. Vor einem stehenden Hindernis fuhr es meist herum, statt anzuhalten."),
        "en": ("Round 6: very dense traffic + more obstacle scenarios, second run",
               "The main brain up to round 7: the round 5 brain trained on with traffic of up to 120 cars per "
               "minute and obstacles more often. An intermediate step (step 750) is chosen, which passes the "
               "obstacle test. In front of a standing obstacle it mostly drove around instead of stopping."),
    },
    "r7_DOB_s1": {
        "short": "R7",
        "de": ("Runde 7: sehr dichter Verkehr + Straßensperren",
               "Das Hauptgehirn bis Runde 8: das Gehirn aus Runde 6 hat weitertrainiert, zusätzlich mit "
               "Straßensperren – drei Hindernisse nebeneinander über die ganze Straßenbreite, an denen nur Anhalten "
               "hilft. Gewählt ist ein Zwischenstand (Schritt 1050). Vor einer Sperre hielt es an; vor einem "
               "einzelnen Hindernis fuhr es meist herum."),
        "en": ("Round 7: very dense traffic + road blockades",
               "The main brain up to round 8: the round 6 brain trained on, additionally with road blockades – "
               "three obstacles side by side across the full width of the road, where only stopping helps. An "
               "intermediate step (step 1050) is chosen. In front of a blockade it stopped; in front of a single "
               "obstacle it mostly drove around."),
    },
    "r7_DOB_s2": {
        "short": "R7'",
        "de": ("Runde 7: Straßensperren, zweiter Lauf",
               "Anderer Zufallsstart: hält ebenfalls vor jeder Sperre an. Bei sehr dichtem Verkehr bleibt der "
               "Durchsatz niedriger und schwankt stärker als beim gewählten Lauf."),
        "en": ("Round 7: road blockades, second run",
               "Different random start: also stops in front of every blockade. In very dense traffic the "
               "throughput stays lower and varies more than in the chosen run."),
    },
    "r7_DO_s1": {
        "short": "R7k",
        "de": ("Runde 7, Kontrolle: ohne Straßensperren",
               "Gleich viele Lernschritte, aber ohne Sperren im Training: Das Anhalten vor Sperren geht "
               "verloren, der Hindernis-Test fällt in mehreren Zwischenständen durch."),
        "en": ("Round 7, control: without road blockades",
               "The same number of learning steps, but without blockades in training: stopping in front of "
               "blockades is lost, the obstacle test fails at several intermediate steps."),
    },
    "r8_M_s1": {
        "short": "R8",
        "de": ("Runde 8: Gedächtnis",
               "Das aktuelle Gehirn: das Gehirn aus Runde 7 mit einem Gedächtnis aus 16 Zahlen pro Auto, das von "
               "Augenblick zu Augenblick weitergegeben wird. Am Anfang ist es leer und ändert nichts am Fahren; "
               "in 300 Lernschritten mit dem Rezept aus Runde 7 hat das Gehirn gelernt, es zu nutzen. Gewählt ist "
               "der letzte Stand (Schritt 1398). Vor einer Sperre hält es in {block} % der Fälle an. Vor einem "
               "einzelnen Hindernis kommt es zu {safe} % unfallfrei vorbei und hält in rund {stop} % davor an."),
        "en": ("Round 8: memory",
               "The current brain: the round 7 brain with a memory of 16 numbers per car, passed on from moment "
               "to moment. At first it is empty and changes nothing about the driving; in 300 learning steps with "
               "the round 7 recipe the brain learned to use it. The last step (step 1398) is chosen. In front of a "
               "blockade it stops in {block}% of cases. In front of a single obstacle it gets past without a crash "
               "in {safe}% of cases and stops in front of it in about {stop}%."),
    },
    "r8_M_s2": {
        "short": "R8'",
        "de": ("Runde 8: Gedächtnis, zweiter Lauf",
               "Anderer Zufallsstart: nutzt das Gedächtnis ebenfalls. Bei sehr dichtem Verkehr kommen mehr Autos "
               "durch als mit dem Gehirn aus Runde 7, etwas weniger als beim gewählten Lauf; sein letzter Stand hat "
               "dabei weniger Unfälle."),
        "en": ("Round 8: memory, second run",
               "Different random start: also uses the memory. In very dense traffic more cars get through than "
               "with the round 7 brain, slightly fewer than in the chosen run; its last step has fewer crashes "
               "there."),
    },
    "r8_C_s1": {
        "short": "R8k",
        "de": ("Runde 8, Kontrolle: ohne Gedächtnis",
               "Gleich viele Lernschritte mit demselben Rezept, aber ohne Gedächtnis: ebenfalls besser als Runde 7, "
               "doch bei sehr dichtem Verkehr kommen weniger Autos durch als mit Gedächtnis, und die Autos stehen "
               "fast doppelt so lange verkeilt."),
        "en": ("Round 8, control: without memory",
               "The same number of learning steps with the same recipe, but without memory: also better than "
               "round 7, yet in very dense traffic fewer cars get through than with memory, and the cars stand "
               "wedged almost twice as long."),
    },
}

STEP_INFO = {
    "es2": {
        "de": ("Start mit Zufallsgehirn; Stehenbleiben kostet Punkte → die Autos fahren los, aber "
               "Rechtsabbiegen endet im Bordstein"),
        "en": ("Start with a random brain; standing still costs points → the cars set off, but turning right "
               "ends on the curb"),
    },
    "es3": {
        "de": "Größere Zufallsänderungen → kaum Fortschritt",
        "en": "Bigger random changes → hardly any progress",
    },
    "es5": {
        "de": "Weg wird entlang der Straße gemessen, Warten kostet → alle drei Richtungen klappen meistens",
        "en": "Progress is measured along the road, waiting costs → all three directions mostly work",
    },
    "ppo6": {
        "de": "Start bei Null; Punkte für jeden Meter Richtung Ziel → die meisten kommen an",
        "en": "Start from zero; points for every metre towards the goal → most cars arrive",
    },
    "ppo8": {
        "de": "Ruhiger ausprobieren, weniger Zappeln → gut, aber nur auf fast leeren Straßen geübt",
        "en": "Calmer experimenting, less twitching → good, but only practised on nearly empty roads",
    },
    "ppo9": {
        "de": "Üben mit viel dichterem Verkehr → Kurve fällt erst, weil es schwerer wird; dann weniger Unfälle",
        "en": ("Practising with much denser traffic → the curve drops at first because it gets harder; then "
               "fewer crashes"),
    },
    "ppo13": {
        "de": ("Kleinere Lernschritte → stetig besser; gegen Ende noch mehr Verkehr, daher wieder etwas mehr "
               "Unfälle"),
        "en": ("Smaller learning steps → steadily better; towards the end even more traffic, so slightly more "
               "crashes again"),
    },
    "ppo17": {
        "de": ("Noch kleinere Schritte, mehr Blick auf die Folgen → bester Stand auf der alten, schmalen "
               "Kreuzung"),
        "en": "Even smaller steps, more look at the consequences → best result on the old, narrow crossing",
    },
    "wide_s1": {
        "de": ("Neue Kreuzung: Straße doppelt so breit, Reifen rutschen in zu schnellen Kurven → von Null an "
               "neu gelernt, die Kurve beginnt wieder unten"),
        "en": ("New crossing: road twice as wide, tyres slip in corners taken too fast → learned from scratch, "
               "the curve starts low again"),
    },
    "wide_s1r": {
        "de": "Rückwärtsgang dazu → kaum benutzt, es geht weiter bergauf",
        "en": "Reverse gear added → hardly used, it keeps climbing",
    },
    "wide_s1ft": {
        "de": "Kleinere Lernschritte → weniger Unfälle",
        "en": "Smaller learning steps → fewer crashes",
    },
    "wide_s1ft2": {
        "de": "Länger feinschleifen → nur noch wenig besser: das Hauptgehirn bis Runde 1",
        "en": "Fine-tuning for longer → only a little better: the main brain up to round 1",
    },
    "r1_x3": {
        "de": ("Neue Regeln: Wracks mit Warnblinker, Bremslicht, eine Lampe, Einfahrt nur rechts; Unfälle "
               "kosten dreimal so viel, geübt wird mit dichterem Verkehr → die Kurve fällt erst, dann lernt "
               "das alte Gehirn um"),
        "en": ("New rules: wrecks with hazard lights, brake light, one lamp, entering only from the right; "
               "crashes cost three times as much, practising with denser traffic → the curve drops at first, "
               "then the old brain relearns"),
    },
    "r1_main_x3": {
        "de": "Länger weitergelernt → rund 40 % weniger Unfälle als vorher: das Hauptgehirn bis Runde 2",
        "en": "Kept learning for longer → about 40% fewer crashes than before: the main brain up to round 2",
    },
    "r2_B_s1": {
        "de": ("Dazu Hindernisse im Training (Wracks, parkende Autos) → bremst viel öfter dahinter ab und "
               "reiht sich ein; noch Unfälle mitten auf der Kreuzung bei dichtem Verkehr"),
        "en": ("Plus obstacles in training (wrecks, parked cars) → brakes behind them far more often and "
               "queues up; still crashes in the middle of the crossing in dense traffic"),
    },
    "r4_S_s1": {
        "de": ("Neues Gehirn: ein gemeinsames Mini-Netz für jeden der bis zu 8 Nachbarn plus Aufmerksamkeit; "
               "erst das vorige Gehirn nachgeahmt, dann weitergelernt → im Vergleich (gleich lange trainiert "
               "wie das bisherige und ein doppelt so breites Netz) die wenigsten Unfälle"),
        "en": ("New brain: one shared mini-network for each of the up to 8 neighbours plus attention; first "
               "imitated the previous brain, then kept learning → fewest crashes in the comparison (trained "
               "for the same time as the previous one and a network twice as wide)"),
    },
    "r5_K_s1": {
        "de": ("Dasselbe Gehirn weitertrainiert, ein Zwischenstand mit wenigen Unfällen und hohem Durchsatz → "
               "bei dichtem Verkehr mehr Autos durch und weniger Unfälle; das Hauptgehirn bis Runde 6"),
        "en": ("The same brain trained on, an intermediate step with few crashes and high throughput → more "
               "cars through and fewer crashes in dense traffic; the main brain up to round 6"),
    },
    "r6_DO_s2": {
        "de": ("Weitertrainiert mit Verkehr bis 120 Autos pro Minute und mehr Hindernis-Szenen, ein Zwischenstand "
               "→ bei sehr dichtem Verkehr deutlich mehr Autos durch, weniger Unfälle und viel weniger Stau; vor "
               "einem stehenden Hindernis fährt es aber meist herum, statt anzuhalten; das Hauptgehirn bis Runde 7"),
        "en": ("Trained on with traffic of up to 120 cars per minute and more obstacle scenarios, an "
               "intermediate step → far more cars through in very dense traffic, fewer crashes and much less "
               "jamming; but in front of a standing obstacle it mostly drives around instead of stopping; the "
               "main brain up to round 7"),
    },
    "r7_DOB_s1": {
        "de": ("Dazu Straßensperren im Training (drei Hindernisse über die ganze Breite, nur Anhalten hilft), "
               "ein Zwischenstand → bei sehr dichtem Verkehr noch mehr Autos durch und weniger Unfälle; vor einer "
               "Sperre hält es an; das Hauptgehirn bis Runde 8"),
        "en": ("Plus road blockades in training (three obstacles across the full width, only stopping helps), "
               "an intermediate step → even more cars through in very dense traffic and fewer crashes; it stops "
               "in front of a blockade; the main brain up to round 8"),
    },
    "r8_M_s1": {
        "de": ("Gedächtnis dazu: 16 Zahlen pro Auto, die von Augenblick zu Augenblick weitergehen → bei sehr "
               "dichtem Verkehr mehr Autos durch und nur noch halb so lange verkeilt; vor einer Sperre hält es "
               "in {block} % der Fälle an, vor einem einzelnen Hindernis in etwa {stop} %"),
        "en": ("Memory added: 16 numbers per car, carried from moment to moment → in very dense traffic more "
               "cars through and wedged for only half as long; it stops in front of a blockade in {block}% "
               "of cases, in front of a single obstacle in about {stop}%"),
    },
}


def run_info(run: str | None, fill=lambda s: s) -> dict:
    """Names / notes of a run for the pages: German in label / note / step, English in *_en (the pages pick one:
    SCX.tl). Unmapped runs: label = the run id, no note. fill(text) fills the {safe} / {stop} / {block} placeholders."""
    if not run:
        return {"short": None, "label": None, "label_en": None, "note": None, "note_en": None, "step": None,
                "step_en": None}
    e = RUN_INFO.get(run, {"short": run, "de": (run, None), "en": (run, None)})
    step = STEP_INFO.get(run, {})
    return {"short": e["short"], "label": e["de"][0], "label_en": e["en"][0], "note": fill(e["de"][1]),
            "note_en": fill(e["en"][1]), "step": fill(step.get("de")), "step_en": fill(step.get("en"))}
