"""Build the static website into site/public/ (commit the result; Cloudflare Pages
serves that folder as is, see docs/DEPLOY.md).

    poetry run python site/build.py                       # runs/ if present, else results/site/
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

runs/ is gitignored. A build from runs/ also writes a compact snapshot of everything it used
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

from obs_labels import obs_layout  # noqa: E402
from smartcrossing.policy import default_in_scale, init_params, layer_sizes, unpack  # noqa: E402
from smartcrossing.policy_compat import load_adapted_policy, physics_of  # noqa: E402
from smartcrossing.sim import SimConfig, js_constants  # noqa: E402

SITE = ROOT / "site"
SRC = SITE / "src"
OUT = SITE / "public"
CACHE = SITE / "data" / "bench_cache.json"
RUNS = ROOT / "runs"
SNAPSHOT = ROOT / "results" / "site"  # committed copy of what the build uses from runs/ (see docstring)
SOURCE = {"kind": "runs", "manifest": None}  # set in main(): "runs" or "snapshot"
MP_START = "fork" if "fork" in mp.get_all_start_methods() else "spawn"  # Windows has no fork

DEMANDS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]  # up to the start page slider maximum
# the site's main brain (wide crossing) and the end of the older line whose training history is
# shown before it (narrow crossing); override with --main / --history
# Checkpoints are named <run>/<file stem> (found in runs/ or in the snapshot); a path to an
# .npz anywhere (e.g. results/wide_s1ft2_ckpt2650.npz) works too if its weights match one.
DEFAULT_MAIN = ["wide_s1ft2/ckpt_2650"]  # = results/wide_s1ft2_ckpt2650.npz
DEFAULT_HISTORY = "ppo17/ckpt_5250"
DEFAULT_MIX = "wide_s0ft/ckpt_2850"  # = results/wide_s0ft_ckpt2850.npz; second brain for mixing on the start page
TIMELINE_DEMANDS = [0.05, 0.15]
BENCH_ENVS, BENCH_SECONDS, BENCH_SEED = 8, 120.0, 1000
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
                   "Noch eine Stunde Feinschliff: nur noch etwas besser – das aktuelle Gehirn."),
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
    "wide_s1ft2": "Länger feinschleifen → nur noch wenig besser: das aktuelle Gehirn",
}


def run_info(run: str | None) -> dict:
    short, label, note = RUN_INFO.get(run, (run, run, None)) if run else (None, None, None)
    return {"short": short, "label": label, "note": note, "step": STEP_INFO.get(run) if run else None}


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
    if run.startswith("ppo") or run.startswith("wide"):  # wide_*: PPO on the wide crossing
        return "ppo"
    return None


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


def policy_vector(d: dict) -> np.ndarray:
    return np.concatenate([np.concatenate([np.ravel(np.asarray(l["W"], dtype=float)), np.asarray(l["b"], dtype=float)]) for l in d["layers"]])


def policy_hash(pol: dict) -> str:
    return hashlib.sha1(policy_vector(pol).tobytes() + json.dumps(pol["in_scale"]).encode()).hexdigest()[:16]


def _entry(run: str, stem: str, file: str, method: str, gen, physics: dict, pol: dict) -> dict:
    return {"id": f"{run}/{stem}", "run": run, "file": file, "method": method, "gen": gen,
            "physics": physics, "policy": pol, "adapted": pol.get("adapted"), "hash": policy_hash(pol)}


def discover(world: str = "any") -> dict:
    """All loadable checkpoints of runs/es*, runs/ppo* and runs/wide* (or of the snapshot):
    id -> info (only those trained on `world` = narrow / wide crossing, unless 'any')."""
    found = {}
    if SOURCE["kind"] == "snapshot":
        for f in sorted((SNAPSHOT / "brains").glob("*/*.npz")):
            with np.load(f) as z:
                info = json.loads(str(z["info"]))
                pol = {**info["policy"], "in_scale": z["in_scale"].tolist(),
                       "layers": [{"W": z[f"W{i}"].tolist(), "b": z[f"b{i}"].tolist()} for i in range(info["n_layers"])]}
            if world != "any" and world_of(info["physics"]) != world:
                continue
            e = _entry(info["run"], f.stem, info["file"], info["method"], info["gen"], info["physics"], pol)
            found[e["id"]] = e
        return found
    if not RUNS.exists():
        return found
    for run_dir in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        method = method_of(run_dir.name)
        if not method:
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
            e = _entry(run_dir.name, f.stem, str(f.relative_to(ROOT)), method, meta["gen"], physics_of(meta["cfg"]), pol)
            found[e["id"]] = e
    return found


def init_of(run: str) -> tuple[str, int] | None:
    """(run, generation) of the checkpoint `run` was warm-started from, if it is a known run."""
    if SOURCE["kind"] == "snapshot":
        r = SOURCE["manifest"]["runs"].get(run, {}).get("init")
        return tuple(r) if r else None
    p = RUNS / run / "args.json"
    init = (json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}).get("init") or ""
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
def _bench_job(job):
    kind, payload, physics, demand = job
    from smartcrossing.baselines import LaneDriver
    from smartcrossing.metrics import summarize
    from smartcrossing.policy import PolicyDriver
    from smartcrossing.sim import Sim

    cfg = SimConfig(**{**physics, "demand": demand})
    sim = Sim(cfg, BENCH_ENVS, seed=BENCH_SEED)
    if kind == "baseline":
        drv = LaneDriver(sim, payload)
    else:
        params, sizes, in_scale = payload
        drv = PolicyDriver(sim, np.asarray(params), sizes, in_scale=np.asarray(in_scale))
    for _ in range(int(BENCH_SECONDS / cfg.dt)):
        sim.step(drv.act())
    r = summarize(sim, BENCH_SECONDS)
    keep = ("throughput_per_min", "demand_per_min", "exit_ok_rate", "crash_cars_per_1000", "crash_cars",
            "offroad_cars", "timeouts", "delay_mean_s")
    return {k: (round(float(r[k]), 3) if r[k] == r[k] else None) for k in keep}


def run_bench(entries: list[tuple[str, str, object, dict, list]], force: bool, workers: int) -> dict:
    """entries: (key, kind, payload, physics, demands) -> {key: {demand: result}}."""
    sim_src = hashlib.sha1((ROOT / "smartcrossing" / "sim.py").read_bytes()
                           + (ROOT / "smartcrossing" / "baselines.py").read_bytes()).hexdigest()[:12]
    cache = {}
    if CACHE.exists() and not force:
        cache = json.loads(CACHE.read_text(encoding="utf-8"))
    jobs, keys = [], []
    for key, kind, payload, physics, demands in entries:
        for d in demands:
            ck = f"{key}|{json.dumps(physics, sort_keys=True)}|{d}|{BENCH_ENVS}x{BENCH_SECONDS}@{BENCH_SEED}|{sim_src}"
            if ck in cache:
                continue
            jobs.append((kind, payload, physics, d))
            keys.append(ck)
    if jobs:
        print(f"  benchmarking {len(jobs)} runs ({BENCH_ENVS} envs x {BENCH_SECONDS:.0f} s each) ...")
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
            ck = f"{key}|{json.dumps(physics, sort_keys=True)}|{d}|{BENCH_ENVS}x{BENCH_SECONDS}@{BENCH_SEED}|{sim_src}"
            out[key][f"{d:g}"] = cache[ck]  # as JS String(d): "1" for 1.0
    return out


# ------------------------------------------------------------------ replays ("Vergleich" side by side)
REPLAY_DEMANDS = [0.05, 0.15, 0.3]
REPLAY_SEED, REPLAY_WARMUP, REPLAY_SECONDS = 7, 15.0, 90.0
GEOMETRY_KEYS = ("L", "W", "Rc", "a_lat_max", "v_reverse")  # the "world": same for all drivers compared
Q_POS, Q_TH, Q_LAMP = 0.05, 0.01, 100.0  # quantisation: 5 cm, 0.01 rad, lamp in 1/100
BRAKE_ACC = -0.5  # m/s^2: brake lights on below this actual acceleration (purely visual)


def _b64(a: np.ndarray) -> str:
    import base64

    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _replay_job(job):
    """Drive one sim (1 env) and record every car as a compact track:
    [t0, man, end_step, end_kind, qx0, qy0, qth0, width, b64, brake] where b64 holds per sample
    (dx, dy, dth[, lamp0, lamp1]) as int8 (width 1) or int16 (width 2), deltas of the
    quantised values (exact reconstruction), and brake the sample ranges [start, end) with the
    brake lights on (acceleration < BRAKE_ACC), flattened. Plus crash marks and the queue per arm."""
    from smartcrossing.baselines import LaneDriver
    from smartcrossing.policy import PolicyDriver
    from smartcrossing.sim import DONE_NAMES, Sim

    kind, payload, physics, demand = job
    cfg = SimConfig(**{**physics, "demand": demand})
    sim = Sim(cfg, 1, seed=REPLAY_SEED)
    if kind == "baseline":
        drv = LaneDriver(sim, payload)
    else:
        params, sizes, in_scale = payload
        drv = PolicyDriver(sim, np.asarray(params), sizes, in_scale=np.asarray(in_scale))
    lamps = kind != "baseline" and cfg.n_signal >= 2
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
        for slot in np.nonzero(sim.alive[0])[0]:
            uid = int(sim.uid[0, slot])
            tr = tracks.get(uid)
            if tr is None:
                tr = tracks[uid] = {"t0": k, "man": int(sim.man[0, slot]), "s": [], "b": [], "end": (None, 0)}
            tr["b"].append(bool(sim.acc[0, slot] < BRAKE_ACC))
            tr["s"].append((sim.x[0, slot], sim.y[0, slot], sim.th[0, slot], *(sim.sig[0, slot, :2] if lamps else ())))
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
                    int(q[0, 0]), int(q[0, 1]), int(q[0, 2]), width, _b64(d.astype(np.int8 if width == 1 else "<i2")), runs])
    return {
        "demand": demand, "steps": n, "dt": cfg.dt, "lamps": lamps,
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
    p = c["policy"]
    return (f"pol:{c['hash']}", "policy", (policy_vector(p).tolist(), p["sizes"], p["in_scale"]), c["physics"])


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
    if SOURCE["kind"] == "snapshot":  # stored with offset 0 (already thinned)
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
    if SOURCE["kind"] == "snapshot":
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
        info = {"run": c["run"], "file": c["file"], "method": c["method"], "gen": c["gen"], "physics": c["physics"],
                "n_layers": len(pol["layers"]),
                "policy": {k: v for k, v in pol.items() if k not in ("layers", "in_scale")}}
        arrays = {"in_scale": np.asarray(pol["in_scale"], dtype=float)}
        for i, l in enumerate(pol["layers"]):
            arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(l["W"], dtype=float), np.asarray(l["b"], dtype=float)
        path = brains / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, info=json.dumps(info, sort_keys=True), **arrays)
    for rel in old - keep:
        (brains / rel).unlink()
    for d in brains.iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    runs = sorted(set(chain_runs) | {found[cid]["run"] for cid in used})
    manifest = {
        "note": "written by site/build.py from runs/ (see its docstring); read when runs/ is absent",
        "runs": {r: {"init": list(init_of(r)) if init_of(r) else None, "world": run_world(r, found)} for r in runs},
        "curves": {r: read_curve(r, 0) for r in sorted(set(chain_runs))},
    }
    (SNAPSHOT / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    size = sum(f.stat().st_size for f in SNAPSHOT.rglob("*") if f.is_file())
    print(f"  snapshot: {len(keep)} brains + manifest in {SNAPSHOT.relative_to(ROOT)}/ ({size / 1024:.0f} KiB)")


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


def timeline(found: dict, best_id: str, n_max: int) -> list[dict]:
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
        elif p == pts[-1] or p[1] == best_id:  # the final / chosen checkpoint always stays
            kept[-1] = p
    pts = kept
    if len(pts) > n_max:
        idx = np.unique(np.linspace(0, len(pts) - 1, n_max).round().astype(int))
        pts = [pts[i] for i in idx]
    return [{"gen": g, "id": cid} for g, cid in pts], chain


def round_policy(d: dict, digits: int = 5) -> dict:
    """Shrink the JSON: weights rounded to `digits` decimals."""
    r = lambda a: np.round(np.asarray(a, dtype=float), digits).tolist()  # noqa: E731
    return {**d, "in_scale": r(d["in_scale"]), "layers": [{"W": r(l["W"]), "b": r(l["b"])} for l in d["layers"]]}


def random_policy() -> dict:
    cfg = SimConfig()
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
    items = []
    for href, label in PAGES:
        cur = ' aria-current="page"' if href == current else ""
        items.append(f'<a href="{href}"{cur}>{label}</a>')
    nav = '<nav class="nav" aria-label="Seiten">' + "".join(items) + "</nav>"
    return f'<header class="top"><a class="brand" href="index.html">{LOGO}Smart Crossing</a>{nav}</header>'


REPO = "https://github.com/Sipsipl/smartcrossing"
FOOTER = (
    '<footer>Smart Crossing – ein Hobbyprojekt. Alles läuft direkt in deinem Browser, '
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


def main(argv=None):
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
    ap.add_argument("--source", choices=["auto", "runs", "snapshot"], default="auto",
                    help="where the checkpoints / logs come from: runs/ (also refreshes the snapshot), the committed "
                         "snapshot results/site/, or auto (runs/ if it holds training runs, else the snapshot)")
    a = ap.parse_args(argv)
    if a.main is None:
        a.main = list(DEFAULT_MAIN)
    kind = a.source
    if kind == "auto":
        kind = "runs" if RUNS.exists() and any(method_of(p.name) for p in RUNS.iterdir() if p.is_dir()) else "snapshot"
    SOURCE["kind"] = kind
    if kind == "snapshot":
        if not (SNAPSHOT / "manifest.json").exists():
            raise SystemExit(f"no runs/ and no snapshot in {SNAPSHOT.relative_to(ROOT)}/")
        SOURCE["manifest"] = json.loads((SNAPSHOT / "manifest.json").read_text(encoding="utf-8"))
    print(f"source: {kind}")

    cfg = SimConfig()
    layout = obs_layout(cfg)  # fails loudly if the sim's observation layout changed
    print("discovering checkpoints ...")
    found = discover(a.world)
    print(f"  {len(found)} loadable checkpoints")

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
    print(f"  best per method: {picked}, main: {main_id}, history: {hist_id}")

    policies, bench_entries = {}, []

    def use(cid, demands):
        c = found[cid]
        if cid not in policies:
            policies[cid] = {
                "id": cid, "run": c["run"], "run_label": run_info(c["run"])["label"], "file": c["file"], "method": c["method"], "gen": c["gen"],
                "physics": c["physics"], "adapted": c["adapted"], "policy": round_policy(c["policy"], 4),
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
            pts, chain = timeline(found, pid, max(2, a.timeline // 2) if i < len(parts) - 1 else a.timeline)
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
        rp = random_policy()
        rid = "zufall"
        policies[rid] = {"id": rid, "run": None, "file": None, "method": "es", "gen": 0, "physics": phys,
                         "adapted": None, "policy": round_policy(rp, 4), "bench": {}}
        bench_entries.append(("pol:random7", "policy", (policy_vector(rp).tolist(), rp["sizes"], rp["in_scale"]), phys, TIMELINE_DEMANDS))
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
    # the rule-based drivers run in the same world (geometry, grip) as the main policy, and also
    # in the world of each other brain on "Vergleich" (old narrow crossing), so every brain is
    # compared on its own crossing
    def world_phys(cid):
        return {**physics_of(cfg.__dict__), **{k: v for k, v in found[cid]["physics"].items() if k in GEOMETRY_KEYS}}

    base_worlds = {}  # world name -> physics; the main brain's world first
    for cid in (main_id, hist_id, picked.get("es")):
        if cid:
            base_worlds.setdefault(world_of(found[cid]["physics"]), world_phys(cid))
    if not base_worlds:
        base_worlds[world_of(physics_of(cfg.__dict__))] = physics_of(cfg.__dict__)
    main_world = next(iter(base_worlds))
    base_phys = base_worlds[main_world]
    base_key = lambda drv, w: f"base:{drv}" if w == main_world else f"base:{drv}:{w}"  # noqa: E731
    for w, phys in base_worlds.items():
        for drv in ("light", "reservation"):
            bench_entries.append((base_key(drv, w), "baseline", drv, phys, DEMANDS))
    bench = run_bench(bench_entries, a.bench, a.workers)
    for cid, p in policies.items():
        key = f"pol:{found[cid]['hash']}" if cid in found else "pol:random7"
        p["bench"] = bench[key]
    baselines_by_world = {w: {drv: bench[base_key(drv, w)] for drv in ("light", "reservation")} for w in base_worlds}
    baselines = baselines_by_world[main_world]

    data = {
        "built": date.today().isoformat(),
        "layout": layout,
        "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        "policies": policies,
        "best": picked,
        "main": found[main_id]["method"] if main_id else None,  # method of the main brain (shown first on the site)
        "main_id": main_id,
        "history": hist_id,
        "mix": mix_id,
        "timelines": timelines,
        "chains": chains,
        "curves": curves,
        "others": others,
        "baselines": baselines,  # in the main brain's world
        "baselines_by_world": baselines_by_world,  # world ("wide" / "narrow") -> {light, reservation}
        "main_world": main_world,
        "demands": DEMANDS,
        "bench_setup": {"envs": BENCH_ENVS, "seconds": BENCH_SECONDS, "seed": BENCH_SEED},
    }
    # side-by-side replays (same traffic) of the two rule-based drivers and the best policy
    replays = None
    if main_id:
        mp_ = found[main_id]
        _, _, payload, phys = pol_entry(mp_)
        replays = record_replays([
            ("light", "baseline", "light", base_phys),
            ("reservation", "baseline", "reservation", base_phys),
            ("policy", "policy", payload, phys),
        ], a.workers)
        replays["policy_id"] = main_id
        replays["world"] = {k: base_phys[k] for k in ("L", "W", "Rc") if k in base_phys}
    data["replay_world"] = world_of(base_phys)
    if kind == "runs":
        write_snapshot(found, [cid for cid in policies if cid in found], [r["run"] for ch in chains.values() for r in ch])
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
    for m, pts in timelines.items():
        print(f"  Zeitreise {m}: " + ", ".join(f"{p['id']}@{p['gen']}" for p in pts))


if __name__ == "__main__":
    main()
