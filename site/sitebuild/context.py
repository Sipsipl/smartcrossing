"""The build context: paths (all below one repository root, so tests can point it at tmp_path), where the
checkpoints / logs come from (runs/, the committed snapshot results/site/, or both merged), the discovered
checkpoints and the stop gates measured so far. Passed explicitly instead of module-global state."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def method_of(run: str) -> str | None:
    """Training method of a run folder name: 'es', 'ppo' (ppo*, wide*, r<N>_* training rounds) or None (no
    training run: prof_*, smoke_*, ...)."""
    if run.startswith("es"):
        return "es"
    # wide_*: PPO on the wide crossing; r<N>_*: training round N (PPO, warm-started)
    if run.startswith("ppo") or run.startswith("wide") or re.match(r"r\d+_", run):
        return "ppo"
    return None


@dataclass
class Context:
    root: Path = REPO
    kind: str = "runs"  # "runs", "snapshot" or "merge" (setup)
    manifest: dict | None = None  # the snapshot's manifest.json (snapshot / merge)
    train_runs: set = field(default_factory=set)  # training runs found in runs/
    found: dict = field(default_factory=dict)  # discovered checkpoints: id -> info (discover.discover)
    gates: dict = field(default_factory=dict)  # checkpoint id -> stop-gate rates (fundamental.gate_of)
    note_ckpt: str | None = None  # checkpoint whose stop gate fills {safe} / {stop} in the run texts

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def snapshot(self) -> Path:
        """Committed copy of what the build uses from runs/ (see the package docstring)."""
        return self.root / "results" / "site"

    @property
    def site(self) -> Path:
        return self.root / "site"

    @property
    def src(self) -> Path:
        return self.site / "src"

    @property
    def out(self) -> Path:
        return self.site / "public"

    @property
    def cache(self) -> Path:
        return self.site / "data" / "bench_cache.json"

    def from_runs(self, run: str) -> bool:
        """True if `run` is read from runs/ (source runs, or merge and the run exists in runs/)."""
        return self.kind == "runs" or (self.kind == "merge" and run in self.train_runs)

    def setup(self, source: str = "auto") -> str:
        """Resolve --source (auto -> merge if runs/ has training runs and a snapshot exists, else whichever
        exists) and read the snapshot manifest if needed; returns the kind."""
        kind = source
        train_runs = {p.name for p in self.runs.iterdir() if p.is_dir() and method_of(p.name)} if self.runs.exists() else set()
        has_snap = (self.snapshot / "manifest.json").exists()
        if kind == "auto":
            kind = ("merge" if train_runs else "snapshot") if has_snap else "runs"
        self.kind, self.train_runs = kind, train_runs
        if kind in ("snapshot", "merge"):
            if not has_snap:
                raise SystemExit(f"no runs/ and no snapshot in {self.snapshot.relative_to(self.root)}/")
            self.manifest = json.loads((self.snapshot / "manifest.json").read_text(encoding="utf-8"))
        print(f"source: {kind}" + (f" (+ runs/: {', '.join(sorted(train_runs))})" if kind == "merge" else ""))
        return kind
