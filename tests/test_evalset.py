"""Standard eval protocol (evalset.py, cli evalset) and the shared bench (bench_one, cli bench)."""

import json
from pathlib import Path

from smartcrossing import cli, evalset
from smartcrossing.sim import SimConfig

ROOT = Path(__file__).resolve().parents[1]
ROUND4 = ROOT / "results" / "round4_best.npz"


def _format_of(path):
    """(kind, keys) of the lines of a committed eval file."""
    return {(d["kind"], tuple(d)) for d in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def test_cli_evalset_tiny_matches_the_committed_format(tmp_path):
    out = tmp_path / "x.jsonl"
    cli.main(["evalset", "--driver", "reservation", "--driver", f"policy:{ROUND4}", "--demands", "0.3", "--seeds",
              "1000", "--seconds", "5", "--envs", "2", "--out", str(out)])
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [(Path(d["driver"]).name, d["kind"]) for d in lines] == [
        ("reservation", "bench"), ("reservation", "gate"), ("round4_best.npz", "bench"), ("round4_best.npz", "gate")]
    assert lines[0]["seed"] == 1000 and lines[0]["demand"] == 0.3
    ref = _format_of(ROOT / "results" / "round4_eval.jsonl")
    # same keys in the same order; the gate lines also carry the block cases (block_*, added with the blockade scenarios of round 7)
    assert all(any(k.startswith("block_") for k in d) for d in lines if d["kind"] == "gate")
    assert {(d["kind"], tuple(k for k in d if not k.startswith("block_"))) for d in lines} == ref
    # appends
    cli.main(["evalset", "--driver", "light", "--demands", "0.05", "--seeds", "1", "--seconds", "2", "--envs", "1",
              "--out", str(out)])
    assert len(out.read_text(encoding="utf-8").splitlines()) == 6


def test_bench_line_is_cli_bench(capsys):
    cli.main(["bench", "--driver", "light", "--demand", "0.2", "--envs", "2", "--seconds", "20", "--seed", "7"])
    printed = json.loads(capsys.readouterr().out.strip())
    line = evalset.bench_line("light", SimConfig(demand=0.2), 7, 2, 20.0)
    printed.pop("wall_s"), line.pop("wall_s")
    assert json.dumps(printed) == json.dumps(line)  # (NaN != NaN: compare the JSON)
    # a callable driver gives the same run as its spec
    from smartcrossing.baselines import LaneDriver

    a = evalset.bench_one("light", SimConfig(demand=0.2), 7, 2, 20.0)
    b = evalset.bench_one(lambda sim: LaneDriver(sim, "light"), SimConfig(demand=0.2), 7, 2, 20.0)
    assert evalset.keep(a) == evalset.keep(b) and set(evalset.keep(a)) == set(evalset.KEEP_KEYS)


def test_evalset_workers_same_lines():
    kw = dict(demands=(0.15,), seeds=(1000, 2000), envs=1, seconds=3.0, gate=False, log=lambda *_: None)
    one = evalset.evalset(["light"], **kw)
    two = evalset.evalset(["light"], workers=2, **kw)
    for d in one + two:
        d.pop("wall_s")
    assert json.dumps(one) == json.dumps(two) and len(one) == 2
