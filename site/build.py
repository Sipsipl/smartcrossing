"""Build the static website into site/public/: poetry run python site/build.py [--help].

Thin entry point; the build lives in the package site/sitebuild/ (its docstring explains the steps).
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]  # smartcrossing (repo root), obs_labels + sitebuild (site/)

from sitebuild.main import main  # noqa: E402

if __name__ == "__main__":
    main()
