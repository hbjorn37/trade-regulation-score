"""Project path resolution.

The pipeline writes to ``intermediate/`` and ``output/`` next to the repo root
and reads samples from ``data/sample/`` by default. A future user can point at
real data by passing ``--data-root`` to the step modules.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_ROOT_DEFAULT = REPO_ROOT / "data" / "sample"
INTERMEDIATE_ROOT = REPO_ROOT / "intermediate"
OUTPUT_ROOT = REPO_ROOT / "output"
