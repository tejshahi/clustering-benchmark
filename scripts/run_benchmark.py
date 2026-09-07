#!/usr/bin/env python3
"""Convenience wrapper so the benchmark can be run as `python scripts/run_benchmark.py ...`
without needing `python -m` / PYTHONPATH juggling.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_bench.run_benchmark import main  # noqa: E402

if __name__ == "__main__":
    main()
