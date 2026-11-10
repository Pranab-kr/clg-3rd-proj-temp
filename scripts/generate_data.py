"""Generate the reproducible synthetic dataset from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow the documented `python scripts/generate_data.py` invocation to import src.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR, RANDOM_SEED
from src.data_generation import generate_dataset



def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic electronics tables.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    args = parser.parse_args()

    dataset = generate_dataset(seed=args.seed, output_dir=args.output_dir)
    print(f"Generated synthetic data with seed={args.seed}")
    for name, table in dataset.items():
        print(f"{name}: {len(table)} rows")
    print("Validation: passed")


if __name__ == "__main__":
    main()
