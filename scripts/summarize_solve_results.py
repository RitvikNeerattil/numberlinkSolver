#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print solved percentage from DeepXube solve results.pkl")
    parser.add_argument("--results", required=True, help="Path to results.pkl")
    args = parser.parse_args()

    results_path = Path(args.results)
    data = pickle.load(results_path.open("rb"))

    solved = data.get("solved", [])
    total = len(solved)
    solved_count = int(sum(bool(x) for x in solved))
    solved_pct = (100.0 * solved_count / total) if total else 0.0

    print(f"results_file: {results_path}")
    print(f"solved: {solved_count}/{total} ({solved_pct:.2f}%)")


if __name__ == "__main__":
    main()
