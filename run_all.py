"""Reproduce the whole submission: tests -> train -> predict -> score.py.

    python run_all.py                  # core pipeline (~2 min)
    python run_all.py --full           # also EDA, model comparison and insights (~20 min)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="also rerun EDA, model comparison and insights")
    full = parser.parse_args().full

    run("-m", "unittest", "discover", "-s", "tests")
    if full:
        run("scripts/eda.py")
        run("scripts/experiments.py")
        run("scripts/compare_algorithms.py")
    run("scripts/train.py")
    run("scripts/predict.py")
    run("score.py", "--predictions", "validation_predictions.csv",
        "--december-predictions", "data/december_chart_inputs.csv")
    if full:
        run("scripts/insights.py")


if __name__ == "__main__":
    main()
