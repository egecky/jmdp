import argparse
import csv
import json
from pathlib import Path


def mean(rows, key):
    vals = [float(r[key]) for r in rows]
    return sum(vals) / max(1, len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/ale")
    args = ap.parse_args()

    for path in sorted(Path(args.root).glob("*/seed_*/**/gap_eval/gap_eval_summary.json")):
        rows = list(csv.DictReader(path.with_name("gap_eval.csv").open()))
        summary = json.loads(path.read_text())
        print()
        print(path)
        print("mean_rmse:", summary.get("mean_rmse"))
        print("var_rmse:", summary.get("var_rmse"))
        print("var_corr:", summary.get("var_corr"))
        print("pred_gap_var_mean:", mean(rows, "pred_var"))
        print("mc_gap_var_mean:", mean(rows, "mc_var"))
        print("pred_ind_var_mean:", mean(rows, "pred_var_independent"))
        print("mc_ind_var_mean:", mean(rows, "mc_var_independent"))


if __name__ == "__main__":
    main()
