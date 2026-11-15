#!/usr/bin/env python3
"""Verify the displayed results from *published CSVs*, not from raw checkpoint inference.

This script is a publication sanity check, not an experimental rerun.
Requires Python 3.10+; no third-party packages or API calls.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/external_36_metrics.csv"
FOLDS = ROOT / "results/fold_summary.csv"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verify():
    rows = read_csv(METRICS)
    expected = {
        (method, family, condition)
        for method in ("tfidf", "minilm", "rbt3")
        for family in ("random", "rule", "stealth", "skeptical", "stochastic", "deceptive")
        for condition in ("raw", "masked")
    }
    keys = [(r["method"], r["trained_fold"], r["condition"]) for r in rows]
    assert len(rows) == 36 and len(set(keys)) == 36 and set(keys) == expected, "Missing/duplicate model conditions"
    assert all(int(r["samples"]) == 1551 and int(r["games"]) == 100 for r in rows), "Dataset sizes differ"
    assert all(abs(float(r["positive_prevalence"]) - 0.30303) < 1e-5 for r in rows), "Prevalence differs"
    assert all(
        float(r["roc_auc_ci_low"]) <= float(r["roc_auc"]) <= float(r["roc_auc_ci_high"])
        for r in rows
    ), "AUC outside its stated interval"

    count_below = sum(float(r["roc_auc"]) < 0.5 for r in rows)
    count_ci_below = sum(float(r["roc_auc_ci_high"]) < 0.5 for r in rows)
    assert count_below == 34, f"Unexpected AUC count: {count_below}"
    assert count_ci_below == 26, f"Unexpected CI count: {count_ci_below}"

    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["condition"])].append(float(r["roc_auc"]))

    summary = read_csv(FOLDS)
    assert len(summary) == 6, "Expected six method/condition summary rows"
    for r in summary:
        key = (r["method"], r["condition"])
        assert len(groups[key]) == 6, f"Expected six trained folds for {key}"
        actual = statistics.median(groups[key])
        archived = float(r["auc_median_across_folds"])
        assert abs(actual - archived) < 1e-10, f"Median mismatch for {key}"

    print("CSV verification PASSED")
    print(f"External evaluation: {len(rows)} trained configurations, 100 shared games, 1,551 observations")
    print(f"AUC below 0.5: {count_below}/{len(rows)}")
    print(f"Pointwise bootstrap CI upper bound below 0.5: {count_ci_below}/{len(rows)}")
    print("Median ROC-AUC across correlated old training folds:")
    for method in ("tfidf", "minilm", "rbt3"):
        raw = statistics.median(groups[(method, "raw")])
        masked = statistics.median(groups[(method, "masked")])
        print(f"  {method:6s}: raw {raw:.4f}; masked {masked:.4f}")
    print("NOTE: This verifies published tables; it does not rerun training/inference.")


if __name__ == "__main__":
    verify()
