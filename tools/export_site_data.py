# -*- coding: utf-8 -*-
"""
Export every number the GitHub Pages site needs into one JSON file.

The site is a static page: it has no Python runtime, so the fitted model has to
travel with it. This script re-fits the final OLS exactly as
`7115056116_hw2.py` does, then writes the coefficients, their full covariance
matrix and the residual scale to `docs/assets/data/site_data.json`.

With the covariance matrix in hand the page can compute, for any slider
combination the visitor dials in:

    point estimate   yhat   = x0 . beta
    mean std error   se     = sqrt(x0' V x0)
    95% CI           yhat +- t(.975, df) * se
    95% PI           yhat +- t(.975, df) * sqrt(sigma^2 + x0' V x0)

which are the same formulas the report uses in section 7.4 - not an
approximation of them.

Run after `7115056116_hw2.py`:

    python tools/export_site_data.py
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

# The main script prints superscripts (R²); on a cp950 console that raises
# before any work happens. Force UTF-8 out before importing it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "data" / "site_data.json"

AGGREGATES = [
    "agg_std", "agg_max", "agg_min", "agg_median",
    "agg_skew", "agg_sum_sq", "agg_sum_cu",
]


def load_hw2_module():
    """Import the main script. Its filename starts with a digit, so the normal
    import statement is not available."""
    path = next(ROOT.glob("*_hw2.py"))
    spec = importlib.util.spec_from_file_location("hw2_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hw2_main"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_csv_rows(path: Path) -> list[dict]:
    """Our tables are written with a BOM; utf-8-sig strips it so the first
    column name is not silently mangled into '\\ufeffterm'."""
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def numify(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if v in ("True", "False"):
                d[k] = (v == "True")
                continue
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def main() -> None:
    hw2 = load_hw2_module()
    print("Loaded", hw2.STUDENT_ID, "- re-fitting the final OLS for the site")

    # Reproduce the script's own pipeline so the exported model is the one the
    # report describes, down to the split.
    df, is_real = hw2.load_data(False)
    S = hw2.data_preparation(df)

    Xa_tr = S["X_aug_train"]
    y_tr = S["y_train"]
    cols = list(hw2.FEATURES) + AGGREGATES

    Xc_tr = sm.add_constant(Xa_tr[cols], has_constant="add")
    ols = sm.OLS(y_tr.values, Xc_tr.values).fit()

    terms = ["const"] + cols
    params = ols.params.tolist()
    cov = np.asarray(ols.cov_params()).tolist()

    # Cross-check against the table the main script already wrote; if these
    # disagree the site would quietly show different numbers from the report.
    ref = numify(read_csv_rows(ROOT / "outputs" / "tables" / "06_ols_coefficients.csv"))
    assert len(ref) == len(terms), f"{len(ref)} rows vs {len(terms)} terms"
    worst = 0.0
    for name, got, row in zip(terms, params, ref):
        assert row["term"] == name, f"term order differs: {row['term']} vs {name}"
        worst = max(worst, abs(got - row["coef"]))
    print(f"  coefficients match 06_ols_coefficients.csv to {worst:.3e}")
    assert worst < 1e-10, "re-fit does not reproduce the reported coefficients"

    # Slider ranges come from the training data the model actually saw.
    raw_tr = Xa_tr[list(hw2.FEATURES)]
    feature_stats = {
        f: {
            "min": float(raw_tr[f].min()),
            "max": float(raw_tr[f].max()),
            "mean": float(raw_tr[f].mean()),
            "median": float(raw_tr[f].median()),
            "std": float(raw_tr[f].std()),
        }
        for f in hw2.FEATURES
    }

    # The model carries agg_sum_sq and agg_sum_cu, so it extrapolates violently
    # once the row sum leaves the range it was fitted on. Ship the observed
    # range so the page can warn instead of quietly printing nonsense.
    row_sum_tr = raw_tr.sum(axis=1)
    row_sum = {
        "min": float(row_sum_tr.min()),
        "max": float(row_sum_tr.max()),
        "mean": float(row_sum_tr.mean()),
        "p01": float(row_sum_tr.quantile(0.01)),
        "p99": float(row_sum_tr.quantile(0.99)),
    }
    print("  train row-sum range: {min:.0f} .. {max:.0f} (mean {mean:.1f})".format(**row_sum))

    results = json.loads(io.open(ROOT / "outputs" / "results.json", encoding="utf-8").read())
    tab = ROOT / "outputs" / "tables"

    payload = {
        "meta": {
            "student_id": hw2.STUDENT_ID,
            "competition": "Kaggle Playground Series S4E5 - Flood Prediction",
            "competition_url": "https://www.kaggle.com/competitions/playground-series-s4e5",
            "target": "FloodProbability",
            "is_real_kaggle_data": bool(results["data_source"]["is_real_kaggle_data"]),
            "data_label": results["data_source"]["label"],
            "n_rows_modelled": int(results["data_source"]["n_rows"]),
            "n_train": int(len(y_tr)),
            "n_test": int(len(S["y_test"])),
            "test_size": hw2.TEST_SIZE,
            "random_state": hw2.RANDOM_STATE,
        },
        "features": list(hw2.FEATURES),
        "aggregates": AGGREGATES,
        "model": {
            "name": results["evaluation"]["final_model"],
            "terms": terms,
            "params": params,
            "cov": cov,
            "scale": float(ols.scale),          # sigma^2, the residual variance
            "df_resid": float(ols.df_resid),
            "rsquared": float(ols.rsquared),
        },
        "feature_stats": feature_stats,
        "row_sum": row_sum,
        "results": results,
        "tables": {
            "feature_profile": numify(read_csv_rows(tab / "02_feature_profile.csv")),
            "selection_ranking": numify(read_csv_rows(tab / "03_feature_selection_ranking.csv")),
            "k_sweep": numify(read_csv_rows(tab / "04_feature_selection_k_sweep.csv")),
            "model_comparison": numify(read_csv_rows(tab / "05_model_comparison.csv")),
            "coefficients": ref,
            "vif": numify(read_csv_rows(tab / "07_vif.csv")),
            "robust_se": numify(read_csv_rows(tab / "09_robust_standard_errors.csv")),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), default=float)

    kb = OUT.stat().st_size / 1024
    print(f"  wrote {OUT.relative_to(ROOT)}  ({kb:.0f} KB)")
    print(f"  terms={len(terms)}  n_train={len(y_tr)}  sigma={ols.scale ** 0.5:.6f}")


if __name__ == "__main__":
    main()
