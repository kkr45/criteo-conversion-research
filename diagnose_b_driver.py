#!/usr/bin/env python3
"""Diagnostic: what drives Arm B logistic test AP ~0.955?

Reuses milestone2_run.py's loader/splitter (seed 530, 500k sample,
chronological 60/20/20). Does NOT write the Milestone 2 report numbers;
it only identifies the driving feature(s) so the report can interpret
B honestly. Run: venv python diagnose_b_driver.py
Outputs: outputs/diag_b_driver.json + printed tables.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import milestone2_run as m2

DATA = "/home/hatch/workspace/anly530/ka1-milestone2/data/CriteoSearchData"
OUT = Path("/home/hatch/workspace/anly530/ka1-milestone2/outputs")
OUT.mkdir(parents=True, exist_ok=True)

df, scanned, delim, header = m2.load_sample(DATA, 500_000)
print(f"scanned={scanned:,} retained={len(df):,}", flush=True)
parts, bounds = m2.chronological_parts(df)
y = {k: v["Sale"].astype(int).to_numpy() for k, v in parts.items()}
tr, va, te = parts["train"], parts["validation"], parts["test"]

out = {"scanned": scanned, "splits": bounds}

# ---- 1. Sentinel (-1 / missing) rate by Sale, per column (train) -------
sent = {}
for col in m2.CRITEO_COLUMNS:
    s = tr[col].astype(str).str.strip()
    is_miss = s.isin(["-1", "-1.0", ""])
    sent[col] = {
        "miss_rate_Sale0": round(float(is_miss[y["train"] == 0].mean()), 4),
        "miss_rate_Sale1": round(float(is_miss[y["train"] == 1].mean()), 4),
    }
out["sentinel_by_sale_train"] = sent
print("\nSentinel rate by Sale (train) — biggest gaps:", flush=True)
gaps = sorted(sent.items(), key=lambda kv: -abs(kv[1]["miss_rate_Sale1"] - kv[1]["miss_rate_Sale0"]))
for col, v in gaps[:8]:
    print(f"  {col}: Sale0 miss={v['miss_rate_Sale0']} Sale1 miss={v['miss_rate_Sale1']}", flush=True)

# ---- 2. Numeric columns by Sale ----------------------------------------
numsum = {}
for col in ["product_price", "nb_clicks_1week", "SalesAmountInEuro",
            "time_delay_for_conversion"]:
    v = pd.to_numeric(tr[col], errors="coerce")
    numsum[col] = {
        "mean_Sale0": round(float(v[y["train"] == 0].mean()), 4),
        "mean_Sale1": round(float(v[y["train"] == 1].mean()), 4),
        "frac_eq_-1_Sale0": round(float((v[y["train"] == 0] == -1).mean()), 4),
        "frac_eq_-1_Sale1": round(float((v[y["train"] == 1].mean()), 4) if False else float((v[y["train"] == 1] == -1).mean()), 4),
        "frac_eq_0_Sale0": round(float((v[y["train"] == 0] == 0).mean()), 4),
        "frac_eq_0_Sale1": round(float((v[y["train"] == 1] == 0).mean()), 4),
    }
out["numeric_by_sale_train"] = numsum
print("\nNumeric by Sale (train):", flush=True)
print(json.dumps(numsum, indent=2), flush=True)

# ---- 3. Single-column target-rate score (train-fitted, tested on test) --
# For each categorical column: score(test row) = smoothed train conversion
# rate of its value. AP of that score isolates one column's signal.
from sklearn.metrics import average_precision_score

def target_rate_ap(col):
    tr_rate = tr.groupby(tr[col].astype(str))["Sale"].agg(["mean", "count"])
    prior = float(y["train"].mean())
    # smoothing: (pos + 20*prior) / (n + 20)
    rate = (tr_rate["mean"] * tr_rate["count"] + 20 * prior) / (tr_rate["count"] + 20)
    s = te[col].astype(str).map(rate).fillna(prior).to_numpy()
    return float(average_precision_score(y["test"], s))

single = {}
for col in m2.B_CATEGORICAL:
    single[col] = round(target_rate_ap(col), 4)
# numeric single-column: raw value as score (and -value)
for col in ["product_price", "nb_clicks_1week"]:
    v = pd.to_numeric(te[col], errors="coerce").fillna(-1).to_numpy()
    single[col + " (raw)"] = round(float(average_precision_score(y["test"], v)), 4)
    single[col + " (neg raw)"] = round(float(average_precision_score(y["test"], -v)), 4)
out["single_column_test_AP"] = single
print("\nSingle-column test AP (train-fitted target rate):", flush=True)
for col, v in sorted(single.items(), key=lambda kv: -kv[1]):
    print(f"  {col}: {v}", flush=True)

# ---- 4. Group ablations: B logistic, one group at a time ----------------
def run_arm(name, num, cat):
    X = {k: m2.make_logistic_xy(v, num, cat) for k, v in parts.items()}
    model = m2.fit_logistic(X["train"], y["train"], num, cat)
    res = {}
    for split in ("validation", "test"):
        res[split] = m2.score(y[split], model.predict_proba(X[split])[:, 1])
    print(f"  {name}: val AP={res['validation']['AP']:.4f} "
          f"test AP={res['test']['AP']:.4f} test ROC-AUC={res['test']['ROC_AUC']:.4f}",
          flush=True)
    return res

groups = {
    "numeric_only": (m2.B_NUMERIC, []),
    "categorical_only": ([], m2.B_CATEGORICAL),
}
for col in m2.B_CATEGORICAL:
    groups[f"only_{col}"] = ([], [col])
abl = {}
print("\nGroup models (B logistic, same splits):", flush=True)
for name, (num, cat) in groups.items():
    abl[name] = run_arm(name, num, cat)
print("\nLeave-one-group-out (B minus group):", flush=True)
loo = {}
for col in m2.B_CATEGORICAL:
    loo[f"B_minus_{col}"] = run_arm(f"B_minus_{col}", m2.B_NUMERIC,
                                    [c for c in m2.B_CATEGORICAL if c != col])
loo["B_minus_numeric"] = run_arm("B_minus_numeric", [], m2.B_CATEGORICAL)
abl.update(loo)
out["group_ablations"] = {k: {s: {m: round(vv, 4) for m, vv in d.items() if isinstance(vv, float)} for s, d in v.items()} for k, v in abl.items()}

# ---- 5. Memorisation check: test rows whose exact B-combo seen in train --
combo_cols = m2.B_CATEGORICAL
tr_keys = set(map(tuple, tr[combo_cols].astype(str).values))
te_keys = list(map(tuple, te[combo_cols].astype(str).values))
seen = np.array([k in tr_keys for k in te_keys])
out["combo_overlap"] = {
    "test_rows_with_exact_B_combo_seen_in_train": int(seen.sum()),
    "test_rows": int(len(te)),
    "frac_seen": round(float(seen.mean()), 4),
    "test_prevalence_seen": round(float(y["test"][seen].mean()), 4) if seen.sum() else None,
    "test_prevalence_unseen": round(float(y["test"][~seen].mean()), 4) if (~seen).sum() else None,
}
print("\nExact B-combo overlap:", json.dumps(out["combo_overlap"], indent=2), flush=True)

# B-logistic test AP on seen vs unseen subsets (refit B once more)
X = {k: m2.make_logistic_xy(v, m2.B_NUMERIC, m2.B_CATEGORICAL) for k, v in parts.items()}
model = m2.fit_logistic(X["train"], y["train"], m2.B_NUMERIC, m2.B_CATEGORICAL)
p_te = model.predict_proba(X["test"])[:, 1]
for label, mask in [("seen", seen), ("unseen", ~seen)]:
    if mask.sum() > 100 and y["test"][mask].sum() > 0:
        ap = float(average_precision_score(y["test"][mask], p_te[mask]))
        print(f"  B-logistic test AP on {label} subset (n={mask.sum():,}): {ap:.4f}", flush=True)
        out[f"B_logistic_test_AP_{label}_subset"] = round(ap, 4)

(OUT / "diag_b_driver.json").write_text(json.dumps(out, indent=2, default=str))
print("\nWrote outputs/diag_b_driver.json", flush=True)
