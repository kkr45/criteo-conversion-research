#!/usr/bin/env python3
"""Follow-up: defensible-arm (D) calibration + sensitivity.

milestone2_run.py calibrates only Arm B (price-contaminated) and runs its
rolling-window / gap sweeps on B. RQ2/RQ3 need the same checks on the
defensible arm D (B minus product_price). Reuses milestone2_run.py's
loader/splitter/fitters (seed 530, 500k sample, chronological 60/20/20).
Outputs: outputs/d_defensible_sensitivity.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import milestone2_run as m2

DATA = "/home/hatch/workspace/anly530/ka1-milestone2/data/CriteoSearchData"
OUT = Path("/home/hatch/workspace/anly530/ka1-milestone2/outputs")

t0 = time.time()
df, scanned, delim, header = m2.load_sample(DATA, 500_000)
parts, bounds = m2.chronological_parts(df)
y = {k: v["Sale"].astype(int).to_numpy() for k, v in parts.items()}
out = {"splits": bounds}

# ---- D logistic + D GBM (same as main run, refit for probabilities) ----
X = {k: m2.make_logistic_xy(v, m2.D_NUMERIC, m2.D_CATEGORICAL) for k, v in parts.items()}
d_log = m2.fit_logistic(X["train"], y["train"], m2.D_NUMERIC, m2.D_CATEGORICAL)
p_dlog_test = d_log.predict_proba(X["test"])[:, 1]
bundle = m2.fit_gbm(parts["train"], y["train"], m2.D_NUMERIC, m2.D_CATEGORICAL)
p_dgbm_test = m2.gbm_predict(bundle, parts["test"])
print(f"D logistic test AP={m2.score(y['test'], p_dlog_test)['AP']:.4f}; "
      f"D GBM test AP={m2.score(y['test'], p_dgbm_test)['AP']:.4f}", flush=True)

calib = {}
for label, p in [("D_logistic", p_dlog_test), ("D_gbm", p_dgbm_test)]:
    rows, ece, slope, intercept = m2.calibration_table(y["test"], p)
    calib[label] = {"bins": rows, "ECE": round(ece, 4),
                    "slope": round(slope, 3), "intercept": round(intercept, 3)}
    print(f"calibration {label}: ECE={ece:.4f} slope={slope:.3f} "
          f"intercept={intercept:.3f}", flush=True)
out["calibration_test_D"] = calib

# ---- D rolling windows (same windows as main run, B -> D) ---------------
ordered = (df.assign(_t=pd.to_datetime(
    pd.to_numeric(df["click_timestamp"], errors="coerce").where(
        pd.to_numeric(df["click_timestamp"], errors="coerce").gt(0)),
    unit="s", utc=True, errors="coerce")).dropna(subset=["_t"])
    .sort_values("_t", kind="stable"))
n = len(ordered)
windows = {}
for wname, (tr_s, tr_e, te_s, te_e) in {
        "W1_train_first40_test_last20": (0.0, 0.4, 0.8, 1.0),
        "W2_train_20to60_test_last20": (0.2, 0.6, 0.8, 1.0)}.items():
    tr = ordered.iloc[int(n * tr_s):int(n * tr_e)]
    te = ordered.iloc[int(n * te_s):int(n * te_e)]
    Xtr = m2.make_logistic_xy(tr, m2.D_NUMERIC, m2.D_CATEGORICAL)
    Xte = m2.make_logistic_xy(te, m2.D_NUMERIC, m2.D_CATEGORICAL)
    m = m2.fit_logistic(Xtr, tr["Sale"].astype(int).to_numpy(),
                        m2.D_NUMERIC, m2.D_CATEGORICAL)
    p = m.predict_proba(Xte)[:, 1]
    windows[wname] = {
        "train_rows": int(len(tr)), "test_rows": int(len(te)),
        "train_range": f"{tr['_t'].min()} .. {tr['_t'].max()}",
        "test_range": f"{te['_t'].min()} .. {te['_t'].max()}",
        **m2.score(te["Sale"].astype(int).to_numpy(), p)}
    print(f"{wname} (D): test AP={windows[wname]['AP']:.4f}", flush=True)
out["rolling_windows_D_logistic"] = windows

# ---- D label-gap sweep (0/7/14/30d, same rule as main run) --------------
valid_start = ordered["_t"].iloc[int(n * 0.6)]
test_start = ordered["_t"].iloc[int(n * 0.8)]
gaps = {}
for gap_days in (0, 7, 14, 30):
    gap = pd.Timedelta(days=gap_days)
    tr = ordered.loc[ordered["_t"] < valid_start - gap]
    va = ordered.loc[(ordered["_t"] >= valid_start)
                     & (ordered["_t"] < test_start - gap)]
    te = ordered.loc[ordered["_t"] >= test_start]
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        gaps[str(gap_days)] = {"feasible": False,
                               "train_rows": int(len(tr)),
                               "validation_rows": int(len(va)),
                               "test_rows": int(len(te)),
                               "note": "A partition is empty at this gap "
                                       "under 60/20/20 splits on 91 days of logs."}
        print(f"gap {gap_days}d (D): INFEASIBLE", flush=True)
        continue
    Xtr = m2.make_logistic_xy(tr, m2.D_NUMERIC, m2.D_CATEGORICAL)
    Xva = m2.make_logistic_xy(va, m2.D_NUMERIC, m2.D_CATEGORICAL)
    Xte = m2.make_logistic_xy(te, m2.D_NUMERIC, m2.D_CATEGORICAL)
    m = m2.fit_logistic(Xtr, tr["Sale"].astype(int).to_numpy(),
                        m2.D_NUMERIC, m2.D_CATEGORICAL)
    gaps[str(gap_days)] = {
        "feasible": True,
        "train_rows": int(len(tr)), "validation_rows": int(len(va)),
        "test_rows": int(len(te)),
        "excluded_rows": int(n - len(tr) - len(va) - len(te)),
        "validation": m2.score(va["Sale"].astype(int).to_numpy(),
                               m.predict_proba(Xva)[:, 1]),
        "test": m2.score(te["Sale"].astype(int).to_numpy(),
                        m.predict_proba(Xte)[:, 1])}
    print(f"gap {gap_days}d (D): val AP={gaps[str(gap_days)]['validation']['AP']:.4f} "
          f"test AP={gaps[str(gap_days)]['test']['AP']:.4f}", flush=True)
out["label_gap_sweep_D_logistic"] = gaps
out["runtime_seconds"] = round(time.time() - t0, 1)
(OUT / "d_defensible_sensitivity.json").write_text(
    json.dumps(out, indent=2, default=str))
print(f"DONE in {out['runtime_seconds']}s; wrote d_defensible_sensitivity.json",
      flush=True)
