#!/usr/bin/env python3
"""ANLY 530 KA1 Milestone 2 — Research Execution and Preliminary Findings.

Leakage-aware purchase-conversion prediction in sponsored search (Criteo
Sponsored Search Conversion Log).

STATUS: EXECUTED 2026-11-23 on the real Criteo file (15,995,634 rows
scanned, 500,000-row sample, seed 530; 452 s). Headline test AP:
A 0.9962, B 0.9553, C 0.9552, D 0.1810, E 0.1809; GBM B 0.9550,
C 0.9554, D 0.1866. The B-driver diagnostic (diagnose_b_driver.py)
showed product_price alone gives test AP 0.9401 — it is 0.0 for every
non-conversion — so D (B minus product_price) is the honest baseline.
Run once the file is present:

    ~/workspace/anly530/week11/.venv/bin/python milestone2_run.py \
        --data /path/to/CriteoSearchData \
        --out  ~/workspace/anly530/ka1-milestone2/outputs

Design (from the approved Milestone 1 protocol, Week 10, scored 100/100):
  * Uniform bounded random sample, seed 530, max 500,000 rows, scanned in
    50,000-row chunks; never loads the full ~16M-row file into memory.
  * Chronological 60/20/20 split by click_timestamp (train / validation /
    test). Model selection on validation only; test evaluated once, at the end.
  * Arm A — leakage-prone diagnostic (B features + SalesAmountInEuro +
    time_delay_for_conversion as raw numerics). Labelled INVALID for
    deployment; included only to quantify the inflation.
  * Arm B — Milestone 1 baseline: numeric product_price, nb_clicks_1week;
    categorical product_age_group, device_type, product_gender,
    product_brand, product_category1..7, product_country. IDs
    (user_id, product_id, product_title, audience_id, partner_id) excluded.
  * Arm C — conservative subset: B minus nb_clicks_1week (uncertain timing;
    audience_id already excluded in B).
  * Arms D/E (added 2026-11-23): D is B minus product_price — the honest
    defensible baseline, after the B-driver diagnostic showed product_price
    alone gives test AP 0.9401 because it is 0.0 for every non-conversion;
    E is categorical features only. GBM is run on B, C and D.
  * Models: L2 logistic regression (C=1.0, liblinear, max_iter=500) for
    A/B/C/D/E; HistGradientBoosting (native categorical handling) for B, C
    and D.
  * Metrics: average precision (scikit-learn, non-interpolated, primary),
    ROC-AUC, log loss, Brier score; constant (train-prevalence) comparator.
  * Calibration: 10-bin reliability table + ECE + slope/intercept for
    B-logistic vs B-GBM on test.
  * Sensitivity: rolling chronological windows + label-gap sweep (0/7/14
    days). A full 30-day embargo is tested and reported as feasible or
    infeasible — with 91 days of logs under 60/20/20 splits it empties the
    validation partition, which is itself a finding (see run log).

Outputs (all aggregates only; no row-level data is written):
  outputs/metrics_milestone2.json, outputs/run_log.txt,
  outputs/pr_curves_test.csv, outputs/calibration_test.csv,
  outputs/fig1_pr_curves.png, outputs/fig2_reliability.png,
  outputs/fig3_test_ap_bars.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CRITEO_COLUMNS = [
    "Sale", "SalesAmountInEuro", "time_delay_for_conversion", "click_timestamp",
    "nb_clicks_1week", "product_price", "product_age_group", "device_type",
    "audience_id", "product_gender", "product_brand", "product_category1",
    "product_category2", "product_category3", "product_category4",
    "product_category5", "product_category6", "product_category7",
    "product_country", "product_id", "product_title", "partner_id", "user_id",
]
OUTCOME_DERIVED = ["SalesAmountInEuro", "time_delay_for_conversion"]
B_NUMERIC = ["product_price", "nb_clicks_1week"]
B_CATEGORICAL = [
    "product_age_group", "device_type", "product_gender", "product_brand",
    "product_category1", "product_category2", "product_category3",
    "product_category4", "product_category5", "product_category6",
    "product_category7", "product_country",
]
C_NUMERIC = ["product_price"]          # B minus nb_clicks_1week
C_CATEGORICAL = list(B_CATEGORICAL)
# Corrected arms (added 2026-11-23 after diagnose_b_driver.py showed
# product_price alone gives test AP 0.9401: it is 0.0 for EVERY
# non-conversion, i.e. a third outcome-derived field Milestone 1 missed.
# D = B minus product_price (the honest defensible baseline);
# E = categorical features only (D minus nb_clicks_1week).
D_NUMERIC = ["nb_clicks_1week"]
D_CATEGORICAL = list(B_CATEGORICAL)
E_NUMERIC = []
E_CATEGORICAL = list(B_CATEGORICAL)
SEED = 530


def log(msg, fh):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def detect_format(path):
    import csv
    with open(path, "r", encoding="utf-8-sig", errors="replace") as h:
        first = h.readline().rstrip("\r\n")
    if not first:
        raise ValueError("Dataset file is empty.")
    delim = "\t" if first.count("\t") >= first.count(",") else ","
    fields = next(csv.reader([first], delimiter=delim))
    if len(fields) != 23:
        raise ValueError(f"Expected 23 fields, found {len(fields)}.")
    header = fields[0].strip().lower() == "sale"
    if not header and fields[0].strip() not in {"0", "1"}:
        raise ValueError("First column is not Sale=0/1; check DATA_PATH.")
    return delim, header


def load_sample(path, sample_rows=500_000, chunk_size=50_000, seed=SEED):
    """Uniform bounded random sample by independent random priorities."""
    delim, header = detect_format(path)
    kwargs = dict(sep=delim, header=0 if header else None,
                  chunksize=chunk_size, low_memory=False,
                  dtype=str, encoding="utf-8-sig")
    if not header:
        kwargs["names"] = CRITEO_COLUMNS
    rng = np.random.default_rng(seed)
    sample = pd.DataFrame()
    keys = np.empty(0, dtype=np.float64)
    scanned = 0
    for chunk in pd.read_csv(path, **kwargs):
        chunk.columns = chunk.columns.astype(str).str.strip()
        if chunk.columns.tolist() != CRITEO_COLUMNS:
            raise ValueError(f"Unexpected columns: {chunk.columns.tolist()}")
        scanned += len(chunk)
        new_keys = rng.random(len(chunk))
        cand = pd.concat((sample, chunk), ignore_index=True)
        cand_keys = np.concatenate((keys, new_keys))
        if len(cand) > sample_rows:
            idx = np.argpartition(cand_keys, sample_rows - 1)[:sample_rows]
            sample, keys = cand.iloc[idx].reset_index(drop=True), cand_keys[idx]
        else:
            sample, keys = cand, cand_keys
    if scanned == 0:
        raise ValueError("No data rows found.")
    sample["Sale"] = pd.to_numeric(sample["Sale"], errors="raise")
    sample["click_timestamp"] = pd.to_numeric(sample["click_timestamp"],
                                              errors="raise")
    assert sample["Sale"].isin((0, 1)).all()
    return sample, scanned, delim, header


def audit(df, fh):
    """RQ1 data-quality audit on the retained sample (aggregates only)."""
    from collections import OrderedDict
    y = df["Sale"].astype("int8")
    out = OrderedDict()
    out["sample_rows"] = int(len(df))
    out["conversions"] = int(y.sum())
    out["prevalence"] = float(y.mean())
    out["exact_duplicate_rows"] = int(df.duplicated().sum())
    cols = []
    for col in df.columns:
        s = df[col].astype("string").str.strip()
        cols.append({
            "column": col,
            "pandas_na_pct": round(100 * float(df[col].isna().mean()), 2),
            "minus_one_sentinel_pct": round(
                100 * float(s.eq("-1").fillna(False).mean()), 2),
            "distinct_values": int(df[col].nunique(dropna=True)),
        })
    out["columns"] = cols
    # Leakage proof: sentinel rate of outcome-derived fields by Sale label.
    leak = []
    for col in OUTCOME_DERIVED:
        num = pd.to_numeric(df[col], errors="coerce")
        for label in (0, 1):
            sub = num[y.eq(label)]
            leak.append({"field": col, "Sale": label, "n": int(len(sub)),
                         "minus_one_pct": round(100 * float(sub.eq(-1).mean()), 2),
                         "non_sentinel_pct": round(
                             100 * float((sub.notna() & sub.ne(-1)).mean()), 2)})
    out["leakage_audit"] = leak
    # product_price audit (added 2026-11-23 after the B-driver diagnostic):
    # price is exactly 0.0 for every non-conversion in the sample, so it
    # behaves as a third outcome-derived field; see diagnose_b_driver.py.
    price = pd.to_numeric(df["product_price"], errors="coerce")
    amount = pd.to_numeric(df["SalesAmountInEuro"], errors="coerce")
    pa = {"price_eq_0_pct_by_Sale": {},
          "price_gt_0_pct_by_Sale": {}}
    for label in (0, 1):
        sub = price[y.eq(label)]
        pa["price_eq_0_pct_by_Sale"][str(label)] = round(
            100 * float(sub.eq(0).mean()), 2)
        pa["price_gt_0_pct_by_Sale"][str(label)] = round(
            100 * float(sub.gt(0).mean()), 2)
    conv = y.eq(1)
    pa["conversions_with_price_eq_salesamount_pct"] = round(
        100 * float((price[conv] == amount[conv]).mean()), 2)
    pa["conversion_rate_among_price_gt_0_pct"] = round(
        100 * float(y[price.gt(0)].mean()), 2)
    pa["conversion_rate_among_price_eq_0_pct"] = round(
        100 * float(y[price.eq(0)].mean()), 2)
    out["product_price_audit"] = pa
    ts = pd.to_numeric(df["click_timestamp"], errors="coerce")
    valid = ts.notna() & ts.gt(0)
    out["invalid_timestamps"] = int((~valid).sum())
    ct = pd.to_datetime(ts[valid], unit="s", utc=True)
    out["first_click_utc"] = str(ct.min())
    out["last_click_utc"] = str(ct.max())
    out["coverage_days"] = float((ct.max() - ct.min()).total_seconds() / 86400)
    weekly = (df.loc[valid].assign(_t=ct)
              .groupby(pd.Grouper(key="_t", freq="7D"))
              .agg(clicks=("Sale", "size"), conversions=("Sale", "sum")))
    weekly["rate"] = (weekly["conversions"] / weekly["clicks"]).round(4)
    out["weekly_prevalence"] = [
        {"week_start_utc": str(i), "clicks": int(r.clicks),
         "conversions": int(r.conversions), "rate": float(r.rate)}
        for i, r in weekly.iterrows()]
    return out


def chronological_parts(df):
    """Sort by click time; return train/valid/test (60/20/20 by row count)."""
    ts = pd.to_numeric(df["click_timestamp"], errors="coerce")
    ct = pd.to_datetime(ts.where(ts.gt(0)), unit="s", utc=True, errors="coerce")
    keep = ct.notna()
    ordered = (df.loc[keep].assign(_click_time=ct[keep])
               .sort_values("_click_time", kind="stable"))
    n = len(ordered)
    i, j = int(n * 0.6), int(n * 0.8)
    parts = {"train": ordered.iloc[:i], "validation": ordered.iloc[i:j],
             "test": ordered.iloc[j:]}
    bounds = {k: {"rows": int(len(v)),
                  "first_utc": str(v["_click_time"].min()),
                  "last_utc": str(v["_click_time"].max()),
                  "prevalence": float(v["Sale"].mean())}
              for k, v in parts.items()}
    return parts, bounds


def make_logistic_xy(part, numeric, categorical, leaky_numeric=()):
    """Feature frame for logistic regression (numeric + one-hot-ready cats)."""
    X = part[list(numeric) + list(categorical) + list(leaky_numeric)].copy()
    for col in list(numeric):
        v = pd.to_numeric(X[col], errors="coerce")
        X[col] = v.mask(v.eq(-1) | ~np.isfinite(v))
    for col in list(leaky_numeric):  # diagnostic only: raw, sentinel kept
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical:
        v = X[col].astype("string").str.strip()
        X[col] = (v.mask(v.isin(["-1", "-1.0", ""]))
                  .fillna("__MISSING__").astype(str))
    return X


def fit_logistic(Xtr, ytr, numeric, categorical, leaky_numeric=()):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num_cols = list(numeric) + list(leaky_numeric)
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median",
                                                   keep_empty_features=True)),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20),
         list(categorical)),
    ], sparse_threshold=1.0)
    model = Pipeline([("preprocess", pre),
                      ("clf", LogisticRegression(max_iter=500,
                                                 solver="liblinear",
                                                 random_state=SEED))])
    model.fit(Xtr, ytr)
    return model


def fit_gbm(Xtr_df, ytr, numeric, categorical):
    """HistGradientBoosting with native categorical handling (ordinal codes).

    FIX (2026-11-23): HistGradientBoosting rejects categorical features
    with cardinality > max_bins (default 255); device_type alone has
    ~11,454 distinct hashed values in train. Rare values are therefore
    grouped per column into "__OTHER__" (keep the 200 most frequent
    train values per column, mapping fitted on train only), which also
    matches the logistic pipeline's min_frequency=20 one-hot treatment
    of rare categories. The keep-sets are stored in the bundle so
    gbm_predict applies the identical mapping.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    X = Xtr_df.copy()
    for col in numeric:
        v = pd.to_numeric(X[col], errors="coerce")
        X[col] = v.mask(v.eq(-1) | ~np.isfinite(v))
    num_imp = SimpleImputer(strategy="median")
    X[numeric] = num_imp.fit_transform(X[numeric])
    keep_sets = {}
    for col in categorical:
        v = X[col].astype("string").str.strip()
        v = (v.mask(v.isin(["-1", "-1.0", ""]))
             .fillna("__MISSING__").astype(str))
        keep = set(v.value_counts().head(200).index)
        keep_sets[col] = keep
        X[col] = v.where(v.isin(keep), "__OTHER__")
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         dtype=np.int64)
    X[categorical] = enc.fit_transform(X[categorical])
    cat_mask = [False] * len(numeric) + [True] * len(categorical)
    cols = list(numeric) + list(categorical)
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, random_state=SEED,
        categorical_features=cat_mask)
    model.fit(X[cols].to_numpy(), ytr)
    return {"model": model, "num_imputer": num_imp, "encoder": enc,
            "columns": cols, "cat_mask": cat_mask,
            "numeric": list(numeric), "categorical": list(categorical),
            "keep_sets": {k: sorted(v) for k, v in keep_sets.items()}}


def gbm_predict(bundle, part):
    X = part[list(bundle["numeric"]) + list(bundle["categorical"])].copy()
    for col in bundle["numeric"]:
        v = pd.to_numeric(X[col], errors="coerce")
        X[col] = v.mask(v.eq(-1) | ~np.isfinite(v))
    X[bundle["numeric"]] = bundle["num_imputer"].transform(X[bundle["numeric"]])
    keep_sets = bundle.get("keep_sets", {})
    for col in bundle["categorical"]:
        v = X[col].astype("string").str.strip()
        v = (v.mask(v.isin(["-1", "-1.0", ""]))
             .fillna("__MISSING__").astype(str))
        if col in keep_sets:
            keep = set(keep_sets[col])
            v = v.where(v.isin(keep), "__OTHER__")
    X[bundle["categorical"]] = bundle["encoder"].transform(X[bundle["categorical"]])
    return bundle["model"].predict_proba(
        X[bundle["columns"]].to_numpy())[:, 1]


def score(y, p):
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 log_loss, roc_auc_score)
    return {"AP": float(average_precision_score(y, p)),
            "ROC_AUC": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "Brier": float(brier_score_loss(y, p)),
            "prevalence": float(np.mean(y)), "rows": int(len(y))}


def calibration_table(y, p, n_bins=10):
    """Equal-width bins on predicted probability; ECE + slope/intercept."""
    from sklearn.linear_model import LogisticRegression as LR
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < n_bins - 1
                               else p <= edges[b + 1])
        if m.sum() == 0:
            rows.append({"bin": b, "n": 0, "mean_pred": None,
                         "frac_pos": None})
            continue
        mp, fp = float(p[m].mean()), float(y[m].mean())
        ece += (m.sum() / len(y)) * abs(mp - fp)
        rows.append({"bin": b, "n": int(m.sum()), "mean_pred": round(mp, 4),
                     "frac_pos": round(fp, 4)})
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(pc / (1 - pc)).reshape(-1, 1)
    cal = LR(C=1e6, solver="lbfgs").fit(logit, y)
    return rows, float(ece), float(cal.coef_[0][0]), float(cal.intercept_[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-rows", type=int, default=500_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import sklearn, matplotlib
    import platform
    fh = open(out / "run_log.txt", "w")
    t0 = time.time()
    log(f"DATA_PATH={args.data}", fh)
    log(f"sample_rows={args.sample_rows} seed={SEED}", fh)
    log(f"python={platform.python_version()} pandas={pd.__version__} "
        f"numpy={np.__version__} sklearn={sklearn.__version__} "
        f"matplotlib={matplotlib.__version__}", fh)

    df, scanned, delim, header = load_sample(args.data, args.sample_rows)
    log(f"separator={delim!r} header={header} scanned={scanned:,} "
        f"retained={len(df):,}", fh)
    audit_out = audit(df, fh)
    log(f"audit: prevalence={audit_out['prevalence']:.4f} "
        f"duplicates={audit_out['exact_duplicate_rows']} "
        f"invalid_ts={audit_out['invalid_timestamps']} "
        f"range={audit_out['first_click_utc']} .. {audit_out['last_click_utc']}",
        fh)

    parts, bounds = chronological_parts(df)
    log(f"splits: {json.dumps(bounds)}", fh)
    y = {k: v["Sale"].astype(int).to_numpy() for k, v in parts.items()}
    train_prev = float(y["train"].mean())

    # ---- Arms A / B / C (logistic) -------------------------------------
    arms = {
        "A_leakage_diagnostic_logistic": (B_NUMERIC, B_CATEGORICAL,
                                          OUTCOME_DERIVED),
        "B_baseline_logistic": (B_NUMERIC, B_CATEGORICAL, ()),
        "C_conservative_logistic": (C_NUMERIC, C_CATEGORICAL, ()),
        "D_defensible_logistic": (D_NUMERIC, D_CATEGORICAL, ()),
        "E_categorical_only_logistic": (E_NUMERIC, E_CATEGORICAL, ()),
    }
    results = {}
    pr_curves = {}
    for name, (num, cat, leaky) in arms.items():
        X = {k: make_logistic_xy(v, num, cat, leaky)
             for k, v in parts.items()}
        model = fit_logistic(X["train"], y["train"], num, cat, leaky)
        results[name] = {}
        for split in ("validation", "test"):
            p = model.predict_proba(X[split])[:, 1]
            results[name][split] = score(y[split], p)
            if split == "test":
                from sklearn.metrics import precision_recall_curve
                prec, rec, _ = precision_recall_curve(y[split], p)
                pr_curves[name] = (rec, prec)
        log(f"{name}: val AP={results[name]['validation']['AP']:.4f} "
            f"test AP={results[name]['test']['AP']:.4f}", fh)

    # Constant comparator (train prevalence) on validation and test.
    for split in ("validation", "test"):
        p_const = np.full(len(y[split]), train_prev)
        results.setdefault("constant_train_prevalence", {})[split] = score(
            y[split], p_const)

    # ---- HistGradientBoosting on B and C --------------------------------
    gbm_bundles, gbm_probs = {}, {}
    for name, (num, cat) in {"B_gbm": (B_NUMERIC, B_CATEGORICAL),
                             "C_gbm": (C_NUMERIC, C_CATEGORICAL),
                             "D_gbm": (D_NUMERIC, D_CATEGORICAL)}.items():
        bundle = fit_gbm(parts["train"], y["train"], num, cat)
        gbm_bundles[name] = bundle
        results[name] = {}
        for split in ("validation", "test"):
            p = gbm_predict(bundle, parts[split])
            gbm_probs[(name, split)] = p
            results[name][split] = score(y[split], p)
            if split == "test":
                from sklearn.metrics import precision_recall_curve
                prec, rec, _ = precision_recall_curve(y[split], p)
                pr_curves[name] = (rec, prec)
        log(f"{name}: val AP={results[name]['validation']['AP']:.4f} "
            f"test AP={results[name]['test']['AP']:.4f}", fh)

    # ---- Calibration (test): B logistic vs B GBM ------------------------
    # Recompute B-logistic test probabilities for the calibration table.
    Xb_test = make_logistic_xy(parts["test"], B_NUMERIC, B_CATEGORICAL)
    Xb_train = make_logistic_xy(parts["train"], B_NUMERIC, B_CATEGORICAL)
    blog = fit_logistic(Xb_train, y["train"], B_NUMERIC, B_CATEGORICAL)
    p_blog_test = blog.predict_proba(Xb_test)[:, 1]
    calib = {}
    for label, p in [("B_logistic", p_blog_test),
                     ("B_gbm", gbm_probs[("B_gbm", "test")])]:
        rows, ece, slope, intercept = calibration_table(y["test"], p)
        calib[label] = {"bins": rows, "ECE": round(ece, 4),
                        "slope": round(slope, 3),
                        "intercept": round(intercept, 3)}
        log(f"calibration {label}: ECE={ece:.4f} slope={slope:.3f} "
            f"intercept={intercept:.3f}", fh)

    # ---- Sensitivity 1: rolling chronological windows (B logistic) ------
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
        Xtr = make_logistic_xy(tr, B_NUMERIC, B_CATEGORICAL)
        Xte = make_logistic_xy(te, B_NUMERIC, B_CATEGORICAL)
        m = fit_logistic(Xtr, tr["Sale"].astype(int).to_numpy(),
                         B_NUMERIC, B_CATEGORICAL)
        p = m.predict_proba(Xte)[:, 1]
        windows[wname] = {
            "train_rows": int(len(tr)), "test_rows": int(len(te)),
            "train_range": f"{tr['_t'].min()} .. {tr['_t'].max()}",
            "test_range": f"{te['_t'].min()} .. {te['_t'].max()}",
            **score(te["Sale"].astype(int).to_numpy(), p)}
        log(f"{wname}: test AP={windows[wname]['AP']:.4f}", fh)

    # ---- Sensitivity 2: label-gap sweep (B logistic) --------------------
    # Notebook-02 rule: train < valid_start - gap; valid in
    # [valid_start, test_start - gap); test >= test_start.
    gaps = {}
    valid_start = ordered["_t"].iloc[int(n * 0.6)]
    test_start = ordered["_t"].iloc[int(n * 0.8)]
    for gap_days in (0, 7, 14, 30):
        gap = pd.Timedelta(days=gap_days)
        tr = ordered.loc[ordered["_t"] < valid_start - gap]
        va = ordered.loc[(ordered["_t"] >= valid_start)
                         & (ordered["_t"] < test_start - gap)]
        te = ordered.loc[ordered["_t"] >= test_start]
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            gaps[str(gap_days)] = {
                "feasible": False,
                "train_rows": int(len(tr)), "validation_rows": int(len(va)),
                "test_rows": int(len(te)),
                "note": "A partition is empty at this gap under 60/20/20 "
                        "splits on 91 days of logs."}
            log(f"gap {gap_days}d: INFEASIBLE "
                f"(train={len(tr)} valid={len(va)} test={len(te)})", fh)
            continue
        Xtr = make_logistic_xy(tr, B_NUMERIC, B_CATEGORICAL)
        Xva = make_logistic_xy(va, B_NUMERIC, B_CATEGORICAL)
        Xte = make_logistic_xy(te, B_NUMERIC, B_CATEGORICAL)
        m = fit_logistic(Xtr, tr["Sale"].astype(int).to_numpy(),
                         B_NUMERIC, B_CATEGORICAL)
        gaps[str(gap_days)] = {
            "feasible": True,
            "train_rows": int(len(tr)), "validation_rows": int(len(va)),
            "test_rows": int(len(te)),
            "excluded_rows": int(n - len(tr) - len(va) - len(te)),
            "validation": score(va["Sale"].astype(int).to_numpy(),
                                m.predict_proba(Xva)[:, 1]),
            "test": score(te["Sale"].astype(int).to_numpy(),
                          m.predict_proba(Xte)[:, 1])}
        log(f"gap {gap_days}d: val AP="
            f"{gaps[str(gap_days)]['validation']['AP']:.4f} test AP="
            f"{gaps[str(gap_days)]['test']['AP']:.4f}", fh)

    # ---- Figures ---------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150})
    # Fig 1: PR curves (test).
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (rec, prec) in pr_curves.items():
        ax.step(rec, prec, where="post",
                label=f"{name} (AP={results[name]['test']['AP']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision (non-interpolated AP)")
    ax.set_title("Precision–recall curves, test set (chronological)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_pr_curves.png")
    plt.close(fig)
    # Fig 2: reliability (test), B logistic vs B GBM.
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, p in [("B logistic", p_blog_test),
                     ("B GBM", gbm_probs[("B_gbm", "test")])]:
        rows, _, _, _ = calibration_table(y["test"], p)
        xs = [r["mean_pred"] for r in rows if r["n"] > 0]
        ys = [r["frac_pos"] for r in rows if r["n"] > 0]
        ax.plot(xs, ys, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed conversion rate (bin)")
    ax.set_title("Reliability diagram, test set")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_reliability.png")
    plt.close(fig)
    # Fig 3: test AP bars A/B/C (+ GBM) and corrected D/E.
    labels = ["A leakage\n(logistic)", "B baseline\n(logistic)",
              "B GBM", "C conservative\n(logistic)", "C GBM",
              "D defensible\n(logistic)", "D GBM",
              "E categorical-only\n(logistic)"]
    keys = ["A_leakage_diagnostic_logistic", "B_baseline_logistic", "B_gbm",
            "C_conservative_logistic", "C_gbm",
            "D_defensible_logistic", "D_gbm", "E_categorical_only_logistic"]
    vals = [results[k]["test"]["AP"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, vals)
    ax.set_ylabel("Test average precision")
    ax.set_title("Test AP by arm — the leakage drop (A vs B)")
    ax.set_ylim(0, max(vals) * 1.15)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_test_ap_bars.png")
    plt.close(fig)

    # ---- Aggregate outputs ----------------------------------------------
    pr_rows = []
    for name, (rec, prec) in pr_curves.items():
        for r, p_ in zip(rec[::10], prec[::10]):
            pr_rows.append({"model": name, "recall": round(float(r), 4),
                            "precision": round(float(p_), 4)})
    pd.DataFrame(pr_rows).to_csv(out / "pr_curves_test.csv", index=False)
    cal_rows = []
    for label, c in calib.items():
        for r in c["bins"]:
            cal_rows.append({"model": label, **r})
    pd.DataFrame(cal_rows).to_csv(out / "calibration_test.csv", index=False)

    metrics = {
        "reproduction": {
            "data_path": args.data, "scanned_rows": scanned,
            "sample_rows": int(len(df)), "seed": SEED,
            "separator": delim, "header_present": header,
            "splits": bounds, "train_prevalence": train_prev,
            "arm_features": {
                "A_leakage_diagnostic_logistic": {
                    "numeric": B_NUMERIC, "categorical": B_CATEGORICAL,
                    "leaky_numeric_added": OUTCOME_DERIVED},
                "B_baseline_logistic": {"numeric": B_NUMERIC,
                                        "categorical": B_CATEGORICAL},
                "C_conservative_logistic": {"numeric": C_NUMERIC,
                                            "categorical": C_CATEGORICAL},
                "B_gbm": {"numeric": B_NUMERIC,
                          "categorical": B_CATEGORICAL},
                "C_gbm": {"numeric": C_NUMERIC,
                          "categorical": C_CATEGORICAL},
                "D_defensible_logistic": {"numeric": D_NUMERIC,
                                          "categorical": D_CATEGORICAL,
                                          "note": "B minus product_price; "
                                                  "product_price is 0.0 for "
                                                  "every non-conversion"},
                "D_gbm": {"numeric": D_NUMERIC,
                          "categorical": D_CATEGORICAL},
                "E_categorical_only_logistic": {"numeric": E_NUMERIC,
                                                "categorical": E_CATEGORICAL}},
            "models": {
                "logistic": "LogisticRegression(C=1.0, liblinear, "
                            "max_iter=500); OneHotEncoder(min_frequency=20); "
                            "median imputation + StandardScaler fitted on "
                            "train only",
                "gbm": "HistGradientBoostingClassifier(max_iter=300, "
                       "learning_rate=0.06, max_leaf_nodes=31, "
                       "min_samples_leaf=20); native categorical handling "
                       "via OrdinalEncoder; per-column rare values grouped "
                       "to '__OTHER__' (top 200 train values kept) because "
                       "raw hashed cardinality exceeds the 255-bin limit"},
            "versions": {"python": platform.python_version(),
                         "pandas": pd.__version__,
                         "numpy": np.__version__,
                         "sklearn": sklearn.__version__,
                         "matplotlib": matplotlib.__version__},
            "runtime_seconds": round(time.time() - t0, 1)},
        "audit": audit_out,
        "results": results,
        "calibration_test": calib,
        "rolling_windows_B_logistic": windows,
        "label_gap_sweep_B_logistic": gaps,
        "leakage_drop_test_AP_A_minus_B": round(
            results["A_leakage_diagnostic_logistic"]["test"]["AP"]
            - results["B_baseline_logistic"]["test"]["AP"], 4),
        "conservative_drop_test_AP_B_minus_C_logistic": round(
            results["B_baseline_logistic"]["test"]["AP"]
            - results["C_conservative_logistic"]["test"]["AP"], 4),
        # Honest drops once product_price is recognised as outcome-derived:
        # A vs the defensible baseline D, and B vs D (the price effect).
        "honest_leakage_drop_test_AP_A_minus_D": round(
            results["A_leakage_diagnostic_logistic"]["test"]["AP"]
            - results["D_defensible_logistic"]["test"]["AP"], 4),
        "price_effect_test_AP_B_minus_D_logistic": round(
            results["B_baseline_logistic"]["test"]["AP"]
            - results["D_defensible_logistic"]["test"]["AP"], 4),
    }
    (out / "metrics_milestone2.json").write_text(
        json.dumps(metrics, indent=2, default=str))
    log(f"DONE in {time.time() - t0:.1f}s. Leakage drop (A-B, test AP) = "
        f"{metrics['leakage_drop_test_AP_A_minus_B']:.4f}; "
        f"B-C drop = {metrics['conservative_drop_test_AP_B_minus_C_logistic']:.4f}; "
        f"honest drop (A-D) = "
        f"{metrics['honest_leakage_drop_test_AP_A_minus_D']:.4f}; "
        f"price effect (B-D) = "
        f"{metrics['price_effect_test_AP_B_minus_D_logistic']:.4f}",
        fh)
    fh.close()
    print(json.dumps(metrics["results"], indent=2))


if __name__ == "__main__":
    main()
