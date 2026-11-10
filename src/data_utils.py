"""Local-only Criteo data loading and leakage guards."""
from pathlib import Path
import numpy as np
import pandas as pd

LEAKY = {"Sale", "SalesAmountInEuro", "time_delay_for_conversion"}
BAD_TIMESTAMP = {"0", "-1", "", "nan", "None"}


def read_bounded_random_sample(path, sample_rows=500_000, chunksize=100_000, seed=530, sep=None):
    """Retain the globally lowest independent random priorities.

    Every input row has equal inclusion probability; unlike df.head(), this
    does not accidentally select the earliest contiguous period. Requires a
    sequential scan of the complete local source (can take several minutes).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if sep is None:
        sep = "\t" if path.suffix.lower() not in {".csv"} else ","
    rng = np.random.default_rng(seed)
    selected, best_keys, total = [], np.array([], dtype=float), 0
    for block in pd.read_csv(path, sep=sep, chunksize=chunksize, low_memory=False):
        block.columns = block.columns.astype(str).str.strip()
        total += len(block)
        k = rng.random(len(block))
        if len(k) > sample_rows:
            take = np.argpartition(k, sample_rows-1)[:sample_rows]
            block, k = block.iloc[take].copy(), k[take]
        selected.append(block)
        best_keys = np.concatenate([best_keys,k])
        merged = pd.concat(selected, ignore_index=True)
        if len(merged)>sample_rows:
            take=np.argpartition(best_keys,sample_rows-1)[:sample_rows]
            selected=[merged.iloc[take].reset_index(drop=True)]
            best_keys=best_keys[take]
        else:
            selected=[merged]
    if not selected: raise ValueError("Empty dataset")
    return selected[0].reset_index(drop=True), total


def clean_click_time_features(df, reduced=False):
    """Drop target and outcomes, guard optional identifiers, replace sentinels."""
    x=df.drop(columns=[c for c in LEAKY if c in df.columns], errors="ignore").copy()
    if reduced:
        x=x.drop(columns=[c for c in ["user_id","product_id","product_title","audience_id","partner_id"] if c in x.columns], errors="ignore")
    x=x.drop(columns=["click_timestamp"],errors="ignore")  # split key, not a raw memorization feature
    for col in x.columns:
        x[col] = x[col].replace([-1,"-1"],np.nan)
    assert not LEAKY.intersection(x.columns)
    return x


def chronological_split(df,train_frac=.6,val_frac=.2):
    if 'click_timestamp' not in df: raise ValueError('Missing click_timestamp')
    ts=pd.to_numeric(df['click_timestamp'],errors='coerce')
    valid=ts.notna() & (ts!=0)
    ordered=df.loc[valid].assign(_order_time=ts[valid]).sort_values('_order_time',kind='stable').drop(columns='_order_time')
    n=len(ordered);i=int(n*train_frac);j=int(n*(train_frac+val_frac))
    if i<1 or j<=i or j>=n: raise ValueError('Not enough timestamp-valid data')
    return ordered.iloc[:i],ordered.iloc[i:j],ordered.iloc[j:]


def leakage_audit(df):
    target=pd.to_numeric(df['Sale'],errors='coerce')
    result=[]
    for col in ['SalesAmountInEuro','time_delay_for_conversion']:
        if col in df:
            sentinel=df[col].astype(str).eq('-1')
            for label in (0,1):
                m=target.eq(label)
                result.append({'column':col,'Sale':label,'rows':int(m.sum()),'sentinel_rate':float(sentinel[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(result)
