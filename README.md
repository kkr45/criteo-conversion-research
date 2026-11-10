# Leakage-aware sponsored-search conversion research

**ANLY 530 | KA1 | Milestone 1** · Research design and feasibility

## Research questions
- RQ1: How much do post-outcome fields inflate reported performance?
- RQ2: How does chronological evaluation compare with random splitting?
- RQ3: Are gains in ranking also gains in probability calibration?

Read the research protocol in `research/KA1_Milestone_1_Research_Protocol.docx`.

## Data and terms
Criteo Sponsored Search Conversion Log; each record is a click, binary `Sale` outcome within a **30-day** attribution window; 90 days of anonymized, subsampled logs. Publisher dataset page: https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/ . **Review the publisher's terms before use. Never commit the raw dataset or a row-level sample.** The official original is tab-separated (`CriteoSearchData`); any converted `.csv` can use commas. All model input columns should be assessed for actual availability at prediction time. Hashed identifiers are not plain-text identities, but still require responsible handling.

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```
Place the downloaded file in `data/`, e.g. `data/CriteoSearchData` (tab-separated) or `data/criteo.csv` (comma-separated). In each notebook, change the `DATA_PATH` cell, then **Run All**. The exploratory notebook scans the full dataset, retaining up to 500,000 uniformly selected records by random priorities (not the first 500,000 clicks), and may take time/memory for a 6.4 GB text file. To run faster, point to an existing local sample and set `SAMPLE_ROWS` accordingly; document its selection mechanism.

## Files
- `notebooks/01_eda_and_leakage_audit.ipynb`: local schema, missingness, label prevalence, timestamp coverage, leakage proof, diagnostics.
- `notebooks/02_chronological_baseline.ipynb`: strict click-time feature logistic baseline, chronological holdout, calibration/PR metrics, and optional diagnostic leakage comparison.
- `src/data_utils.py`: bounded sampling, guarded predictor selection, chronological splitting.
- `tests/test_data_utils.py`: synthetic non-sensitive verification.
- `research/references.md`: source links; `research/protocol_word_count.txt`: word-count audit.

## Key safeguards
1. Never include `Sale`, `SalesAmountInEuro`, or `time_delay_for_conversion` among valid predictors.
2. Do not normalize/encode outside of training-fold pipelines.
3. Keep test data untouched until model selection is done.
4. Treat historical splits as **retrospective**: click-time separation does not by itself guarantee that 30-day labels would have matured at train time. A 30-day gap sensitivity analysis is proposed, not asserted to solve all label-timing uncertainty.
5. Report **average precision** (non-interpolated scikit-learn AP), not a trapezoidal PR-area under the same name. Report positive prevalence and calibration.
6. This repository contains *research design and runnable exploratory scripts*, not verified final empirical findings.

## Reproduction record
Record input basename, number of scanned rows, sample selection, SHA256 checksum (optional for large data), timestamps, seed=530, Python and library versions, split boundaries, selected columns, and metrics. Save only aggregate outputs, not row-level source data.

## Timing
Week 10: protocol, local EDA, baselines. Week 11: model/regime controls. Week 12: ablations, metrics, interpretation. Week 13: final paper and 5-minute presentation.

## Contribution
Individual project; contributing student performs data analysis, experiment design, coding, verification, manuscript, repository and oral presentation. Replace with named/team-specific roles if a team is formed. Follow course disclosure rules for research-assistance tools.
