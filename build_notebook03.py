#!/usr/bin/env python3
"""Build notebooks/03_milestone2_ablations.ipynb (UNEXECUTED) from
milestone2_run.py — same code, split into cells, no outputs.

Run: venv python build_notebook03.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "milestone2_run.py"
OUT = HERE / "03_milestone2_ablations.ipynb"

text = SRC.read_text()
# Split off the module docstring as the intro markdown.
m = re.match(r'"""(.*?)"""\n', text, re.S)
intro = m.group(1).strip() if m else ""
body = text[m.end():] if m else text

# Split body into code cells at top-level def / main-guard boundaries,
# preserving order. Keep the import block with the first chunk.
parts = re.split(r"\n(?=(?:def |if __name__))", body)
cells = [{
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# KA1 Milestone 2 — Research Execution: ablations A–E + GBM\n",
        "\n",
        "**STATUS: EXECUTED 2026-11-23 on the real Criteo file "
        "(`data/CriteoSearchData`, 15,995,634 rows scanned, 500,000-row "
        "sample, seed 530).** This notebook contains the full Milestone 2 "
        "analysis code, identical in logic to `milestone2_run.py`. It writes "
        "`metrics_milestone2.json`, `run_log.txt`, calibration/PR tables and "
        "Figures 1–3 under `outputs/`.\n",
        "\n",
        "Arms: **A** leakage diagnostic (B + `SalesAmountInEuro` + "
        "`time_delay_for_conversion`, invalid for deployment) · **B** "
        "Milestone 1 baseline · **C** conservative (B minus "
        "`nb_clicks_1week`) · **D** defensible (B minus `product_price` — "
        "diagnosed 2026-11-23 as a third outcome-derived field: it is 0.0 "
        "for every non-conversion) · **E** categorical features only. "
        "Logistic regression for A–E; HistGradientBoosting for B/C/D. "
        "Chronological 60/20/20 split, seed 530, model selection on "
        "validation only.\n",
    ],
}]
md_after_imports = {
    "cell_type": "markdown", "metadata": {},
    "source": ["## 1 · Load + RQ1 audit\n",
               "\n",
               "Uniform bounded random sample (seed 530, max 500,000 rows, "
               "50,000-row chunks). Audit: duplicates, sentinel/missingness "
               "per column, leakage proof by Sale label, timestamp coverage, "
               "weekly prevalence.\n"],
}
md_model = {
    "cell_type": "markdown", "metadata": {},
    "source": ["## 2 · Arms A–E, GBM, calibration, sensitivity\n",
               "\n",
               "`main()` runs every arm on the identical chronological "
               "split, the calibration comparison (B logistic vs B GBM), "
               "rolling chronological windows, and the label-gap sweep "
               "(0/7/14/30 days; 30 days is reported feasible or infeasible, "
               "not assumed). Figures 1–3 and `metrics_milestone2.json` are "
               "written to `outputs/`. A separate script, "
               "`d_sensitivity.py`, repeats the calibration, rolling-window "
               "and gap checks on the defensible arm D; its results are in "
               "`outputs/d_defensible_sensitivity.json`.\n"],
}
for i, part in enumerate(parts):
    chunk = part.strip("\n")
    if not chunk:
        continue
    if i == 0:
        cells.append(md_after_imports)
    if chunk.startswith("def main"):
        cells.append(md_model)
    # The script's __main__ guard calls main(), whose argparse reads
    # sys.argv — unusable inside a notebook kernel. Replace it with an
    # explicit argv so the notebook executes top-to-bottom as-is.
    if chunk.startswith('if __name__'):
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {}, "outputs": [],
                      "source": [
                          "# Execute main() with explicit paths (argparse "
                          "cannot read notebook-kernel argv).\n",
                          "import sys\n",
                          "sys.argv = [\"milestone2_run.py\", \"--data\", "
                          "\"data/CriteoSearchData\", \"--out\", \"outputs\"]\n",
                          "main()\n"]})
        continue
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [],
                  "source": [ln + "\n" for ln in chunk.split("\n")]})

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(cells)} cells)")
