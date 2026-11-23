#!/usr/bin/env python3
"""Execute 03_milestone2_ablations.ipynb in-process and save the executed copy.

Why in-process: `jupyter nbconvert --execute` spawns a separate kernel
process and was SIGTERMed mid-run on this VM (2026-11-23, ~160 s in,
during the 4-minute file scan). Running the cells through an IPython
InteractiveShell in this same process has the same memory footprint as
the already-successful `milestone2_run.py` runs, so it should complete.
Cell outputs are captured as stream outputs; errors are recorded, not
hidden, and the script exits non-zero if any cell errors.
"""
import io
import json
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell

HERE = Path(__file__).resolve().parent
SRC = HERE / "03_milestone2_ablations.ipynb"
DST = HERE / "03_milestone2_ablations_executed.ipynb"

nb = nbformat.read(str(SRC), as_version=4)
shell = InteractiveShell.instance()
shell.colors = "NoColor"

t0 = time.time()
errors = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    src = cell.source
    buf_out, buf_err = io.StringIO(), io.StringIO()
    cell.outputs = []
    cell.execution_count = i + 1
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        res = shell.run_cell(src, store_history=False)
    out_text = buf_out.getvalue()
    err_text = buf_err.getvalue()
    if out_text:
        cell.outputs.append(nbformat.v4.new_output(
            "stream", name="stdout", text=out_text))
    if err_text:
        cell.outputs.append(nbformat.v4.new_output(
            "stream", name="stderr", text=err_text))
    if res.error_before_exec or res.error_in_exec:
        err = res.error_in_exec or res.error_before_exec
        cell.outputs.append(nbformat.v4.new_output(
            "error", ename=type(err).__name__, evalue=str(err),
            traceback=[str(err)]))
        errors.append((i, type(err).__name__, str(err)))
        print(f"cell {i}: ERROR {type(err).__name__}: {err}", flush=True)
    else:
        print(f"cell {i}: ok ({time.time() - t0:.0f}s elapsed)", flush=True)

nbformat.write(nb, str(DST))
print(f"wrote {DST} in {time.time() - t0:.0f}s; errors={errors}", flush=True)
if errors:
    sys.exit(1)
