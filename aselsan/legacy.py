"""Version 1, run from its own notebook (legacy/main_v1.ipynb) on the frozen data, to reproduce its numbers.

The code cells of version 1 are executed in order in one namespace, with these changes only: the `!pip` lines and
the yfinance import are removed, the Yahoo download is replaced by the frozen file (same columns, same window), and
figure cells and `display` are skipped. Nothing else of version 1 is modified, so the results are what its code computes.

Author: Muhammed Ikbal Yilmaz
"""

import json
import os
import tempfile
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "legacy" / "main_v1.ipynb"
CELLS = [4, 7, 9, 11, 13, 17, 20, 22, 23, 26, 28]   # estimation, robustness, OOS, ML; figure cells 6, 15, 30 skipped


def _cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]]


def v1_final_df(yahoo, cds):
    """The merged table of version 1 (cell 2): outer join of the Yahoo closes and the CDS series, rows without an
    ASELS close dropped, then forward- and backward-filled."""
    master = yahoo.join(cds.to_frame("CDS"), how="outer")
    master = master.dropna(subset=["ASELS"]).sort_index().ffill().bfill()
    return master[["ASELS", "BITCOIN", "BRENT", "DXY", "USDTRY", "BIST100", "US10Y", "VIX", "CDS"]]


def run_v1(yahoo, cds):
    """Executes the analysis cells of version 1 and returns their namespace (final_df, res, oos, perf,
    horse_race, ...). Printing of version 1 is suppressed; its output directory is a temporary folder."""
    import contextlib
    import io

    import matplotlib
    matplotlib.use("Agg")
    cells = _cells()
    ns = {"display": lambda *a, **k: None}
    setup = "\n".join(line for line in cells[1].splitlines()
                      if not line.lstrip().startswith("!") and line.strip() != "import yfinance as yf")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        os.chdir(tmp)
        try:
            exec(compile(setup, "v1 cell 1", "exec"), ns)
            ns["final_df"] = v1_final_df(yahoo, cds)
            ns["OUTDIR"], ns["PREFIX"] = tmp, "ASELS_ARX"
            for i in CELLS:
                src = "\n".join(line for line in cells[i].splitlines() if not line.lstrip().startswith("!"))
                exec(compile(src, f"v1 cell {i}", "exec"), ns)
        finally:
            os.chdir(cwd)
    return ns
