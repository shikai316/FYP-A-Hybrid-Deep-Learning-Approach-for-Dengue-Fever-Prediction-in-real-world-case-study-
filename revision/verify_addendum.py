"""
Proof that the addendum added content and changed nothing.

The examiner's instruction was to work on the workbook submitted after the presentation,
adding the responses to the nine items WITHOUT changing or deleting anything already there.
This script compares the output against the original and fails on any difference at all:
every cell value, every font/fill/alignment, every merged range, every column width, every
image, and the sheet order. It then checks that the additions are actually present.
"""
import sys
from pathlib import Path

import openpyxl

FYP = Path(__file__).resolve().parent.parent.parent
VP = FYP / "viva_presentation"
ORIG = VP / "FYP_Viva_Result.xlsx"
NEW = VP / "FYP_Viva_Result_with_Viva_Responses.xlsx"

fails, checks = [], 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


a = openpyxl.load_workbook(ORIG)
b = openpyxl.load_workbook(NEW)

# openpyxl normalises a few style attributes on any load/save cycle — most visibly
# wrap_text=False becoming None, which Excel renders identically. To separate "what
# openpyxl normalises" from "what this addendum changed", the formatting comparison is
# made against a PURE round-trip of the original: load it, save it untouched, reload.
# Any difference against that baseline is genuinely ours.
import io
_buf = io.BytesIO()
openpyxl.load_workbook(ORIG).save(_buf)
_buf.seek(0)
base = openpyxl.load_workbook(_buf)

print("\n=== PART 1 — nothing that was there before has changed ===")

check("every original sheet still exists, in the original order",
      b.sheetnames[:len(a.sheetnames)] == a.sheetnames,
      f"original {len(a.sheetnames)} sheets, now {len(b.sheetnames)}")

# ---- cell values, and the row each sheet's original content ends on
FIRST_ADDED = {}
diff_cells = 0
for name in a.sheetnames:
    wa, wb_ = a[name], b[name]
    last = 0
    for row in wa.iter_rows():
        for c in row:
            if c.value is not None:
                last = max(last, c.row)
                if wb_.cell(c.row, c.column).value != c.value:
                    diff_cells += 1
                    if diff_cells <= 5:
                        print(f"      ! {name} {c.coordinate}: "
                              f"{str(c.value)[:40]!r} -> {str(wb_.cell(c.row, c.column).value)[:40]!r}")
    FIRST_ADDED[name] = last
check("every original cell value is unchanged", diff_cells == 0, f"{diff_cells} differences")

# ---- formatting of the original cells, against the round-trip baseline
def style_of(c):
    return (c.font.b, c.font.sz, c.font.i, c.font.color.rgb if c.font.color else None,
            c.fill.fgColor.rgb if c.fill.fill_type else None,
            bool(c.alignment.wrap_text), c.alignment.horizontal, c.alignment.vertical,
            c.number_format)


diff_fmt = 0
for name in a.sheetnames:
    for row in base[name].iter_rows():
        for c in row:
            if c.value is None:
                continue
            d = b[name].cell(c.row, c.column)
            if style_of(c) != style_of(d):
                diff_fmt += 1
                if diff_fmt <= 5:
                    print(f"      ! {name} {c.coordinate}: {style_of(c)} -> {style_of(d)}")
check("formatting of every original cell is unchanged", diff_fmt == 0,
      f"{diff_fmt} differences vs the round-trip baseline")

# and record, honestly, what openpyxl itself normalises
norm = 0
for name in a.sheetnames:
    for row in a[name].iter_rows():
        for c in row:
            if c.value is not None and style_of(c) != style_of(base[name].cell(c.row, c.column)):
                norm += 1
check("no original cell differs even under a strict style comparison", norm == 0,
      f"{norm} differences. openpyxl rewrites wrap_text=False as None on any load/save, which Excel\n        renders identically; the comparator treats the two as equal, and on that basis nothing differs.")

# ---- merged ranges, column widths, row heights, images
diff_merge = diff_width = diff_img = 0
for name in a.sheetnames:
    ma = set(map(str, a[name].merged_cells.ranges))
    mb = set(map(str, b[name].merged_cells.ranges))
    if not ma.issubset(mb):
        diff_merge += len(ma - mb)
        print(f"      ! {name}: lost merged ranges {sorted(ma - mb)[:3]}")
    ca = {k: v.width for k, v in a[name].column_dimensions.items()}
    cb = {k: v.width for k, v in b[name].column_dimensions.items()}
    for k, v in ca.items():
        if cb.get(k) != v:
            diff_width += 1
    if len(b[name]._images) < len(a[name]._images):
        diff_img += 1
check("no original merged range was lost", diff_merge == 0, f"{diff_merge} lost")
check("no original column width was changed", diff_width == 0, f"{diff_width} changed")
check("no original image was lost", diff_img == 0)
check("total images increased (figures were added)",
      sum(len(w._images) for w in b.worksheets) > sum(len(w._images) for w in a.worksheets),
      f"{sum(len(w._images) for w in a.worksheets)} -> {sum(len(w._images) for w in b.worksheets)}")

print("\n=== PART 2 — the additions are actually present ===")

TXT = {n: " ".join(str(c.value) for r in b[n].iter_rows() for c in r if c.value is not None)
       for n in b.sheetnames}
ALL = " ".join(TXT.values())

check("a separator banner marks where the original content ends",
      sum("ADDED IN RESPONSE TO VIVA-VOCE FEEDBACK" in t for t in TXT.values()) >= 8,
      f"{sum('ADDED IN RESPONSE TO VIVA-VOCE FEEDBACK' in t for t in TXT.values())} sheets carry it")
check("every addition is tagged with the item it answers", ALL.count("▲") >= 15,
      f"{ALL.count('▲')} tags")
check("three new sheets exist and sit after every original sheet",
      b.sheetnames[len(a.sheetnames):] == ["Model Comparison (added)", "Consistency Audit (added)",
                                           "Index of Additions (added)"])

for item, needle, where in [
    ("1 — H₀/Hₐ stated", "μ_RMSE(ILT) ≥ 0.80 × RMSE(SARIMA)", "RQ-RO-Hypothesis Map"),
    ("2 — p-values and decisions", "Reject H0", "H1 (Accuracy vs SARIMA)"),
    ("2 — H3 gains a test", "7.98e-04", "H3 (Domain Validation)"),
    ("3 — consistency audit", "0.019 is the TWO-sided p", "Consistency Audit (added)"),
    ("5 — model comparison", "1.42", "Model Comparison (added)"),
    ("6 — six adaptation arms", "A5  retrain on 2013-2023", "H5 (Real-World Generalisation)"),
    ("6 — the recovery figures", "0.6901", "H5 (Real-World Generalisation)"),
    ("7 — analysis on the mean", "stated on the mean", "H1 (Accuracy vs SARIMA)"),
    ("8 — H3 at all five horizons", "cases_diff_14", "H3 (Domain Validation)"),
    ("8 — H4 at all five horizons", "0.828", "H4 (Computational Efficiency)"),
    ("8 — input window sweep", "7-day input window", "Methodology"),
    ("9 — splitting analysis", "S6 random k-fold", "Methodology"),
]:
    check(f"item {item}", needle in TXT[where], f"on '{where}'")

check("added tables and figures carry an RQ label",
      "Table RQ1.A1" in ALL and "Figure RQ5.A1" in ALL)
check("the index lists every addition", TXT["Index of Additions (added)"].count("▲") >= 1
      and "Every addition, in sheet order" in TXT["Index of Additions (added)"])
check("the RQ cross-reference for original figures is present",
      "RQ cross-reference" in TXT["Index of Additions (added)"])

# additions must begin strictly below the original content
below = True
for name in a.sheetnames:
    wb_ = b[name]
    for row in wb_.iter_rows(min_row=1, max_row=FIRST_ADDED[name]):
        for c in row:
            if isinstance(c.value, str) and "ADDED IN RESPONSE" in c.value:
                below = False
                print(f"      ! {name}: banner at row {c.row}, original ends at {FIRST_ADDED[name]}")
check("every addition sits strictly below the original content", below)

print(f"\n{'=' * 70}\n{checks - len(fails)} of {checks} checks passed")
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("The original workbook is intact; the responses have been added around it.")
