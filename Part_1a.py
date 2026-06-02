"""
PART 1A — DATA INGESTION & DATA QUALITY REPORT
Clinical Trials Dataset | 1,000 Interventional Oncology Trials

"""

import ast
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION  — update FILE path if needed
# ============================================================

FILE = (
    "/Users/khusbuagarwal/Downloads/SampleDateExtract.xlsx - 1000_inteventional_trials.csv"
)
OUT = Path("/Users/khusbuagarwal/Downloads/I3_health")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    """Print a bold section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def parse_list_col(val) -> list:
    """
    Safely parse a Python list-literal string stored in a CSV cell.
    Returns a Python list, or an empty list on failure.
    Examples:
      "['NSCLC', 'Lung']"           -> ['NSCLC', 'Lung']
      "[['Drug A'], ['Drug B']]"    -> [['Drug A'], ['Drug B']]
      NaN / None                    -> []
    """
    if pd.isna(val):
        return []
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else [result]
    except Exception:
        return [str(val)]


def flatten_list_col(series: pd.Series) -> list:
    """
    Flatten all values in a list-column Series into a single list of strings.
    Handles nested lists (list-of-lists) one level deep.
    """
    items = []
    for val in series.dropna():
        parsed = parse_list_col(val)
        for item in parsed:
            if isinstance(item, list):
                items.extend(str(x).strip() for x in item if x)
            elif item:
                items.append(str(item).strip())
    return items


def count_non_ascii(text: str) -> int:
    """Return number of characters with ord > 127."""
    return sum(1 for c in str(text) if ord(c) > 127)


# ============================================================
# SECTION 1 — DATASET OVERVIEW
# ============================================================

section("1. DATASET OVERVIEW")

df = pd.read_csv(FILE)

print(f"  Rows      : {df.shape[0]:,}")
print(f"  Columns   : {df.shape[1]}")
print()
print(f"  {'#':<4}  {'Column Name':<32}  {'Dtype'}")
print("  " + "-" * 55)
for i, col in enumerate(df.columns, 1):
    print(f"  {i:<4}  {col:<32}  {df[col].dtype}")

print("\n  First 3 rows (key columns only):")
preview = ["nct_id", "phase", "recruitment_status", "enrollment", "start_date"]
avail   = [c for c in preview if c in df.columns]
print(df[avail].head(3).to_string(index=False))


# ============================================================
# SECTION 2 — FIELD COMPLETENESS REPORT
# ============================================================

section("2. FIELD COMPLETENESS REPORT")

quality_report = pd.DataFrame({
    "dtype"         : df.dtypes.astype(str),
    "null_count"    : df.isnull().sum(),
    "missing_%"     : round(df.isnull().mean() * 100, 2),
    "unique_values" : df.nunique(dropna=True),
    "completeness_%": round((1 - df.isnull().mean()) * 100, 2),
})

def _severity(pct: float) -> str:
    if pct == 0:      return "Excellent"
    elif pct < 5:     return "Good"
    elif pct < 20:    return "Warning"
    else:             return "Critical"

quality_report["severity"] = quality_report["missing_%"].apply(_severity)

print(quality_report.sort_values("missing_%", ascending=False).to_string())

low_complete = quality_report[quality_report["completeness_%"] < 95]
mean_complete = quality_report["completeness_%"].mean()

print(f"\n  Columns below 95% completeness ({len(low_complete)}): "
      f"{list(low_complete.index)}")
print(f"  Mean completeness across all {len(df.columns)} columns : "
      f"{mean_complete:.1f}%")


# ============================================================
# SECTION 3 — DUPLICATE ANALYSIS
# ============================================================

section("3. DUPLICATE ANALYSIS")

n_dup_rows = df.duplicated().sum()
print(f"  Exact duplicate rows         : {n_dup_rows}")

if "nct_id" in df.columns:
    n_dup_nct = df["nct_id"].duplicated().sum()
    print(f"  Duplicate nct_id values      : {n_dup_nct}")
    print(f"  Unique nct_ids               : {df['nct_id'].nunique():,}")

if "ID-datalake" in df.columns:
    print(f"  Duplicate ID-datalake values : {df['ID-datalake'].duplicated().sum()}")

if n_dup_rows == 0 and n_dup_nct == 0:
    print("\n  -> Zero duplicates. Every trial has a unique registry ID. ✓")
else:
    dup_nct_df = df[df["nct_id"].duplicated(keep=False)]
    print("\n  Sample duplicate nct_ids:")
    print(dup_nct_df["nct_id"].value_counts().head(5).to_string())


# ============================================================
# SECTION 4 — CATEGORICAL FIELD AUDIT
# ============================================================

section("4. CATEGORICAL FIELD AUDIT")

cat_cols = ["phase", "recruitment_status", "enrollment_type"]

phase_distribution  = pd.DataFrame()
status_distribution = pd.DataFrame()

for col in cat_cols:
    if col not in df.columns:
        continue
    print(f"\n  {'--- ' + col.upper() + ' ---':}")
    dist = df[col].value_counts(dropna=False)
    pcts = (dist / len(df) * 100).round(1)
    print(f"  {'Value':<40}  {'Count':>6}  {'Pct':>6}")
    print("  " + "-" * 56)
    for val, cnt in dist.items():
        label = "(missing)" if pd.isna(val) else str(val)
        print(f"  {label:<40}  {cnt:>6}  {pcts[val]:>5.1f}%")

    if col == "phase":
        phase_distribution = dist.reset_index()
        phase_distribution.columns = ["phase", "count"]
    elif col == "recruitment_status":
        status_distribution = dist.reset_index()
        status_distribution.columns = ["recruitment_status", "count"]


# ============================================================
# SECTION 5 — DATE AUDIT
# ============================================================

section("5. DATE AUDIT")

date_cols_raw = ["start_date", "completion_date", "primary_completion_date"]
date_cols     = [c for c in date_cols_raw if c in df.columns]
parse_failures_total = 0

for col in date_cols:
    parsed     = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    raw_nulls  = int(df[col].isna().sum())
    failures   = int(parsed.isna().sum()) - raw_nulls
    parse_failures_total += failures
    valid      = parsed.dropna()

    print(f"\n  {col}:")
    print(f"    Raw nulls        : {raw_nulls}")
    print(f"    Parse failures   : {failures}")
    if len(valid) > 0:
        print(f"    Earliest         : {valid.min().date()}")
        print(f"    Latest           : {valid.max().date()}")
        fmt_sample = df[col].dropna().iloc[0] if df[col].notna().any() else "—"
        print(f"    Format sample    : {fmt_sample}")

# Date consistency — completion should not precede start
if {"start_date", "completion_date"}.issubset(df.columns):
    start_dt = pd.to_datetime(df["start_date"], dayfirst=True, errors="coerce")
    comp_dt  = pd.to_datetime(df["completion_date"], dayfirst=True, errors="coerce")
    invalid  = int((comp_dt < start_dt).sum())
    print(f"\n  Completion date before start date : {invalid}")
    if invalid > 0:
        print("  -> Possible data entry errors or protocol amendments.")

# Rows with ALL dates missing
if date_cols:
    all_null = df[date_cols].isnull().all(axis=1).sum()
    print(f"  Rows with ALL dates missing       : {all_null}")


# ============================================================
# SECTION 6 — NUMERIC FIELD AUDIT & OUTLIERS
# ============================================================

section("6. NUMERIC FIELD AUDIT & OUTLIERS")

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if num_cols:
    print(df[num_cols].describe().T.round(2).to_string())

if "enrollment" in df.columns:
    enroll = pd.to_numeric(df["enrollment"], errors="coerce")
    print(f"\n  Enrollment field detail:")
    print(f"    Valid (non-null)   : {enroll.notna().sum():>6}")
    print(f"    Missing (null)     : {enroll.isna().sum():>6}")
    print(f"    Zero               : {int((enroll == 0).sum()):>6}")
    print(f"    Negative           : {int((enroll < 0).sum()):>6}")
    print(f"    Median             : {enroll.median():>6.0f}")
    print(f"    Mean               : {enroll.mean():>6.1f}")
    print(f"    Max                : {enroll.max():>6.0f}")
    print(f"    Min (excl. zeros)  : {enroll[enroll > 0].min():>6.0f}")

    # IQR outlier detection
    Q1, Q3 = enroll.quantile(0.25), enroll.quantile(0.75)
    IQR    = Q3 - Q1
    lo_f   = Q1 - 1.5 * IQR
    hi_f   = Q3 + 1.5 * IQR
    outliers = int(((enroll < lo_f) | (enroll > hi_f)).sum())
    print(f"\n    IQR            : {IQR:.0f}")
    print(f"    Lower fence    : {lo_f:.0f}")
    print(f"    Upper fence    : {hi_f:.0f}")
    print(f"    Outliers (IQR) : {outliers}  "
          f"({100*outliers/enroll.notna().sum():.1f}% of valid)")


# ============================================================
# SECTION 7 — LIST-ENCODED COLUMN OVERVIEW
# ============================================================

section("7. LIST-ENCODED COLUMN OVERVIEW")

list_cols = [c for c in [
    "indications", "interventions_drugs", "drugs_datalake",
    "main_technologies", "specific_technologies",
    "target_names", "target_abbreviations",
] if c in df.columns]

print(f"  {'Column':<30}  {'Non-null':>8}  {'Sample value (first 70 chars)'}")
print("  " + "-" * 84)
for col in list_cols:
    nn     = int(df[col].notna().sum())
    sample = str(df[col].dropna().iloc[0])[:70] if nn > 0 else "—"
    print(f"  {col:<30}  {nn:>8}  {sample}")

print("""
  NOTE: All 7 columns store Python list literals as plain strings.
  Columns 12-15 (indications...specific_technologies) contain flat lists.
  Columns 16-18 (target_names...target_abbreviations) contain nested
  list-of-lists — one inner list per drug slot in the trial.
  These structures require ast.literal_eval() to parse correctly.
""")


# ============================================================
# SECTION 8 — DIRTY VALUE DETECTION
# ============================================================

section("8. DIRTY VALUE DETECTION")

print("  Checking categorical columns for casing inconsistencies,\n"
      "  whitespace issues, and near-duplicate labels.\n")

for col in ["phase", "recruitment_status", "enrollment_type"]:
    if col not in df.columns:
        continue
    raw_vals = df[col].dropna().unique().tolist()

    # Group by upper-stripped form
    upper_map: dict = {}
    for v in raw_vals:
        key = str(v).strip().upper()
        upper_map.setdefault(key, []).append(v)

    inconsistent = {k: vs for k, vs in upper_map.items() if len(vs) > 1}
    ws_issues    = [v for v in raw_vals if str(v) != str(v).strip()]

    print(f"  {col.upper()}:")
    if inconsistent:
        print(f"    Case inconsistencies found ({len(inconsistent)}):")
        for k, vs in list(inconsistent.items())[:5]:
            print(f"      '{k}' appears as: {vs}")
    else:
        print(f"    No case inconsistencies ✓")

    if ws_issues:
        print(f"    Whitespace padding: {ws_issues[:5]}")
    else:
        print(f"    No whitespace issues ✓")
    print()


# ============================================================
# SECTION 9 — MOJIBAKE / ENCODING DETECTION
# ============================================================

section("9. MOJIBAKE / ENCODING DETECTION")

print(f"  Scanning {len(df.select_dtypes('object').columns)} text columns "
      f"for non-ASCII characters (ord > 127).\n")

text_cols         = df.select_dtypes("object").columns.tolist()
encoding_summary  = []

for col in text_cols:
    affected = int(df[col].dropna().apply(
        lambda x: count_non_ascii(str(x)) > 0).sum())
    if affected > 0:
        sample = next((str(v) for v in df[col].dropna()
                       if count_non_ascii(str(v)) > 0), "—")
        encoding_summary.append({
            "column"       : col,
            "rows_affected": affected,
            "sample"       : sample[:70],
        })
        print(f"  {col:<30}: {affected:>4} rows affected")
        print(f"    Sample: {sample[:70]}")
        print()

if not encoding_summary:
    print("  No encoding issues detected. ✓")
else:
    print(f"  Total columns affected : {len(encoding_summary)}")
    print("""
  ROOT CAUSE:
    Data was likely encoded as latin-1 (ISO-8859-1) but read as UTF-8.
    Characters like α (alpha), β (beta) render as multi-byte garbage.

  FIX (apply in Part 1B):
    for col in affected_cols:
        df[col] = df[col].str.encode('latin-1', errors='ignore') \\
                         .str.decode('utf-8', errors='replace')
    """)


# ============================================================
# SECTION 10 — LIST-COLUMN DEPTH ANALYSIS
# ============================================================

section("10. LIST-COLUMN DEPTH ANALYSIS")

print("  Counting items per cell to understand list-column cardinality.\n")
print(f"  {'Column':<30}  {'Min':>4}  {'Median':>7}  {'Max':>4}  "
      f"{'Empty%':>7}  {'Multi-item%':>12}")
print("  " + "-" * 72)

depth_rows = []
for col in list_cols:
    lengths = df[col].apply(lambda x: len(parse_list_col(x)))
    row = {
        "column"      : col,
        "min_items"   : int(lengths.min()),
        "median_items": float(lengths.median()),
        "max_items"   : int(lengths.max()),
        "pct_empty"   : round(100 * (lengths == 0).mean(), 1),
        "pct_multi"   : round(100 * (lengths > 1).mean(), 1),
    }
    depth_rows.append(row)
    print(f"  {col:<30}  {row['min_items']:>4}  "
          f"{row['median_items']:>7.1f}  {row['max_items']:>4}  "
          f"{row['pct_empty']:>6.1f}%  {row['pct_multi']:>11.1f}%")

depth_df = pd.DataFrame(depth_rows)


# ============================================================
# SECTION 11 — COMBINATION THERAPY ANALYSIS
# ============================================================

section("11. COMBINATION THERAPY ANALYSIS")

if "interventions_drugs" in df.columns:
    drug_counts = df["interventions_drugs"].apply(
        lambda x: len(parse_list_col(x)))

    monotherapy = int((drug_counts == 1).sum())
    combo       = int((drug_counts >= 2).sum())
    no_drug     = int((drug_counts == 0).sum())
    n           = len(df)

    print(f"  Monotherapy  (1 drug)    : {monotherapy:>5}  ({100*monotherapy/n:.1f}%)")
    print(f"  Combination  (>=2 drugs) : {combo:>5}  ({100*combo/n:.1f}%)")
    print(f"  No drug listed           : {no_drug:>5}  ({100*no_drug/n:.1f}%)")
    print(f"\n  Drug count distribution:")
    vc = drug_counts.value_counts().sort_index()
    for count_val, freq in vc.items():
        bar = chr(9608) * min(int(freq / 5), 40)
        print(f"    {count_val:>3} drug(s): {freq:>4}  {bar}")
    print(f"\n  Median drugs per trial   : {drug_counts.median():.1f}")
    print(f"  Maximum drugs in 1 trial : {drug_counts.max()}")
    print(f"\n  NOTE: 'combination trial' is defined as >=2 unique drugs.")
    print(f"  This definition aligns with Part 2B D6 (therapy_type column).")


# ============================================================
# SECTION 12 — TOP VALUES IN LIST COLUMNS
# ============================================================

section("12. TOP VALUES IN LIST COLUMNS")

top_col_map = {
    "indications"          : ("Top 10 Indications",    10),
    "main_technologies"    : ("Top 10 Technologies",   10),
    "target_abbreviations" : ("Top 15 Targets",        15),
}

for col, (label, top_n) in top_col_map.items():
    if col not in df.columns:
        continue
    all_vals = flatten_list_col(df[col])
    counter  = Counter(v for v in all_vals if v and v != "nan")
    total    = len(df)

    print(f"\n  {label}:")
    print(f"  {'Rank':<5}  {'Value':<38}  {'Trials':>7}  {'% Trials':>9}")
    print("  " + "-" * 62)
    for rank, (val, cnt) in enumerate(counter.most_common(top_n), 1):
        print(f"  {rank:<5}  {val:<38}  {cnt:>7}  {100*cnt/total:>8.1f}%")


# ============================================================
# SECTION 13 — DATE RANGE ANALYSIS
# ============================================================

section("13. DATE RANGE ANALYSIS")

if "start_date" in df.columns:
    start_dt = pd.to_datetime(df["start_date"], dayfirst=True, errors="coerce")
    start_yr = start_dt.dt.year

    print("  Trials by start decade:")
    for d in range(1980, 2030, 10):
        cnt = int(((start_yr >= d) & (start_yr < d + 10)).sum())
        if cnt > 0:
            bar = chr(9608) * int(cnt / 5)
            print(f"    {d}s: {cnt:>4}  {bar}")

    print("\n  Trials by start year (2015 onwards):")
    for yr in range(2015, 2027):
        cnt = int((start_yr == yr).sum())
        bar = chr(9608) * int(cnt / 2)
        print(f"    {yr}: {cnt:>4}  {bar}")

    # Duration statistics
    if "primary_completion_date" in df.columns:
        end_dt    = pd.to_datetime(df["primary_completion_date"],
                                    dayfirst=True, errors="coerce")
        dur_days  = (end_dt - start_dt).dt.days
        valid_dur = dur_days[(dur_days.notna()) & (dur_days >= 0)]
        print(f"\n  Trial duration (start -> primary completion):")
        print(f"    n with valid duration : {len(valid_dur)}")
        print(f"    Median                : {valid_dur.median():.0f} days  "
              f"({valid_dur.median()/365:.1f} yrs)")
        print(f"    Mean                  : {valid_dur.mean():.0f} days  "
              f"({valid_dur.mean()/365:.1f} yrs)")
        print(f"    Min                   : {valid_dur.min():.0f} days")
        print(f"    Max                   : {valid_dur.max():.0f} days  "
              f"({valid_dur.max()/365:.1f} yrs)")


# ============================================================
# SECTION 14 — ENROLLMENT TYPE ANALYSIS
# ============================================================

section("14. ENROLLMENT TYPE ANALYSIS")

if "enrollment_type" in df.columns and "enrollment" in df.columns:
    enroll_n = pd.to_numeric(df["enrollment"], errors="coerce")

    print(f"  {'Type':<15}  {'Count':>6}  {'Pct':>6}  "
          f"{'Median Enroll':>14}  {'Mean Enroll':>12}")
    print("  " + "-" * 58)

    for et, grp in df.groupby("enrollment_type", dropna=True):
        ev   = pd.to_numeric(grp["enrollment"], errors="coerce").dropna()
        cnt  = len(grp)
        pct  = 100 * cnt / len(df)
        med  = ev.median() if len(ev) > 0 else np.nan
        mean = ev.mean()   if len(ev) > 0 else np.nan
        print(f"  {et:<15}  {cnt:>6}  {pct:>5.1f}%  "
              f"{med:>14.0f}  {mean:>12.0f}")

    missing_et = int(df["enrollment_type"].isna().sum())
    print(f"  {'(missing)':<15}  {missing_et:>6}  "
          f"{100*missing_et/len(df):>5.1f}%")

    print("""
  ACTUAL    = verified count of patients enrolled at data extraction.
  ESTIMATED = target enrollment set at trial registration.

  Key insight: ACTUAL trials have lower median enrollment (43 vs 70),
  suggesting that many larger ESTIMATED trials are still active.
  Enroll type is a useful downstream feature for imputation decisions.
""")


# ============================================================
# SECTION 15 — WEIGHTED HEALTH SCORE
# ============================================================

section("15. DATASET QUALITY HEALTH SCORE")

# Dimension A — Completeness (max 40 pts)
# Every 1% average missing costs 2 points
avg_missing = quality_report["missing_%"].mean()
score_completeness = max(40 - avg_missing * 2, 0)

# Dimension B — Deduplication (max 20 pts)
score_dedup = max(20 - df.duplicated().sum() * 2, 0)

# Dimension C — Date integrity (max 15 pts)
score_dates = max(15 - parse_failures_total * 2, 0)

# Dimension D — ID uniqueness (max 15 pts)
if "nct_id" in df.columns:
    id_dup_pct      = df["nct_id"].duplicated().sum() / len(df) * 100
    score_id_unique = max(15 - id_dup_pct * 3, 0)
else:
    score_id_unique = 0

# Dimension E — Encoding integrity (max 10 pts)
score_encoding = max(10 - len(encoding_summary) * 2, 0)

scores = {
    "Completeness"  : (score_completeness, 40),
    "Deduplication" : (score_dedup,        20),
    "Date integrity": (score_dates,        15),
    "ID uniqueness" : (score_id_unique,    15),
    "Encoding"      : (score_encoding,     10),
}
total_score = sum(v for v, _ in scores.values())

print(f"\n  {'Dimension':<20}  {'Score':>6}  {'Max':>5}  {'Pct':>6}  Rating")
print("  " + "-" * 54)
for dim, (score, mx) in scores.items():
    pct    = 100 * score / mx
    rating = ("Excellent" if pct >= 90 else
              ("Good"     if pct >= 75 else
               ("Fair"    if pct >= 60 else "Needs Work")))
    print(f"  {dim:<20}  {score:>6.1f}  {mx:>5}  {pct:>5.0f}%  {rating}")
print("  " + "-" * 54)
print(f"  {'TOTAL':<20}  {total_score:>6.1f}  {'100':>5}")

print(f"\n  Dataset Quality Score: {total_score:.1f}/100")
if   total_score >= 85: print("  -> GOOD   : Suitable for analysis with minor caveats.")
elif total_score >= 70: print("  -> FAIR   : Usable but quality issues need monitoring.")
else:                   print("  -> POOR   : Significant issues — investigate before analysis.")


# ============================================================
# SECTION 16 — CLINICAL CONSISTENCY CHECKS
# ============================================================

section("16. CLINICAL CONSISTENCY CHECKS")

# Check 1: COMPLETED trials missing completion date
if "recruitment_status" in df.columns and "completion_date" in df.columns:
    n = int(((df["recruitment_status"] == "COMPLETED")
             & df["completion_date"].isna()).sum())
    print(f"  Completed trials missing completion date  : {n}")

# Check 2: COMPLETED trials with zero/missing enrollment
if "enrollment" in df.columns:
    enroll_num = pd.to_numeric(df["enrollment"], errors="coerce")
    n2 = int(((df["recruitment_status"] == "COMPLETED")
              & (enroll_num.fillna(0) <= 0)).sum())
    print(f"  Completed trials with zero/null enrollment: {n2}")

# Check 3: Missing phase
print(f"  Trials missing phase label               : "
      f"{df['phase'].isna().sum() if 'phase' in df.columns else 'N/A'}")

# Check 4: Missing primary completion date
print(f"  Trials missing primary completion date   : "
      f"{df['primary_completion_date'].isna().sum() if 'primary_completion_date' in df.columns else 'N/A'}")

# Check 5: Missing enrollment
print(f"  Trials with missing enrollment           : "
      f"{enroll_num.isna().sum() if 'enrollment' in df.columns else 'N/A'}")

print(f"\n  INTERPRETATION:")
print(f"  All checks above are expected in real-world registry data.")
print(f"  They will be resolved in Part 1B via imputation flags.")


# ============================================================
# SECTION 17 — QUALITY SUMMARY
# ============================================================

section("17. QUALITY SUMMARY")

print(f"  Total trials                   : {len(df):,}")
print(f"  Total columns                  : {len(df.columns)}")
print(f"  Exact duplicate rows           : {df.duplicated().sum()}")
print(f"  Duplicate nct_ids              : "
      f"{df['nct_id'].duplicated().sum() if 'nct_id' in df.columns else 'N/A'}")
print(f"  Columns below 95% completeness : {len(low_complete)}")
print(f"  Date parse failures            : {parse_failures_total}")
print(f"  Columns with encoding issues   : {len(encoding_summary)}")
print(f"  Mean field completeness        : {mean_complete:.1f}%")
print(f"  Dataset quality score          : {total_score:.1f}/100")

print(f"\n  KEY ISSUES TO RESOLVE IN PART 1B:")
issues = []
if len(low_complete) > 0:
    issues.append(
        f"  1. {len(low_complete)} columns below 95% complete "
        f"-> {list(low_complete.index)}")
if len(encoding_summary) > 0:
    issues.append(
        f"  2. {len(encoding_summary)} column(s) with mojibake "
        f"-> re-encode from latin-1")
issues.append(
    f"  {len(issues)+1}. 7 list-columns need ast.literal_eval() parsing "
    f"-> normalise to bridge tables")
issues.append(
    f"  {len(issues)+1}. Phase labels (PHASE1, PHASE1/PHASE2 ...) "
    f"-> standardise to Phase 1, Phase 1/2")
issues.append(
    f"  {len(issues)+1}. Status labels (ALL CAPS) "
    f"-> map to mixed-case canonical values")
for issue in issues:
    print(issue)


# 1. Main quality report
quality_report.to_csv(OUT / "data_quality_report.csv")
print(f"  data_quality_report.csv        ({len(quality_report)} rows)")

# 2. Phase distribution
if not phase_distribution.empty:
    phase_distribution.to_csv(OUT / "phase_distribution.csv", index=False)
    print(f"  phase_distribution.csv         ({len(phase_distribution)} rows)")

# 3. Status distribution
if not status_distribution.empty:
    status_distribution.to_csv(OUT / "status_distribution.csv", index=False)
    print(f"  status_distribution.csv        ({len(status_distribution)} rows)")

# 4. List-column depth
depth_df.to_csv(OUT / "list_column_depth.csv", index=False)
print(f"  list_column_depth.csv          ({len(depth_df)} rows)")

# 5. Encoding issues (only if any found)
if encoding_summary:
    pd.DataFrame(encoding_summary).to_csv(
        OUT / "encoding_issues.csv", index=False)
    print(f"  encoding_issues.csv            ({len(encoding_summary)} rows)")

print(f"\n  All outputs saved to: {OUT.resolve()}")

print("\n" + "=" * 70)
print("PART 1A — DATA QUALITY REPORT COMPLETE")
print(f"  Score: {total_score:.1f}/100  |  "
      f"Trials: {len(df):,}  |  "
      f"Columns: {len(df.columns)}")
print("=" * 70)