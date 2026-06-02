"""
PART 1B -- CLEAN ANALYTICAL SCHEMA DESIGN & IMPLEMENTATION
Clinical Trials Dataset  |  1,000 Interventional Oncology Trials

"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

FILE       = (
    "/Users/khusbuagarwal/Downloads/SampleDateExtract.xlsx - 1000_inteventional_trials.csv"
)
OUTPUT_DIR = Path("/Users/khusbuagarwal/Downloads/I3_health")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_eval(val) -> list:
    """Parse a Python list-literal string into a real list."""
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return []


def fix_encoding(text: str) -> str:
    """
    Repair mojibake: latin-1 bytes misread as UTF-8.
    Example: 'PDGFRî±' -> 'PDGFRalpha'  (actually 'PDGFRα')
    """
    if not isinstance(text, str) or "Î" not in text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except Exception:
        return text


def flatten_nested(val) -> list:
    """
    Flatten list-of-lists one level deep and apply encoding fix.
    "[['Antibody'], ['Small Molecule']]" -> ['Antibody', 'Small Molecule']
    """
    parsed = safe_eval(val)
    result = []
    for item in parsed:
        if isinstance(item, list):
            for x in item:
                if x:
                    result.append(fix_encoding(str(x)))
        elif item:
            result.append(fix_encoding(str(item)))
    return result


# ============================================================
# SECTION 1 -- LOAD
# ============================================================

section("1. LOAD RAW DATA")

df = pd.read_csv(FILE)
print(f"  Loaded : {len(df):,} rows  x  {len(df.columns)} columns")
print(f"  Columns: {df.columns.tolist()}")


# ============================================================
# SECTION 2 -- CONTROLLED VOCABULARY MAPS
# ============================================================

section("2. CONTROLLED VOCABULARY MAPS")

# ----------------------------------------------------------
# 2A. Phase -> human-readable label
# ----------------------------------------------------------
PHASE_MAP = {
    "EARLY_PHASE1":  "Phase 1",    # FDA EARLY_PHASE1 = sub-category of Phase 1
    "PHASE1":        "Phase 1",
    "PHASE1/PHASE2": "Phase 1/2",
    "PHASE2":        "Phase 2",
    "PHASE2/PHASE3": "Phase 2/3",
    "PHASE3":        "Phase 3",
    "PHASE4":        "Phase 4",
}

PHASE_INT_MAP = {
    "Phase 1":   1.0,
    "Phase 1/2": 1.5,
    "Phase 2":   2.0,
    "Phase 2/3": 2.5,
    "Phase 3":   3.0,
    "Phase 4":   4.0,
}

# ----------------------------------------------------------
# 2B. Recruitment status -> canonical label
# ----------------------------------------------------------
STATUS_MAP = {
    "COMPLETED":               "Completed",
    "RECRUITING":              "Recruiting",
    "ENROLLING_BY_INVITATION": "Recruiting",
    "ACTIVE_NOT_RECRUITING":   "Active, Not Recruiting",
    "NOT_YET_RECRUITING":      "Not Yet Recruiting",
    "TERMINATED":              "Terminated",
    "WITHDRAWN":               "Withdrawn",
    "SUSPENDED":               "Suspended",
    "UNKNOWN":                 "Unknown",
}

# ----------------------------------------------------------
# 2C. Indication -> canonical oncology group
#     42 entries; lookup is .lower().strip() so casing never breaks it
# ----------------------------------------------------------
INDICATION_MAP = {
    # Breast
    "breast cancer":                           "Breast Cancer",
    "breast neoplasm":                         "Breast Cancer",
    # Lung
    "non-small cell lung cancer":              "NSCLC",
    "thoracic neoplasm":                       "NSCLC",
    "lung cancer":                             "NSCLC",
    "lung neoplasm":                           "NSCLC",
    # Colorectal
    "colorectal cancer":                       "Colorectal Cancer",
    "colon cancer":                            "Colorectal Cancer",
    "colon carcinoma":                         "Colorectal Cancer",
    "rectal cancer":                           "Colorectal Cancer",
    "anorectal cancer":                        "Colorectal Cancer",
    "small intestinal adenocarcinoma":         "Colorectal Cancer",
    # Pancreatic / Hepatic / GI
    "pancreatic cancer":                       "Pancreatic Cancer",
    "digestive system neoplasm":               "Pancreatic Cancer",
    "hepatocellular carcinoma":                "Hepatocellular Carcinoma",
    "liver cancer":                            "Hepatocellular Carcinoma",
    "esophageal cancer":                       "Gastric Cancer",
    "gastric cancer":                          "Gastric Cancer",
    "gastroesophageal junction cancer":        "Gastric Cancer",
    # Blood cancers
    "acute myeloid leukemia":                  "AML",
    "multiple myeloma":                        "Multiple Myeloma",
    "leukemia":                                "Leukemia",
    "acute lymphoblastic leukemia":            "Leukemia",
    "chronic lymphocytic leukemia":            "Leukemia",
    "non-hodgkin lymphoma":                    "Lymphoma",
    "diffuse large b-cell lymphoma":           "Lymphoma",
    "follicular lymphoma":                     "Lymphoma",
    "hodgkin lymphoma":                        "Lymphoma",
    "hematology":                              "Hematologic Malignancies",
    "myelodysplastic syndrome":                "Hematologic Malignancies",
    "myeloproliferative neoplasm":             "Hematologic Malignancies",
    # GU / GYN
    "prostate cancer":                         "Prostate Cancer",
    "male reproductive system neoplasm":       "Prostate Cancer",
    "ovarian cancer":                          "Ovarian Cancer",
    "epithelial neoplasm":                     "Ovarian Cancer",
    "cervical cancer":                         "Ovarian Cancer",
    "female reproductive system neoplasm":     "Ovarian Cancer",
    # Head & Neck
    "head and neck neoplasm":                  "Head & Neck Cancer",
    "head and neck squamous cell carcinoma":   "Head & Neck Cancer",
    "nasopharyngeal cancer":                   "Head & Neck Cancer",
    # CNS
    "glioblastoma":                            "CNS Tumors",
    "glioma":                                  "CNS Tumors",
    "central nervous system neoplasm":         "CNS Tumors",
    "brain metastasis":                        "CNS Tumors",
    # Skin
    "skin melanoma":                           "Melanoma",
    "uveal melanoma":                          "Melanoma",
    "melanoma":                                "Melanoma",
    # Bladder / Renal
    "bladder cancer":                          "Bladder/Renal Cancer",
    "renal cell carcinoma":                    "Bladder/Renal Cancer",
    # Solid tumors (broad)
    "solid tumors":                            "Solid Tumors (NOS)",
    "advanced solid tumor":                    "Solid Tumors (NOS)",
    "refractory solid tumor":                  "Solid Tumors (NOS)",
    "cancer":                                  "Solid Tumors (NOS)",
    "neoplasms":                               "Solid Tumors (NOS)",
}

# Priority order when a trial matches multiple groups
INDICATION_PRIORITY = [
    "Breast Cancer", "NSCLC", "Colorectal Cancer",
    "Pancreatic Cancer", "Hepatocellular Carcinoma",
    "AML", "Multiple Myeloma", "Prostate Cancer", "Ovarian Cancer",
    "Gastric Cancer", "Lymphoma", "Leukemia",
    "Head & Neck Cancer", "CNS Tumors", "Melanoma",
    "Bladder/Renal Cancer", "Hematologic Malignancies",
    "Solid Tumors (NOS)",
]

def map_indication(ind_list: list) -> str:
    """Map a list of raw indications to one canonical group."""
    found = set()
    for ind in ind_list:
        group = INDICATION_MAP.get(str(ind).lower().strip())
        if group:
            found.add(group)
    for priority in INDICATION_PRIORITY:
        if priority in found:
            return priority
    return "Other"


# ----------------------------------------------------------
# 2D. Technology -> canonical tech group
#     Comprehensive map: ALL 35 raw values covered
#     Returns 'Other' explicitly for anything unrecognised
# ----------------------------------------------------------
TECH_MAP = {
    # Antibody classes
    "antibody":                                              "Monoclonal Antibody",
    "monoclonal antibody":                                   "Monoclonal Antibody",
    "bispecific antibody":                                   "Bispecific Antibody",
    "bispecific t-cell engager":                             "Bispecific Antibody",
    "tetraspecific antibody":                                "Bispecific Antibody",
    "antibody drug conjugate (adc)":                         "ADC",
    "antibody fragment drug conjugate":                      "ADC",
    "antibody radiopharmaceutical":                          "Radiopharmaceutical",
    # Small molecules
    "small molecule":                                        "Small Molecule",
    "molecular glue":                                        "Small Molecule",
    "hormone":                                               "Small Molecule",
    # Cell therapies
    "chimeric antigen receptor t-cell therapy (car-t)":      "CAR-T",
    "t-cell chimeric antigen receptor (car) therapy":        "CAR-T",
    "chimeric antigen receptor gamma delta":                 "CAR-T",
    "t-cell therapy":                                        "Cell Therapy",
    "cell therapy":                                          "Cell Therapy",
    "tumor inflitrating lymphocyte therapy (til)":           "Cell Therapy",
    "mesenchymal stem cell therapy (mscs)":                  "Cell Therapy",
    "live biotherapeutics":                                  "Cell Therapy",
    # Vaccines
    "cancer vaccine":                                        "Vaccine",
    "subunit vaccine":                                       "Vaccine",
    "whole tumor cell vaccine":                              "Vaccine",
    "dendritic cell vaccine":                                "Vaccine",
    "dna vaccine":                                           "Vaccine",
    "vaccine adjuvant":                                      "Vaccine",
    # Protein / fusion therapies
    "engineered protein therapy":                            "Protein Therapy",
    "other protein therapy":                                 "Protein Therapy",
    "fusion protein":                                        "Protein Therapy",
    "interleukin":                                           "Protein Therapy",
    "colony-stimulating factor":                             "Protein Therapy",
    "cytokine fusion protein (immunocytokine)":              "Protein Therapy",
    "toxin fusion protein (immunotoxin)":                    "Protein Therapy",
    "synthetic peptide":                                     "Protein Therapy",
    # Nucleic acid / gene therapy
    "gene therapy":                                          "Gene Therapy",
    "gene transfer therapy":                                 "Gene Therapy",
    "mrna":                                                  "Gene Therapy",
    "small interfering rna (sirna)":                         "Gene Therapy",
    "antisense oligonucleotide (aso)":                       "Gene Therapy",
    "dna oligonucleotide therapy":                           "Gene Therapy",
    "oligonucleotide therapy":                               "Gene Therapy",
    # Nuclear medicine / imaging
    "radiopharmaceutical imaging":                           "Radiopharmaceutical",
    "radiopharmaceutical":                                   "Radiopharmaceutical",
    "radiopharmaceutical therapy":                           "Radiopharmaceutical",
    "imaging agent":                                         "Radiopharmaceutical",
    # Bacteria / oncolytic
    "oncolytic bacteria":                                    "Cell Therapy",
}

def map_tech(tech_list: list) -> str:
    """
    Map a list of technology strings to one canonical group.
    BUG FIX: now returns 'Other' explicitly for unrecognised values
    instead of leaking raw strings into tech_group.
    """
    for t in tech_list:
        group = TECH_MAP.get(str(t).lower().strip())
        if group:
            return group
    # Nothing matched -- return 'Other', not the raw value
    return "Other"


print("  PHASE_MAP         : 7 entries")
print("  STATUS_MAP        : 9 entries")
print(f"  INDICATION_MAP    : {len(INDICATION_MAP)} entries -> {len(INDICATION_PRIORITY)} canonical groups")
print(f"  TECH_MAP          : {len(TECH_MAP)} entries -> 10 canonical groups")


# ============================================================
# SECTION 3 -- PARSE LIST COLUMNS
# ============================================================

section("3. PARSE LIST COLUMNS")

df["_ind_list"]  = df["indications"].apply(safe_eval)
df["_drug_list"] = df["interventions_drugs"].apply(safe_eval)
df["_tech_flat"] = df["specific_technologies"].apply(flatten_nested)
df["_tgt_flat"]  = df["target_abbreviations"].apply(flatten_nested)

print("  indications         -> _ind_list  (flat list)")
print("  interventions_drugs -> _drug_list (flat list)")
print("  specific_technologies -> _tech_flat (flattened, encoding fixed)")
print("  target_abbreviations  -> _tgt_flat  (flattened, encoding fixed)")

# Spot-check encoding fix
sample_bad = df["target_abbreviations"].dropna().head(200)
still_bad  = sum("Î" in str(v) for v in sample_bad)
print(f"\n  Mojibake check on first 200 rows: {still_bad} remaining (should be 0)")


# ============================================================
# SECTION 4 -- APPLY TRANSFORMATIONS
# ============================================================

section("4. APPLY TRANSFORMATIONS")

# 4A. Phase
df["phase_clean"] = df["phase"].map(PHASE_MAP).fillna("Unknown")
df["phase_int"]   = df["phase_clean"].map(PHASE_INT_MAP)   # NaN for Unknown
print("  phase_clean, phase_int ✓")

# 4B. Status
df["status_clean"] = df["recruitment_status"].map(STATUS_MAP).fillna("Unknown")
print("  status_clean ✓")

# 4C. Date parsing (DD/MM/YYYY format in source)
for col in ["start_date", "completion_date", "primary_completion_date"]:
    df[col + "_dt"] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
print("  *_dt date columns ✓")

# 4D. Derived numeric fields
df["start_year"]          = df["start_date_dt"].dt.year.astype("Int64")
df["trial_duration_days"] = (
    df["completion_date_dt"] - df["start_date_dt"]).dt.days
df["enrollment_n"]        = pd.to_numeric(
    df["enrollment"], errors="coerce").astype("Int64")
print("  start_year, trial_duration_days, enrollment_n ✓")

# 4E. Drug / therapy derived fields
df["n_drugs"]     = df["_drug_list"].apply(len)
df["n_targets"]   = df["_tgt_flat"].apply(len)            # NEW: target count

# BUG FIX: 0-drug trials should NOT be labelled Monotherapy
def assign_therapy_type(n: int) -> str:
    if n == 0:   return "No Drug / Biomarker"
    elif n == 1: return "Monotherapy"
    else:        return "Combination"

df["therapy_type"]   = df["n_drugs"].apply(assign_therapy_type)
df["is_combination"] = df["n_drugs"] > 1
print("  n_drugs, n_targets, therapy_type, is_combination ✓")

# 4F. Indication & technology groups
df["indication_group"] = df["_ind_list"].apply(map_indication)
df["tech_group"]       = df["_tech_flat"].apply(map_tech)
print("  indication_group, tech_group ✓")

# 4G. Duration bucket
df["duration_bucket"] = pd.cut(
    df["trial_duration_days"],
    bins=[-1, 365, 730, 1095, 99999],
    labels=["< 1 Year", "1-2 Years", "2-3 Years", "> 3 Years"]
)
print("  duration_bucket ✓")

# 4H. Outcome (tiered proxy label)
#   Positive  : Completed
#   Negative  : Terminated / Withdrawn / Suspended
#   Censored  : Recruiting / Active / Not Yet Recruiting
#   Unknown   : Unknown (registry not updated)

def assign_outcome(status: str) -> str:
    if status == "Completed":
        return "Positive"
    elif status in ("Terminated", "Withdrawn", "Suspended"):
        return "Negative"
    elif status in ("Recruiting", "Active, Not Recruiting",
                    "Not Yet Recruiting"):
        return "Censored"
    else:
        return "Unknown"

df["outcome"]        = df["status_clean"].apply(assign_outcome)
df["outcome_binary"] = np.where(df["outcome"] == "Positive", 1,
                        np.where(df["outcome"] == "Negative", 0, np.nan))
print("  outcome (tiered), outcome_binary (1/0/NaN) ✓")

# 4I. Quality flags
df["flag_no_dates"]    = df[["start_date_dt", "completion_date_dt",
                              "primary_completion_date_dt"]].isna().all(axis=1)
df["flag_zero_enroll"] = df["enrollment_n"].fillna(0) <= 0
df["flag_no_phase"]    = df["phase"].isna()
print("  flag_no_dates, flag_zero_enroll, flag_no_phase ✓")


# ============================================================
# SECTION 5 -- BUILD MASTER TRIALS TABLE
# ============================================================

section("5. BUILD TRIALS_CLEAN")

TRIALS_COLS = [
    "ID-datalake",
    "nct_id",
    "brief_title",
    "official_title",
    "phase",
    "phase_clean",
    "phase_int",
    "recruitment_status",
    "status_clean",
    "outcome",
    "outcome_binary",
    "start_date_dt",
    "completion_date_dt",
    "primary_completion_date_dt",
    "start_year",
    "trial_duration_days",
    "duration_bucket",
    "enrollment_n",
    "enrollment_type",
    "n_drugs",
    "n_targets",             # NEW column
    "therapy_type",
    "is_combination",
    "indication_group",
    "tech_group",
    "flag_no_dates",
    "flag_zero_enroll",
    "flag_no_phase",
]

trials_clean = df[TRIALS_COLS].copy()
trials_clean = trials_clean.rename(columns={"ID-datalake": "trial_id"})

print(f"  trials_clean : {trials_clean.shape[0]} rows  x  {trials_clean.shape[1]} columns")
print(f"  Columns: {trials_clean.columns.tolist()}")


# ============================================================
# SECTION 6 -- BRIDGE TABLE: trial_indications
# ============================================================

section("6. BRIDGE TABLE: trial_indications")

ind_rows = []
for _, row in df.iterrows():
    for ind in row["_ind_list"]:
        group = INDICATION_MAP.get(str(ind).lower().strip(), "Other")
        ind_rows.append({
            "nct_id":           row["nct_id"],
            "indication_raw":   str(ind),
            "indication_group": group,
        })

trial_indications = pd.DataFrame(ind_rows)
print(f"  trial_indications : {len(trial_indications)} rows")
print(f"  Unique nct_ids    : {trial_indications['nct_id'].nunique()}")
print(f"  Avg per trial     : {len(trial_indications)/trial_indications['nct_id'].nunique():.1f}")


# ============================================================
# SECTION 7 -- BRIDGE TABLE: trial_drugs
# ============================================================

section("7. BRIDGE TABLE: trial_drugs")

drug_rows = []
for _, row in df.iterrows():
    drugs    = row["_drug_list"]
    n        = len(drugs)
    for drug in drugs:
        drug_rows.append({
            "nct_id":           row["nct_id"],
            "drug_name":        str(drug),
            "is_combination":   n > 1,
            "n_drugs_in_trial": n,
        })

trial_drugs = pd.DataFrame(drug_rows)
print(f"  trial_drugs       : {len(trial_drugs)} rows")
print(f"  Unique nct_ids    : {trial_drugs['nct_id'].nunique()}")
print(f"  Avg drugs/trial   : {len(trial_drugs)/trial_drugs['nct_id'].nunique():.1f}")
print(f"  Null drug names   : {trial_drugs['drug_name'].isna().sum()}")
print(f"\n  Top 10 most common drugs:")
top10 = trial_drugs["drug_name"].value_counts().head(10)
for drug, cnt in top10.items():
    print(f"    {drug:<40}  {cnt:>4}")


# ============================================================
# SECTION 8 -- BRIDGE TABLE: trial_targets
# ============================================================

section("8. BRIDGE TABLE: trial_targets")

target_rows = []
for _, row in df.iterrows():
    seen = set()   # deduplicate within one trial (defensive)
    for tgt in row["_tgt_flat"]:
        if tgt and tgt not in seen:
            seen.add(tgt)
            target_rows.append({
                "nct_id": row["nct_id"],
                "target": tgt,
            })

trial_targets = pd.DataFrame(target_rows)

# Verify encoding fix
remaining_bad = trial_targets["target"].str.contains("Î", na=False).sum()
print(f"  trial_targets     : {len(trial_targets)} rows")
print(f"  Unique nct_ids    : {trial_targets['nct_id'].nunique()}")
print(f"  Mojibake remaining: {remaining_bad}  (should be 0)")
print(f"\n  Top 15 most common targets:")
top15 = trial_targets["target"].value_counts().head(15)
for tgt, cnt in top15.items():
    print(f"    {tgt:<30}  {cnt:>4}")


# ============================================================
# SECTION 9 -- VALIDATION REPORT
# ============================================================

section("9. VALIDATION REPORT")

def null_profile(table: pd.DataFrame, name: str) -> pd.DataFrame:
    n = len(table)
    rows = []
    for col in table.columns:
        nulls = int(table[col].isna().sum())
        rows.append({
            "table":         name,
            "column":        col,
            "dtype":         str(table[col].dtype),
            "row_count":     n,
            "null_count":    nulls,
            "null_pct":      round(100 * nulls / n, 1) if n > 0 else 0.0,
            "unique_values": int(table[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)

validation_report = pd.concat([
    null_profile(trials_clean,      "trials_clean"),
    null_profile(trial_indications, "trial_indications"),
    null_profile(trial_drugs,       "trial_drugs"),
    null_profile(trial_targets,     "trial_targets"),
], ignore_index=True)

# Print only columns with nulls > 0 in trials_clean
print("  Columns with null values in trials_clean:")
tc_nulls = validation_report[
    (validation_report["table"] == "trials_clean") &
    (validation_report["null_count"] > 0)
][["column","null_count","null_pct"]].sort_values("null_pct", ascending=False)
print(tc_nulls.to_string(index=False))
print(f"\n  validation_report : {len(validation_report)} rows (all 4 tables)")


# ============================================================
# SECTION 10 -- SCHEMA SUMMARY (data dictionary)
# ============================================================

section("10. SCHEMA SUMMARY")

SCHEMA_DICT = {
    # trials_clean
    "trial_id":                      ("trials_clean", "int",      "Unique datalake identifier (from ID-datalake)"),
    "nct_id":                        ("trials_clean", "str",      "ClinicalTrials.gov registry ID (primary key)"),
    "brief_title":                   ("trials_clean", "str",      "Short study title"),
    "official_title":                ("trials_clean", "str",      "Full official study title (6 nulls)"),
    "phase":                         ("trials_clean", "str",      "Raw phase label e.g. PHASE2 (40 nulls)"),
    "phase_clean":                   ("trials_clean", "str",      "Canonical label e.g. Phase 2 / Phase 1/2"),
    "phase_int":                     ("trials_clean", "float",    "Numeric ordinal: 1.0/1.5/2.0/2.5/3.0/4.0 (40 nulls)"),
    "recruitment_status":            ("trials_clean", "str",      "Raw ALL-CAPS status e.g. COMPLETED"),
    "status_clean":                  ("trials_clean", "str",      "Canonical mixed-case status e.g. Completed"),
    "outcome":                       ("trials_clean", "str",      "Proxy label: Positive/Negative/Censored/Unknown"),
    "outcome_binary":                ("trials_clean", "float",    "1=Completed, 0=Negative, NaN=Censored/Unknown"),
    "start_date_dt":                 ("trials_clean", "datetime", "Parsed start date (5 nulls)"),
    "completion_date_dt":            ("trials_clean", "datetime", "Parsed completion date (52 nulls)"),
    "primary_completion_date_dt":    ("trials_clean", "datetime", "Parsed primary completion date (51 nulls)"),
    "start_year":                    ("trials_clean", "int",      "Year extracted from start_date"),
    "trial_duration_days":           ("trials_clean", "int",      "Days from start to completion (54 nulls)"),
    "duration_bucket":               ("trials_clean", "category", "<1yr / 1-2yr / 2-3yr / >3yr"),
    "enrollment_n":                  ("trials_clean", "int",      "Enrollment count as integer (26 nulls)"),
    "enrollment_type":               ("trials_clean", "str",      "ACTUAL or ESTIMATED (44 nulls)"),
    "n_drugs":                       ("trials_clean", "int",      "Number of drugs in the trial"),
    "n_targets":                     ("trials_clean", "int",      "Number of molecular targets in the trial"),
    "therapy_type":                  ("trials_clean", "str",      "Monotherapy / Combination / No Drug / Biomarker"),
    "is_combination":                ("trials_clean", "bool",     "True if >=2 drugs"),
    "indication_group":              ("trials_clean", "str",      "Canonical oncology indication (18 groups + Other)"),
    "tech_group":                    ("trials_clean", "str",      "Canonical technology class (10 groups + Other)"),
    "flag_no_dates":                 ("trials_clean", "bool",     "True if all date fields are null"),
    "flag_zero_enroll":              ("trials_clean", "bool",     "True if enrollment = 0 or missing"),
    "flag_no_phase":                 ("trials_clean", "bool",     "True if phase is null"),
    # bridge tables
    "indication_raw":                ("trial_indications", "str", "Raw indication string from source data"),
    "indication_group (bridge)":     ("trial_indications", "str", "Canonical indication group"),
    "drug_name":                     ("trial_drugs",  "str",      "Individual drug name"),
    "n_drugs_in_trial":              ("trial_drugs",  "int",      "Total drugs in the parent trial"),
    "target":                        ("trial_targets","str",      "Molecular target abbreviation (encoding-fixed)"),
}

schema_summary = pd.DataFrame(
    [(col, tbl, dtype, desc)
     for col, (tbl, dtype, desc) in SCHEMA_DICT.items()],
    columns=["column", "table", "dtype", "description"]
)
print(f"  schema_summary : {len(schema_summary)} rows (full data dictionary)")
print(schema_summary.to_string(index=False))


# ============================================================
# SECTION 11 -- DISTRIBUTION CHECKS
# ============================================================

section("11. DISTRIBUTION CHECKS ON DERIVED COLUMNS")

checks = {
    "[A] phase_clean":      trials_clean["phase_clean"],
    "[B] status_clean":     trials_clean["status_clean"],
    "[C] outcome":          trials_clean["outcome"],
    "[D] indication_group": trials_clean["indication_group"],
    "[E] tech_group":       trials_clean["tech_group"],
    "[F] therapy_type":     trials_clean["therapy_type"],
    "[G] duration_bucket":  trials_clean["duration_bucket"],
}

for label, series in checks.items():
    print(f"\n  {label}:")
    vc = series.value_counts(dropna=False)
    for val, cnt in vc.items():
        pct = 100 * cnt / len(trials_clean)
        print(f"    {cnt:5d}  ({pct:5.1f}%)  {val}")

print("\n  [H] outcome_binary:")
print(f"    {int((trials_clean['outcome_binary']==1).sum()):5d}  Positive  (1)")
print(f"    {int((trials_clean['outcome_binary']==0).sum()):5d}  Negative  (0)")
print(f"    {int(trials_clean['outcome_binary'].isna().sum()):5d}  Censored  (NaN)")

print("\n  [I] n_drugs distribution:")
vc_drugs = trials_clean["n_drugs"].value_counts().sort_index()
for val, cnt in vc_drugs.items():
    bar = chr(9608) * min(int(cnt / 5), 40)
    print(f"    {val:>3} drug(s): {cnt:4d}  {bar}")

print("\n  [J] start_year distribution:")
yr = trials_clean["start_year"].value_counts().sort_index().dropna()
for year, cnt in yr.items():
    bar = chr(9608) * min(int(cnt / 3), 40)
    print(f"    {int(year)}: {cnt:4d}  {bar}")


# ============================================================
# SECTION 12 -- QUALITY GATE
# ============================================================

section("12. SCHEMA QUALITY GATE")

ASSERTIONS = {
    "No duplicate nct_id in trials_clean":
        trials_clean["nct_id"].duplicated().sum() == 0,

    "outcome_binary only contains 1, 0, NaN":
        trials_clean["outcome_binary"].dropna().isin([0, 1]).all(),

    "phase_int range: 1.0 <= x <= 4.0":
        trials_clean["phase_int"].dropna().between(1, 4).all(),

    "No mojibake in trial_targets":
        (~trial_targets["target"].str.contains("Î", na=False)).all(),

    "trial_drugs nct_ids all in trials_clean":
        trial_drugs["nct_id"].isin(trials_clean["nct_id"]).all(),

    "trial_targets nct_ids all in trials_clean":
        trial_targets["nct_id"].isin(trials_clean["nct_id"]).all(),

    "trial_indications nct_ids all in trials_clean":
        trial_indications["nct_id"].isin(trials_clean["nct_id"]).all(),

    "No outcome_binary=1 for non-Completed trial":
        len(trials_clean[
            (trials_clean["outcome_binary"] == 1) &
            (trials_clean["status_clean"] != "Completed")
        ]) == 0,

    "therapy_type never Monotherapy when n_drugs=0 (bug fix check)":
        len(trials_clean[
            (trials_clean["n_drugs"] == 0) &
            (trials_clean["therapy_type"] == "Monotherapy")
        ]) == 0,

    "tech_group has no raw leaked strings (bug fix check)":
        # All tech_group values should be in our canonical set
        trials_clean["tech_group"].isin([
            "Monoclonal Antibody", "Small Molecule", "ADC", "CAR-T",
            "Bispecific Antibody", "Vaccine", "Protein Therapy",
            "Radiopharmaceutical", "Gene Therapy", "Cell Therapy", "Other"
        ]).all(),
}

all_passed = True
for assertion, result in ASSERTIONS.items():
    icon = "✅" if result else "❌"
    print(f"  {icon}  {assertion}")
    if not result:
        all_passed = False

print()
print(f"  {'✅  ALL 10 ASSERTIONS PASSED' if all_passed else '❌  SOME ASSERTIONS FAILED'}")


# ============================================================
# SECTION 13 -- SAVE ALL OUTPUTS
# ============================================================

section("13. SAVE ALL OUTPUTS")

OUTPUTS = {
    "trials_clean.csv":       trials_clean,
    "trial_indications.csv":  trial_indications,
    "trial_drugs.csv":        trial_drugs,
    "trial_targets.csv":      trial_targets,
    "validation_report.csv":  validation_report,
    "schema_summary.csv":     schema_summary,
}

for filename, table in OUTPUTS.items():
    path = OUTPUT_DIR / filename
    table.to_csv(path, index=False)
    print(f"  ✅  {filename:<30}  {len(table):>6,} rows  ->  {path}")


# ============================================================
# SECTION 14 -- FINAL SUMMARY
# ============================================================

section("14. PART 1B COMPLETE")

resolved      = int((trials_clean["outcome_binary"]==1).sum()
                    + (trials_clean["outcome_binary"]==0).sum())
success_rate  = (trials_clean["outcome_binary"]==1).sum() / resolved
other_ind_pct = round(100*(trials_clean["indication_group"]=="Other").mean(), 1)
other_tec_pct = round(100*(trials_clean["tech_group"]=="Other").mean(), 1)

print(f"""
  Tables created          :  6
    trials_clean          :  {len(trials_clean):,} rows  x  {trials_clean.shape[1]} cols
    trial_indications     :  {len(trial_indications):,} rows
    trial_drugs           :  {len(trial_drugs):,} rows
    trial_targets         :  {len(trial_targets):,} rows
    validation_report     :  {len(validation_report):,} rows
    schema_summary        :  {len(schema_summary):,} rows (data dictionary)

  Derived columns added   :  phase_clean, phase_int, status_clean,
                             outcome, outcome_binary, indication_group,
                             tech_group, n_drugs, n_targets (NEW),
                             therapy_type, is_combination, start_year,
                             trial_duration_days, duration_bucket,
                             enrollment_n + 3x quality flags

  SOME IMPORTANT FIX
    1 tech_group      :  map_tech() now returns 'Other' explicitly
                             (was leaking 35 raw values into tech_group)
    2 therapy_type    :  n_drugs=0 -> 'No Drug / Biomarker'
                             (was incorrectly labelled 'Monotherapy')
    3 indication      :  INDICATION_MAP expanded 30->42 entries
                             'Other' reduced to {other_ind_pct}%
                             (was 38.8%)

  Coverage stats:
    indication_group 'Other' : {other_ind_pct}%
    tech_group 'Other'        : {other_tec_pct}%
    Mojibake remaining        : {trial_targets['target'].str.contains('Î',na=False).sum()}

  Outcome distribution:
    Positive  (Completed)          :  {int((trials_clean['outcome_binary']==1).sum()):,}
    Negative  (Term/Withdr/Susp)   :  {int((trials_clean['outcome_binary']==0).sum()):,}
    Censored/Unknown               :  {int(trials_clean['outcome_binary'].isna().sum()):,}
    Resolved denominator           :  {resolved:,}
    Proxy success rate             :  {100*success_rate:.1f}%

""")