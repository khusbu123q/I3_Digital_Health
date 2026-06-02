# Oncology Clinical Trial Success Rate Analysis
**i3 Digital Health Technical Assessment** | Khusbu Agarwal

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.x-green.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![scipy](https://img.shields.io/badge/scipy-1.x-red.svg)](https://scipy.org/)

---

## Overview

End-to-end data engineering and analytics pipeline on **1,000 interventional oncology
clinical trials** from ClinicalTrials.gov. The dataset contains no direct success signal —
no endpoint result, no regulatory decision, no reason trials stopped. The challenge was to
define a defensible proxy metric, quantify its uncertainty, and identify what actually drives
variation in trial completion across 12 stratification dimensions.

**Core finding:** Disease biology (indication) and resource commitment (enrollment size)
are the two dominant drivers of trial completion. Phase, drug complexity, and therapy type
do not independently predict completion once those two factors are accounted for.

---

## Repository Structure

```
I3_Digital_Health/
│
├── Part_1a.py                   # Data quality report — 19 sections
├── Part_1b.py                   # Schema normalisation — 4 tables, 27 cols
├── Part_2a.py                   # Proxy metric definition & validation
├── Part_2b.py                   # 12-dimension cohort analysis
│
├── outputs/
│   ├── ── SCHEMA (Part 1B) ──────────────────────────────────────────
│   ├── trials_clean.csv                  # Master: 1,000 × 27 cols
│   ├── trial_drugs.csv                   # Bridge: 2,716 rows
│   ├── trial_targets.csv                 # Bridge: 2,674 rows
│   ├── trial_indications.csv             # Bridge: 2,231 rows
│   ├── schema_summary.csv                # Data dictionary
│   ├── validation_report.csv             # Null profiles, all 4 tables
│   │
│   ├── ── METRICS (Part 2A) ─────────────────────────────────────────
│   ├── success_rate_metrics.csv          # Core rate + all p-values
│   ├── sensitivity_analysis.csv          # 4 scenarios + spreads
│   ├── assumption_registry.csv           # A1–A7 decision log
│   ├── phase_stratified_rates.csv
│   ├── indication_stratified_rates.csv
│   ├── tech_stratified_rates.csv
│   ├── enrollment_bucket_rates.csv
│   ├── duration_bucket_rates.csv
│   ├── year_trend.csv
│   ├── ind_phase_crosstab.csv
│   │
│   ├── ── COHORT ANALYSIS (Part 2B) ─────────────────────────────────
│   ├── 2b_d1_indication.csv ... 2b_d12_duration.csv
│   ├── 2b_pairwise_effect_sizes.csv
│   ├── 2b_summary_all_dimensions.csv
│   │
│   └── ── CHARTS ────────────────────────────────────────────────────
│       ├── quality_dashboard.png          # 6-panel Part 1A chart
│       ├── 2b_stratified_dashboard.png    # 9-panel Part 2B chart
│       ├── 2b_heatmap_ind_phase.png       # Indication × Phase heatmap
│       └── 2b_enrollment_interaction.png  # Enrollment × Indication chart
│
├── .gitignore
└── README.md
```

---

## Pipeline

```
Raw CSV  (1,000 rows × 18 cols)
        │
        ▼
  Part_1a.py  ──►  Quality report · completeness · encoding audit · health score
        │
        ▼
  Part_1b.py  ──►  Normalised schema · 4 tables · 27 derived cols · quality gate
        │
        ▼
  Part_2a.py  ──►  Proxy metric · sensitivity analysis · logistic regression
        │
        ▼
  Part_2b.py  ──►  12-dimension cohort analysis · charts · rankings
```

Each script reads from the outputs of the previous one.
Parts 2A and 2B both read from `outputs/trials_clean.csv`.

---

## Setup

**Install dependencies:**
```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib
```

**Configure file paths** at the top of each script:
```python
FILE       = "/path/to/SampleDateExtract_xlsx_-_1000_inteventional_trials.csv"
OUTPUT_DIR = Path("/path/to/outputs/")
```

**Run in order:**
```bash
python Part_1a.py
python Part_1b.py
python Part_2a.py
python Part_2b.py
```

---

## Part 1A — Data Quality Report

Profiles the raw 18-column dataset before any transformation.

| Metric | Value |
|---|---|
| Mean field completeness | 98.8% |
| Columns below 95% complete | 2 (`completion_date`, `primary_completion_date`) |
| Duplicate rows / NCT IDs | 0 / 0 |
| Columns with encoding issues | 7 |
| Rows with mojibake (`target_abbreviations`) | 102 |
| Combination therapy trials | 680 (68.0%) |
| Max drugs in one trial | 21 |
| Dataset quality score | **87.5 / 100** |

Two structural issues detected:
- 7 columns store Python list-literal strings requiring `ast.literal_eval()` to parse
- Latin-1 / UTF-8 encoding mismatch causes mojibake in `target_abbreviations` (PDGFRα → PDGFRî±)

---

## Part 1B — Schema Normalisation

Transforms the flat file into a normalised schema with controlled vocabularies
and engineered features.

### Tables

| Table | Rows | Description |
|---|---|---|
| `trials_clean` | 1,000 × 27 | Master table with all derived columns |
| `trial_drugs` | 2,716 | One row per drug per trial |
| `trial_targets` | 2,674 | One row per molecular target per trial |
| `trial_indications` | 2,231 | One row per indication per trial |

### Key Derived Columns

| Column | Type | Description |
|---|---|---|
| `phase_clean` | str | Canonical label: Phase 1, Phase 1/2, Phase 2 … |
| `phase_int` | float | Numeric ordinal: 1.0 / 1.5 / 2.0 / 2.5 / 3.0 / 4.0 |
| `outcome` | str | Positive / Negative / Censored / Unknown |
| `outcome_binary` | float | 1 = Completed, 0 = Negative, NaN = Censored |
| `indication_group` | str | 18 canonical oncology groups (54-entry map) |
| `tech_group` | str | 10 canonical technology classes (45-entry map) |
| `n_drugs` | int | Number of drugs in the trial |
| `n_targets` | int | Number of molecular targets |
| `therapy_type` | str | Monotherapy / Combination / No Drug / Biomarker |
| `enrollment_n` | int | Enrollment count as integer |
| `trial_duration_days` | int | Completion − start in days |
| `flag_zero_enroll` | bool | True if enrollment = 0 or missing |

### Bugs Found and Fixed

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | `map_tech()` returned raw unmapped string | 35 raw values leaked into `tech_group` | Return `'Other'` explicitly |
| 2 | `n_drugs=0` labelled `'Monotherapy'` | 11 trials mislabelled | Three-way assignment → `'No Drug / Biomarker'` |
| 3 | `INDICATION_MAP` only 30 entries | 38.8% fell to `'Other'` | Expanded to 54 entries → Other reduced to 21.7% |

**Quality gate:** 10 assertions, all pass. Two assertions directly verify the bug fixes.

---

## Part 2A — Proxy Metric Definition

### Why a Proxy Is Needed

The dataset has **no** success flag, endpoint result, hazard ratio, regulatory decision,
or `why_stopped` field. The only available signal is `recruitment_status`.

### Outcome Mapping

| Status | Outcome | Binary | Rationale |
|---|---|---|---|
| Completed | Positive | 1 | Ran to planned protocol end |
| Terminated / Withdrawn / Suspended | Negative | 0 | Stopped before planned end |
| Recruiting / Active / Not Yet Recruiting | Censored | NaN | No outcome event yet — excluded |
| Unknown | Unknown | NaN | Cannot distinguish failed vs stale registry |

> ⚠️ **This proxy sits at Stage 2 of a 6-stage clinical pipeline.**
> Completion ≠ efficacy. Real oncology approval rates from Phase 1 are ~5–10%.

### Core Result

| Metric | Value |
|---|---|
| Positive (Completed) | 453 |
| Negative (Stopped) | 167 |
| Censored / Unknown | 380 |
| Resolved denominator | **620** |
| **Proxy success rate** | **73.1%** |
| Wilson 95% CI | [69.4%, 76.4%] |
| Bootstrap 95% CI | [69.7%, 76.5%] |

### Sensitivity Analysis

| Scenario | Rate | Δ vs Baseline |
|---|---|---|
| S1 — Baseline (Unknown excluded) | 73.1% | — |
| S2 — Suspended as Censored | 73.5% | +0.47 pp |
| S3 — Unknown as Negative (pessimistic) | 61.1% | −11.93 pp |
| S4 — Unknown as Positive (optimistic) | 77.5% | +4.40 pp |
| **Uncertainty spread** | **16.3 pp** | driven by Unknown assumption |

### Logistic Regression

Trained on n=573 resolved trials. **5-fold cross-validated AUC = 0.700 ± 0.095**

| Feature | Odds Ratio | Direction |
|---|---|---|
| `enrollment_n` | **7.13** | Strongest positive predictor |
| `phase_int` | 0.82 | Negative once enrollment controlled |
| `is_combination` | 0.89 | Marginal |

---

## Part 2B — Stratified Cohort Analysis

12 stratification dimensions tested. **5 statistically significant, 7 not.**

### All 12 Dimensions Ranked

| Rank | Dimension | Range | p-value | Significant |
|---|---|---|---|---|
| 1 | D5 Indication × Phase | 64.3 pp | n/a | — |
| 2 | **D1 Indication** | **48.1 pp** | **0.0014** | **✓** |
| 3 | D4 Target Class | 42.1 pp | 0.273 | — |
| 4 | **D11 Zero Enrollment** | **42.1 pp** | **< 0.0001** | **✓** |
| 5 | **D7 Enrollment Bucket** | **32.8 pp** | **< 0.0001** | **✓** |
| 6 | **D8 Trial Era** | **27.1 pp** | **0.0001** | **✓** |
| 7 | **D12 Duration Bucket** | **26.8 pp** | **< 0.0001** | **✓** |
| 8 | D2 Phase | 20.0 pp | 0.391 | ⚠️ not sig |
| 9 | D3 Technology | 14.8 pp | 0.087 | — |
| 10 | D9 Drug Count | 3.6 pp | 0.783 | — |
| 11 | D10 Precision vs Cytotoxic | 3.3 pp | 0.763 | — |
| 12 | D6 Therapy Type | 1.6 pp | 0.749 | — |

### Headline Findings

**Indication — only significant categorical predictor**
- Lymphoma 100.0% → Multiple Myeloma 51.9% (48.1 pp range, p = 0.0014)
- Pairwise: Lymphoma vs MM — Cohen's h = 1.53, Fisher p = 0.000145

**Phase — NOT significant (p = 0.391)**
- The apparent Phase 4 > Phase 3 > Phase 2 gradient is an enrollment artefact
- Phase 3 median enrollment = 243 pts vs Phase 1 = 24 pts
- Once enrollment controlled: phase has a negative coefficient (OR = 0.82)
- Kendall tau trend test: τ = 0.027, p = 0.475 — no monotonic trend

**Enrollment — strongest predictor (KW H = 50.41, p < 0.000001)**
- 1–20 pts: 58.0% → 51–100 pts: 90.8% (32.8 pp jump)
- Completed trials median = 45 pts; stopped trials median = 12 pts
- Enrollment × Indication: AML +71 pp (p = 0.0045), NSCLC +58 pp (p = 0.0001)

**Drug count — NOT significant (p = 0.994)**
- Range across all buckets: only 3.6 pp
- Completed and stopped trials share identical median drug count = 2.0
- Protocol complexity is not the bottleneck

### Pairwise Effect Sizes

| Comparison | Cohen's h | Effect | Fisher p |
|---|---|---|---|
| Lymphoma vs Multiple Myeloma | 1.534 | Very Large | 0.000145 |
| Lymphoma vs NSCLC | 1.302 | Large | 0.000696 |
| Colorectal vs AML | 0.948 | Large | 0.021952 |
| HER2 vs Proteasome | 1.061 | Large | 0.005454 |
| HER2 vs PD-1 | 0.780 | Medium | 0.027095 |
| TOP2 vs RNR | 0.531 | Medium | 0.038406 |

---

## Limitations

- **Stage 2 proxy only** — completion ≠ efficacy; real approval rates ~5–10%
- **Right-censoring** — 380 trials still active; 2020+ cohort shows 52% (censoring artefact)
- **Unknown exclusion** — 121 trials (12.1%) create a 16.3 pp uncertainty band
- **Missing fields** — `why_stopped` would separate futility from safety from business withdrawal

---

## Author

**Khusbu Agarwal**
MSc Big Data Biology — Institute of Bioinformatics and Applied Biotechnology (IBAB), Bangalore
Supervised by Dr. Nithya Ramakrishnan

[![LinkedIn](https://img.shields.io/badge/LinkedIn-khusbuagarwal-blue)](https://linkedin.com/in/khusbuagarwal)
[![GitHub](https://img.shields.io/badge/GitHub-khusbu123q-black)](https://github.com/khusbu123q)
📧 Khusbu1053@gmail.com
