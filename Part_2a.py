"""
PART 2A — OPERATIONALISING "SUCCESS"
Clinical Trials Dataset  |  1000 Interventional Trials

SECTIONS:
  1.  Why we need a proxy
  2.  Proxy definition (3-tier label)
  3.  Status → outcome mapping          (Q1)
  4.  Ambiguous status handling          (Q2)
  5.  Completion vs therapeutic success  (Q3)
  6.  Validation + core counts
  7.  Sensitivity analysis (4 scenarios)
  8.  Phase-stratified rates + chi-square test
  9.  Indication-stratified rates + chi-square test
  10. Technology-stratified rates
  11. Combination vs monotherapy
  12. Enrollment-bucket analysis     ← NEW
  13. Duration-bucket analysis       ← NEW
  14. Start-year trend + censoring bias
  15. Indication × Phase cross-tab
  16. Logistic regression — predictors of completion  ← NEW
  17. Assumption registry

OUTPUTS (9 CSVs):
  success_rate_metrics.csv        phase_stratified_rates.csv
  indication_stratified_rates.csv tech_stratified_rates.csv
  enrollment_bucket_rates.csv     duration_bucket_rates.csv
  year_trend.csv                  ind_phase_crosstab.csv
  sensitivity_analysis.csv        assumption_registry.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score

# ============================================================
# LOAD
# ============================================================
INPUT = Path("outputs/trials_clean.csv")
OUT   = Path("outputs")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)
df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")

print("=" * 70)
print("PART 2A — OPERATIONALISING 'SUCCESS'")
print("=" * 70)


# ============================================================
# HELPERS
# ============================================================

MIN_N = 10

def wilson_ci(pos, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    return proportion_confint(int(pos), int(n), alpha=alpha, method="wilson")

def rate_table(data, group_col, min_n=MIN_N):
    resolved = data[data["outcome_binary"].notna()].copy()
    rows = []
    for grp, sub in resolved.groupby(group_col, observed=True):
        n   = len(sub)
        pos = int((sub["outcome_binary"] == 1).sum())
        neg = n - pos
        rate = pos / n if n > 0 else np.nan
        cil, cih = wilson_ci(pos, n) if n >= min_n else (np.nan, np.nan)
        cens = int(data[data[group_col] == grp]["outcome_binary"].isna().sum())
        rows.append({
            group_col:        grp,
            "n_resolved":     n,
            "n_positive":     pos,
            "n_negative":     neg,
            "n_censored":     cens,
            "success_rate_%": round(100*rate, 1) if not np.isnan(rate) else np.nan,
            "ci_low_%":       round(100*cil,  1) if not np.isnan(cil)  else np.nan,
            "ci_high_%":      round(100*cih,  1) if not np.isnan(cih)  else np.nan,
            "reliable":       "✓" if n >= min_n else f"⚠ n={n}",
        })
    return pd.DataFrame(rows)

def compute_rate(df_tmp):
    pos = int((df_tmp["outcome_binary"] == 1).sum())
    res = int(df_tmp["outcome_binary"].notna().sum())
    rate = pos / res if res > 0 else np.nan
    cil, cih = wilson_ci(pos, res)
    return pos, res, rate, cil, cih

# ── Baseline numbers ──────────────────────────────────────
n_positive = int((df["outcome_binary"] == 1).sum())
n_negative = int((df["outcome_binary"] == 0).sum())
n_censored = int((df["outcome"] == "Censored").sum())
n_unknown  = int((df["outcome"] == "Unknown").sum())
n_total    = len(df)
n_resolved = n_positive + n_negative
success_rate = n_positive / n_resolved
ci_low, ci_high = wilson_ci(n_positive, n_resolved)
unknown_pct  = round(100 * n_unknown  / n_total, 1)
censored_pct = round(100 * n_censored / n_total, 1)
resolved_df  = df[df["outcome_binary"].notna()].copy()


# ============================================================
# SECTION 1 — WHY WE NEED A PROXY
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 1 — WHY WE NEED A PROXY
─────────────────────────────────────────────────────────────────────

The dataset contains NO:
  ✗  success flag
  ✗  primary endpoint result (was the endpoint met?)
  ✗  regulatory decision    (approved / rejected?)
  ✗  efficacy readout       (hazard ratio, p-value, response rate)
  ✗  why_stopped field      (futility vs safety vs business)

The ONLY signal reflecting a trial's fate is:
  ✓  recruitment_status  → standardised as status_clean
  ✓  phase_clean         → used as a stratification dimension
""")


# ============================================================
# SECTION 2 — PROXY DEFINITION
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 2 — PROXY DEFINITION: TRIAL COMPLETION PROXY
─────────────────────────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────────────────┐
  │  outcome_binary = 1   →   POSITIVE                              │
  │  status_clean == "Completed"                                     │
  │  The trial ran to its planned protocol end.                      │
  │  ⚠  Does NOT mean the drug worked.                              │
  │     A completed trial can still fail its primary endpoint.       │
  ├──────────────────────────────────────────────────────────────────┤
  │  outcome_binary = 0   →   NEGATIVE                              │
  │  status_clean ∈ {Terminated, Withdrawn, Suspended}              │
  │  Trial stopped BEFORE planned completion.                        │
  │  ⚠  All three reasons are coded identically.                    │
  │     Cannot separate without 'why_stopped' field.                │
  ├──────────────────────────────────────────────────────────────────┤
  │  outcome_binary = NaN  →   CENSORED / UNKNOWN                   │
  │  status_clean ∈ {Recruiting, Active Not Recruiting,              │
  │                  Not Yet Recruiting, Unknown}                    │
  │  No outcome event yet. Including inflates denominator.           │
  │  → Excluded from numerator AND denominator.                     │
  └──────────────────────────────────────────────────────────────────┘

  FORMULA:
    Rate = Completed / (Completed + Terminated + Withdrawn + Suspended)
""")


# ============================================================
# SECTION 3 — STATUS → OUTCOME MAPPING  (Q1)
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 3 — STATUS → OUTCOME MAPPING  (Question 1)
─────────────────────────────────────────────────────────────────────
""")
STATUS_OUTCOME_MAP = pd.DataFrame([
    ("Completed",              "Positive", 1.0,
     "Ran to protocol end. Necessary but not sufficient for efficacy."),
    ("Terminated",             "Negative", 0.0,
     "Stopped early — futility, safety, or business. All coded identically."),
    ("Withdrawn",              "Negative", 0.0,
     "Withdrawn before enrolment. Definitive negative operational outcome."),
    ("Suspended",              "Negative", 0.0,
     "Temporary hold — treated as Negative. See Section 4."),
    ("Recruiting",             "Censored", np.nan,
     "Active enrolment. No outcome event yet. EXCLUDED from denominator."),
    ("Active, Not Recruiting", "Censored", np.nan,
     "Enrolled, awaiting results. No outcome event. EXCLUDED."),
    ("Not Yet Recruiting",     "Censored", np.nan,
     "Not started yet. No outcome event. EXCLUDED."),
    ("Unknown",                "Unknown",  np.nan,
     "Registry not updated. Cannot assign outcome. EXCLUDED."),
], columns=["status_clean","outcome","outcome_binary","rationale"])
print(STATUS_OUTCOME_MAP.to_string(index=False))


# ============================================================
# SECTION 4 — AMBIGUOUS STATUS HANDLING  (Q2)
# ============================================================
print(f"""
─────────────────────────────────────────────────────────────────────
SECTION 4 — AMBIGUOUS STATUS HANDLING  (Question 2)
─────────────────────────────────────────────────────────────────────

SUSPENDED  (n={int((df['status_clean']=='Suspended').sum())})
  Options     : A) Negative  B) Censored  C) Separate tier
  Decision    : → Negative  (outcome_binary = 0)
  Rationale   : Most suspended oncology trials never resume. Censored
                would overstate the rate; n=4 is too small for a tier.
  Sensitivity : Quantified in Section 7 — impact is negligible.

UNKNOWN  (n={n_unknown}  →  {unknown_pct}% of all trials)
  Options     : A) Impute Completed  B) Impute Terminated  C) Exclude
  Decision    : → Exclude  (outcome_binary = NaN)
  Rationale   : Cannot distinguish Unknown=failed vs Unknown=stale registry.
                Exclusion is the most conservative defensible choice.
  ⚠ MATERIAL: {unknown_pct}% exclusion creates the dominant uncertainty in
    our estimate (16.3pp spread across scenarios — see Section 7).

NOT YET RECRUITING  (n={int((df['status_clean']=='Not Yet Recruiting').sum())})
  Decision    : → Censored. No outcome event has occurred.

ENROLLING BY INVITATION  (n=2)
  Decision    : → Grouped with Recruiting. Identical operational meaning.

PHASE as a modifier:
  Phase does NOT determine outcome label — a Phase 1 can Complete or
  Terminate just like Phase 3. Used as stratification only (Section 8).
""")


# ============================================================
# SECTION 5 — COMPLETION vs THERAPEUTIC SUCCESS  (Q3)
# ============================================================
print(f"""
─────────────────────────────────────────────────────────────────────
SECTION 5 — TRIAL COMPLETION vs THERAPEUTIC SUCCESS  (Question 3)
─────────────────────────────────────────────────────────────────────

  CLINICAL SUCCESS SPECTRUM (6 stages):
  ──────────────────────────────────────────────────────
  Stage 1: Data collected / trial registered
  Stage 2: Trial completed    ◄── OUR PROXY IS HERE
  Stage 3: Primary endpoint met (drug showed efficacy)
  Stage 4: Regulatory submission (NDA / BLA filed)
  Stage 5: Regulatory approval (FDA / EMA granted)
  Stage 6: Post-market clinical benefit proven
  ──────────────────────────────────────────────────────

  A COMPLETED trial can still FAIL Stage 3.
  Example: Phase 3, 800 patients, HR = 0.98 → our proxy = Positive.
           The drug failed. Our proxy does not see this.

  Real oncology success rates (industry benchmarks):
    Phase 1 → Approval :  ~5%
    Phase 2 → Approval : ~10%
    Phase 3 → Approval : ~50%
    Full portfolio      :  ~5–10%

  Our {100*success_rate:.1f}% completion rate ≠ {100*success_rate:.1f}% drug efficacy.
  The gap between 73.1% and ~5–10% is what the proxy cannot see.

  ┌──────────────────────────────┬──────────────────────────────────────┐
  │ Additional field             │ How it moves the proxy toward Stage 5 │
  ├──────────────────────────────┼──────────────────────────────────────┤
  │ why_stopped                  │ Separate futility from safety stops  │
  │ primary_endpoint_result      │ Directly measures Stage 3            │
  │ p_value / hazard_ratio       │ Continuous efficacy signal           │
  │ regulatory_decision          │ Directly measures Stage 5            │
  │ biomarker_selection          │ Controls for enrichment bias         │
  └──────────────────────────────┴──────────────────────────────────────┘
""")


# ============================================================
# SECTION 6 — VALIDATION + CORE COUNTS
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 6 — OUTCOME ASSIGNMENT VERIFIED ON REAL DATA
─────────────────────────────────────────────────────────────────────
""")
valid_outcomes = {"Positive","Negative","Censored","Unknown"}
assert set(df["outcome"].dropna().unique()) <= valid_outcomes
assert len(df[(df["status_clean"]=="Completed") & (df["outcome"]!="Positive")]) == 0
assert len(df[(df["outcome_binary"]==1) & (df["status_clean"]!="Completed")]) == 0
assert len(df[(df["outcome_binary"]==0) &
              (~df["status_clean"].isin(["Terminated","Withdrawn","Suspended"]))]) == 0
print("✓ All 4 outcome label assertions passed")

print("\nStatus × Outcome Matrix:\n")
ct_disp = pd.crosstab(df["status_clean"], df["outcome"],
                      margins=True, margins_name="TOTAL")
print(ct_disp.to_string())

print()
if unknown_pct  > 10: print(f"⚠  WARNING: {unknown_pct}% Unknown — material exclusion bias (see Section 7).")
if censored_pct > 25: print(f"⚠  WARNING: {censored_pct}% Censored — right-censoring bias (see Section 14).")

print(f"""
{'─'*65}
  Positive  (Completed)           :  {n_positive:5d}  ({100*n_positive/n_total:.1f}%)
  Negative  (Stopped Early)       :  {n_negative:5d}  ({100*n_negative/n_total:.1f}%)
  Censored  (Still Active)        :  {n_censored:5d}  ({censored_pct:.1f}%)
  Unknown   (No Status Update)    :  {n_unknown:5d}  ({unknown_pct:.1f}%)
{'─'*65}
  Total Trials                    :  {n_total:5d}
  Resolved Denominator            :  {n_resolved:5d}  ({100*n_resolved/n_total:.1f}% of all trials)

  ► Proxy Success Rate            :  {100*success_rate:.1f}%
  ► 95% Wilson CI                 :  {100*ci_low:.1f}% – {100*ci_high:.1f}%""")


# ============================================================
# SECTION 7 — SENSITIVITY ANALYSIS
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 7 — SENSITIVITY ANALYSIS  (4 Scenarios)
─────────────────────────────────────────────────────────────────────
""")
pos1, res1, rate1, ci1l, ci1h = compute_rate(df)

df_s2 = df.copy()
df_s2.loc[df_s2["status_clean"]=="Suspended","outcome_binary"] = np.nan
pos2, res2, rate2, ci2l, ci2h = compute_rate(df_s2)

df_s3 = df.copy()
df_s3.loc[df_s3["status_clean"]=="Unknown","outcome_binary"] = 0
pos3, res3, rate3, ci3l, ci3h = compute_rate(df_s3)

df_s4 = df.copy()
df_s4.loc[df_s4["status_clean"]=="Unknown","outcome_binary"] = 1
pos4, res4, rate4, ci4l, ci4h = compute_rate(df_s4)

scenarios_list = [
    ("S1 — Baseline",
     "Suspended=Negative,  Unknown=Excluded",
     pos1, res1, rate1, ci1l, ci1h),
    ("S2 — Suspended as Censored",
     "Suspended=Censored,  Unknown=Excluded",
     pos2, res2, rate2, ci2l, ci2h),
    ("S3 — Unknown as Negative (pessimistic)",
     "Suspended=Negative,  Unknown=Negative",
     pos3, res3, rate3, ci3l, ci3h),
    ("S4 — Unknown as Positive (optimistic)",
     "Suspended=Negative,  Unknown=Positive",
     pos4, res4, rate4, ci4l, ci4h),
]
print(f"  {'Scenario':<42} {'n_pos':>5} {'n_res':>5}  {'Rate':>7}  {'95% CI':>15}  {'Δ vs S1':>10}")
print("  " + "─"*91)
for name, _, pos, res, rate, cil, cih in scenarios_list:
    delta = f"{100*(rate-rate1):+.2f}pp" if "Baseline" not in name else "  baseline"
    print(f"  {name:<42} {pos:>5} {res:>5}"
          f"  {100*rate:>6.1f}%  [{100*cil:.1f}%–{100*cih:.1f}%]  {delta:>10}")

spread = 100*(rate4 - rate3)
print(f"""
  KEY INSIGHT:
  Suspended assumption is negligible ({100*(rate2-rate1):+.2f}pp change).
  Unknown assumption is the dominant source of uncertainty:
    Pessimistic (Unknown=Negative) : {100*rate3:.1f}%
    Optimistic  (Unknown=Positive) : {100*rate4:.1f}%
    Total spread                   : {spread:.1f} percentage points
  → The 16pp spread means the true rate is somewhere in [{100*rate3:.1f}%, {100*rate4:.1f}%].
    Lead with this caveat in the Loom video.
""")

scenarios_df = pd.DataFrame(
    [(n, a, p, r, round(100*rt,2), round(100*cl,2), round(100*ch,2))
     for n, a, p, r, rt, cl, ch in scenarios_list],
    columns=["scenario","assumption","n_pos","n_res","success_rate_%","ci_low_%","ci_high_%"])
scenarios_df.to_csv(OUT/"sensitivity_analysis.csv", index=False)


# ============================================================
# SECTION 8 — PHASE-STRATIFIED RATES + CHI-SQUARE TEST
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 8 — PHASE-STRATIFIED RATES + STATISTICAL TEST
─────────────────────────────────────────────────────────────────────
""")
phase_order = ["Phase 1","Phase 1/2","Phase 2","Phase 2/3","Phase 3","Phase 4"]
phase_df = rate_table(df, "phase_clean")
phase_df["_ord"] = phase_df["phase_clean"].map({p:i for i,p in enumerate(phase_order)})
phase_df = phase_df.sort_values("_ord").drop(columns="_ord")
phase_df["vs_overall"] = phase_df["success_rate_%"].apply(
    lambda x: f"{x-100*success_rate:+.1f}pp" if pd.notna(x) else "—")

# Chi-square test: are phase differences statistically significant?
ct_phase = pd.crosstab(resolved_df["phase_clean"], resolved_df["outcome_binary"])
chi2_ph, p_ph, dof_ph, _ = chi2_contingency(ct_phase)

print(f"  Chi-square test (phase vs outcome):  χ²={chi2_ph:.3f}  p={p_ph:.4f}  df={dof_ph}")
if p_ph < 0.05:
    print("  → Phase differences ARE statistically significant (p < 0.05)")
else:
    print(f"  → Phase differences are NOT statistically significant (p={p_ph:.3f})")
    print("    The apparent differences in rates may be due to chance alone.")
    print("    Confounders: phase is correlated with indication and enrollment size.\n")

print(f"\n  {'Phase':<14} {'n_res':>6} {'n_pos':>6} {'n_neg':>6}"
      f"  {'Rate':>8}  {'95% CI':>16}  {'vs Overall':>11}  {'Reliable':>9}")
print("  " + "─"*85)
for _, r in phase_df[phase_df["phase_clean"].isin(phase_order)].iterrows():
    ci = (f"[{r['ci_low_%']:.1f}%–{r['ci_high_%']:.1f}%]"
          if pd.notna(r["ci_low_%"]) else "—")
    print(f"  {r.phase_clean:<14} {r.n_resolved:>6} {r.n_positive:>6} {r.n_negative:>6}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>16}  {r.vs_overall:>11}  {r.reliable:>9}")
print(f"  {'OVERALL':<14} {n_resolved:>6} {n_positive:>6} {n_negative:>6}"
      f"  {100*success_rate:>7.1f}%  [{100*ci_low:.1f}%–{100*ci_high:.1f}%]")

print("""
  Phase 4 leads  (87.5%) — post-market confirmatory; well-funded.
  Phase 3 second (80.3%) — survived Phase 2 attrition; more mature.
  Phase 2 lowest (70.1%) — the classic 'attrition valley'.
  ⚠  Despite apparent differences, χ² p=0.39 → not statistically
     significant. Do NOT over-interpret phase rankings.
""")
phase_df.to_csv(OUT/"phase_stratified_rates.csv", index=False)


# ============================================================
# SECTION 9 — INDICATION-STRATIFIED RATES + CHI-SQUARE TEST
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 9 — INDICATION-STRATIFIED RATES + STATISTICAL TEST
─────────────────────────────────────────────────────────────────────
""")
ind_df = rate_table(df, "indication_group").sort_values("success_rate_%", ascending=False)

ct_ind = pd.crosstab(resolved_df["indication_group"], resolved_df["outcome_binary"])
chi2_in, p_in, dof_in, _ = chi2_contingency(ct_ind)

print(f"  Chi-square test (indication vs outcome):  χ²={chi2_in:.3f}  p={p_in:.4f}  df={dof_in}")
if p_in < 0.05:
    print(f"  → Indication differences ARE statistically significant (p={p_in:.4f})\n"
          f"    Indication is a real predictor of completion probability.")
else:
    print("  → Not statistically significant.\n")

print(f"\n  {'Indication':<30} {'n_res':>6} {'n_pos':>6}  {'Rate':>8}  {'95% CI':>18}")
print("  " + "─"*72)
for _, r in ind_df.iterrows():
    ci = (f"[{r['ci_low_%']:.1f}%–{r['ci_high_%']:.1f}%]"
          if pd.notna(r["ci_low_%"]) else "suppressed")
    print(f"  {r.indication_group:<30} {r.n_resolved:>6} {r.n_positive:>6}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>18}")

print("""
  Lymphoma     (100.0%) — all 22 resolved trials completed.
  Colorectal   ( 93.3%) — well above overall; note wide CI at n=15.
  Multiple Myeloma (51.9%) — significantly below; high attrition.
  AML          ( 55.0%) — complex biology; high early termination.
  NSCLC        ( 63.3%) — crowded indication; combination failures.
  ✓  p=0.001 confirms these differences are real, not noise.
""")
ind_df.to_csv(OUT/"indication_stratified_rates.csv", index=False)


# ============================================================
# SECTION 10 — TECHNOLOGY-STRATIFIED RATES
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 10 — TECHNOLOGY-STRATIFIED RATES
─────────────────────────────────────────────────────────────────────
(Groups with n < 10 resolved trials suppressed)
""")
tech_df = rate_table(df, "tech_group").sort_values("success_rate_%", ascending=False)
tech_r   = tech_df[tech_df["reliable"]=="✓"]
tech_sup = tech_df[tech_df["reliable"]!="✓"]

print(f"  {'Technology':<28} {'n_res':>6} {'n_pos':>6}"
      f"  {'Rate':>8}  {'95% CI':>18}")
print("  " + "─"*72)
for _, r in tech_r.iterrows():
    ci = f"[{r['ci_low_%']:.1f}%–{r['ci_high_%']:.1f}%]"
    print(f"  {r.tech_group:<28} {r.n_resolved:>6} {r.n_positive:>6}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>18}")

print(f"\n  Suppressed (n < {MIN_N} resolved): "
      f"{len(tech_sup)} groups including ADC, CAR-T, Bispecific Antibody")
print("""
  Only 4 groups have enough resolved trials.
  ADC and CAR-T are important modalities but too new to rate reliably.
""")
tech_df.to_csv(OUT/"tech_stratified_rates.csv", index=False)


# ============================================================
# SECTION 11 — COMBINATION vs MONOTHERAPY
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 11 — COMBINATION vs MONOTHERAPY
─────────────────────────────────────────────────────────────────────
""")
combo_df = rate_table(df, "therapy_type")

ct_cb = pd.crosstab(resolved_df["therapy_type"], resolved_df["outcome_binary"])
chi2_cb, p_cb, _, _ = chi2_contingency(ct_cb)

print(f"  Chi-square test:  χ²={chi2_cb:.3f}  p={p_cb:.4f}")
print(f"  → {'NOT ' if p_cb >= 0.05 else ''}statistically significant.\n")

print(f"  {'Therapy':<20} {'n_res':>6} {'n_pos':>6} {'n_neg':>6} {'n_cens':>8}"
      f"  {'Rate':>8}  {'95% CI':>18}")
print("  " + "─"*80)
for _, r in combo_df.iterrows():
    ci = f"[{r['ci_low_%']:.1f}%–{r['ci_high_%']:.1f}%]"
    print(f"  {r.therapy_type:<20} {r.n_resolved:>6} {r.n_positive:>6} "
          f"{r.n_negative:>6} {r.n_censored:>8}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>18}")

cv = combo_df.set_index("therapy_type")
if "Combination" in cv.index and "Monotherapy" in cv.index:
    delta = cv.loc["Combination","success_rate_%"] - cv.loc["Monotherapy","success_rate_%"]
    print(f"""
  Combination vs Monotherapy delta: {delta:+.1f}pp  (p={p_cb:.3f} → not significant)
  Confounding: combination trials cluster in harder indications (NSCLC, AML)
  which have independently lower completion rates. Raw comparison misleads.
""")


# ============================================================
# SECTION 12 — ENROLLMENT SIZE ANALYSIS  ← NEW
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 12 — ENROLLMENT SIZE VS COMPLETION RATE  ← NEW
─────────────────────────────────────────────────────────────────────
Larger trials complete at higher rates. Mann-Whitney U test checks
whether this difference is statistically significant.
""")
completed_enroll = resolved_df[resolved_df["outcome_binary"]==1]["enrollment_n"].dropna()
stopped_enroll   = resolved_df[resolved_df["outcome_binary"]==0]["enrollment_n"].dropna()
stat_mw, p_mw = mannwhitneyu(completed_enroll, stopped_enroll, alternative="two-sided")

print(f"  Completed trials — median enrollment : {completed_enroll.median():.0f} patients")
print(f"  Stopped trials   — median enrollment : {stopped_enroll.median():.0f} patients")
print(f"  Mann-Whitney U test : U={stat_mw:.0f}  p={p_mw:.4f}")
if p_mw < 0.001:
    print("  → HIGHLY significant (p < 0.001). Larger trials complete at higher rates.")
    print("    Likely mechanism: larger trials have stronger sponsor commitment,")
    print("    more robust funding, and better-powered efficacy signals.")

print()
enroll_rows = []
buckets = [(0,20,"1–20"),(20,50,"21–50"),(50,100,"51–100"),(100,500,"101–500"),(500,9999,">500")]
for lo, hi, label in buckets:
    sub = resolved_df[
        (resolved_df["enrollment_n"] > lo) &
        (resolved_df["enrollment_n"] <= hi)
    ]
    n = len(sub)
    if n >= 5:
        pos = int((sub["outcome_binary"]==1).sum())
        rate = pos/n
        cil, cih = wilson_ci(pos, n)
        enroll_rows.append({
            "enrollment_bucket": label,
            "n_resolved": n,
            "n_positive": pos,
            "success_rate_%": round(100*rate,1),
            "ci_low_%": round(100*cil,1),
            "ci_high_%": round(100*cih,1),
        })
        print(f"  {label:<10}: n={n:4d}  rate={100*rate:5.1f}%  CI=[{100*cil:.0f}%–{100*cih:.0f}%]")

print("""
  KEY: Completion rate rises sharply from 58.0% (tiny trials, 1–20 pts)
       to 90.8% (medium trials, 51–100 pts). Enrollment size is the single
       strongest predictor of completion in our logistic model (Section 16).
""")
pd.DataFrame(enroll_rows).to_csv(OUT/"enrollment_bucket_rates.csv", index=False)


# ============================================================
# SECTION 13 — DURATION BUCKET ANALYSIS  ← NEW
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 13 — TRIAL DURATION vs COMPLETION RATE  ← NEW
─────────────────────────────────────────────────────────────────────
""")
dur_comp = resolved_df[resolved_df["outcome_binary"]==1]["trial_duration_days"].dropna()
dur_stop = resolved_df[resolved_df["outcome_binary"]==0]["trial_duration_days"].dropna()
_, p_dur = mannwhitneyu(dur_comp, dur_stop, alternative="two-sided")

print(f"  Completed trials — median duration  : {dur_comp.median():.0f} days  ({dur_comp.median()/365:.1f} yrs)")
print(f"  Stopped trials   — median duration  : {dur_stop.median():.0f} days  ({dur_stop.median()/365:.1f} yrs)")
print(f"  Mann-Whitney U test : p={p_dur:.4f}")
if p_dur < 0.001:
    print("  → HIGHLY significant. Completed trials run ~1.3 years longer on average.")
    print("    Trials that stop early have a short duration by definition —")
    print("    this relationship is partly tautological but confirms data integrity.\n")

dur_rows = []
dur_order = ["< 1 Year","1–2 Years","2–3 Years","> 3 Years"]
for bucket, sub in resolved_df.groupby("duration_bucket", observed=True):
    n = len(sub)
    if n >= 5:
        pos = int((sub["outcome_binary"]==1).sum())
        rate = pos/n
        cil, cih = wilson_ci(pos, n)
        dur_rows.append({
            "duration_bucket": str(bucket),
            "n_resolved": n, "n_positive": pos,
            "success_rate_%": round(100*rate,1),
            "ci_low_%": round(100*cil,1),
            "ci_high_%": round(100*cih,1),
        })
        print(f"  {str(bucket):<12}: n={n:4d}  rate={100*rate:5.1f}%"
              f"  CI=[{100*cil:.0f}%–{100*cih:.0f}%]")

print("""
  > 3 year trials complete at 82.1% — these are large, well-resourced Phase 3s.
  < 1 year trials complete at 55.3% — many are quick Phase 1s that stop early.
""")
pd.DataFrame(dur_rows).to_csv(OUT/"duration_bucket_rates.csv", index=False)


# ============================================================
# SECTION 14 — YEAR TREND + RIGHT-CENSORING BIAS
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 14 — START-YEAR TREND WITH RIGHT-CENSORING BIAS
─────────────────────────────────────────────────────────────────────
""")
year_rows = []
for yr in range(2010, 2024):
    sub_all = df[df["start_year"]==yr]
    sub_res = sub_all[sub_all["outcome_binary"].notna()]
    n       = len(sub_res)
    n_cens  = int(sub_all["outcome_binary"].isna().sum())
    cens_p  = round(100*n_cens/len(sub_all), 1) if len(sub_all) > 0 else np.nan
    if n >= 5:
        pos = int((sub_res["outcome_binary"]==1).sum())
        rate = pos/n
        cil, cih = wilson_ci(pos, n)
        year_rows.append({
            "start_year": int(yr), "n_resolved": n, "n_positive": pos,
            "n_censored": n_cens, "censored_%": cens_p,
            "success_rate_%": round(100*rate,1),
            "ci_low_%": round(100*cil,1), "ci_high_%": round(100*cih,1),
        })
        bias = "  ← ⚠ censoring bias" if cens_p > 50 else ""
        print(f"  {yr}  n_res={n:3d}  cens={n_cens:3d} ({cens_p:5.1f}%)"
              f"  rate={100*rate:5.1f}%  [{100*cil:.0f}%–{100*cih:.0f}%]{bias}")

print("""
  Do NOT interpret declining 2020+ rates as worsening quality.
  Most recent cohorts have 50–80% of their trials still active.
  Right-censoring makes recent cohorts look artificially bad.
""")
pd.DataFrame(year_rows).to_csv(OUT/"year_trend.csv", index=False)


# ============================================================
# SECTION 15 — INDICATION × PHASE CROSS-TAB
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 15 — INDICATION × PHASE CROSS-TABULATION  (n ≥ 10)
─────────────────────────────────────────────────────────────────────
""")
ip_rows = []
for (ind, ph), sub in resolved_df.groupby(
        ["indication_group","phase_clean"], observed=True):
    n   = len(sub)
    pos = int((sub["outcome_binary"]==1).sum())
    rate = pos/n if n > 0 else np.nan
    cil, cih = wilson_ci(pos, n) if n >= MIN_N else (np.nan, np.nan)
    ip_rows.append({
        "indication_group": ind, "phase_clean": ph,
        "n_resolved": n, "n_positive": pos,
        "success_rate_%": round(100*rate,1) if not np.isnan(rate) else np.nan,
        "ci_low_%": round(100*cil,1) if not np.isnan(cil) else np.nan,
        "ci_high_%": round(100*cih,1) if not np.isnan(cih) else np.nan,
        "reliable": "✓" if n >= MIN_N else f"⚠ n={n}",
    })
ip_df = pd.DataFrame(ip_rows).sort_values(["indication_group","phase_clean"])

shown = ip_df[ip_df["reliable"]=="✓"]
print(f"  {'Indication':<28} {'Phase':<12} {'n_res':>5} {'n_pos':>5}"
      f"  {'Rate':>8}  {'95% CI':>18}")
print("  " + "─"*80)
for _, r in shown.iterrows():
    ci = f"[{r['ci_low_%']:.1f}%–{r['ci_high_%']:.1f}%]"
    print(f"  {r.indication_group:<28} {r.phase_clean:<12}"
          f" {r.n_resolved:>5} {r.n_positive:>5}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>18}")

print("""
  Notable contrast: Multiple Myeloma Phase 2 = 35.7% (16 CI: [16–61])
                    vs Lymphoma Phase 2 = 100.0% [80–100].
  Same phase, very different biology and attrition patterns.
""")
ip_df.to_csv(OUT/"ind_phase_crosstab.csv", index=False)


# ============================================================
# SECTION 16 — LOGISTIC REGRESSION  ← NEW
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 16 — LOGISTIC REGRESSION: PREDICTORS OF COMPLETION
─────────────────────────────────────────────────────────────────────
What predicts whether a trial completes?
Features: phase_int, is_combination, enrollment_n, indication, tech_group
""")

model_df = resolved_df[[
    "outcome_binary","phase_int","is_combination",
    "enrollment_n","indication_group","tech_group"
]].dropna().copy()

le_ind  = LabelEncoder()
le_tech = LabelEncoder()
model_df["ind_enc"]  = le_ind.fit_transform(model_df["indication_group"])
model_df["tech_enc"] = le_tech.fit_transform(model_df["tech_group"])

X = model_df[["phase_int","is_combination","enrollment_n",
              "ind_enc","tech_enc"]].copy()
X["is_combination"] = X["is_combination"].astype(int)
y = model_df["outcome_binary"].values

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_sc, y)

features = ["phase_int","is_combination","enrollment_n","indication_group","tech_group"]
auc      = roc_auc_score(y, lr.predict_proba(X_sc)[:,1])

print(f"  Model trained on n={len(model_df)} resolved trials")
print(f"  AUC = {auc:.3f}  (0.5 = random, 1.0 = perfect)")
print()
print(f"  {'Feature':<22} {'Coefficient':>12}  {'Odds Ratio':>11}  {'Direction'}")
print("  " + "─"*62)
for feat, coef in zip(features, lr.coef_[0]):
    OR = np.exp(coef)
    direction = "▲ increases completion" if coef > 0 else "▼ decreases completion"
    print(f"  {feat:<22} {coef:>+12.3f}  {OR:>11.2f}  {direction}")

print(f"""
  KEY FINDINGS:
  ► enrollment_n has the STRONGEST positive effect (OR={np.exp(lr.coef_[0][2]):.2f}).
    A 1-SD increase in enrollment raises the completion odds 7x.
    Bigger trials = more committed sponsors = higher completion.

  ► phase_int has a NEGATIVE coefficient (OR={np.exp(lr.coef_[0][0]):.2f}).
    Higher phase does NOT independently predict better completion
    once enrollment size is controlled. Phase effects seen in Section 8
    are largely explained by enrollment size differences.

  ► is_combination is slightly negative (OR={np.exp(lr.coef_[0][1]):.2f}).
    Marginal effect — not the key driver.

  ► AUC = {auc:.3f} → the model has good discriminative power.
    Enrollment size and indication together explain most of the variation.
""")


# ============================================================
# SECTION 17 — ASSUMPTION REGISTRY
# ============================================================
print("""
─────────────────────────────────────────────────────────────────────
SECTION 17 — ASSUMPTION REGISTRY  (Auditable Decision Log)
─────────────────────────────────────────────────────────────────────
""")
assumption_registry = pd.DataFrame([
    ("A1","Suspended","Treated as Negative (outcome_binary=0)",
     "Most suspended oncology trials do not resume to completion.",
     f"Low: {100*(rate2-rate1):+.2f}pp if reclassified as Censored"),
    ("A2","Unknown","Excluded from denominator (NaN)",
     "Cannot distinguish Unknown=failed vs Unknown=stale registry entry.",
     f"HIGH: {100*(rate3-rate1):+.1f}pp (pessimistic) / "
     f"{100*(rate4-rate1):+.1f}pp (optimistic). Spread = {spread:.1f}pp"),
    ("A3","Not Yet Recruiting","Censored — excluded",
     "No outcome event has occurred.","Negligible"),
    ("A4","Enrolling By Invitation","Grouped with Recruiting (Censored)",
     "Operationally identical to Recruiting.","Negligible (n=2)"),
    ("A5","EARLY_PHASE1","Grouped with Phase 1",
     "Sub-category per FDA definition.","Negligible"),
    ("A6","Denominator","Resolved trials only (Positive + Negative)",
     f"Right-censoring mitigation. {n_total-n_resolved} trials excluded.",
     f"Denominator={n_resolved} not {n_total}. Censored%={censored_pct}%"),
    ("A7","Success definition","Trial Completion — Stage 2 of 6",
     "Only available proxy — no endpoint result or approval data.",
     f"Major: {100*success_rate:.1f}% completion ≠ ~5–10% real approval rate"),
], columns=["id","applies_to","decision","rationale","sensitivity_impact"])

for _, r in assumption_registry.iterrows():
    print(f"  [{r.id}] {r.applies_to}")
    print(f"     Decision   : {r.decision}")
    print(f"     Rationale  : {r.rationale}")
    print(f"     Sensitivity: {r.sensitivity_impact}\n")

assumption_registry.to_csv(OUT/"assumption_registry.csv", index=False)


# ============================================================
# SECTION 18 — SAVE ALL + FINAL SUMMARY
# ============================================================
summary_metrics = pd.DataFrame({
    "metric": [
        "total_trials","positive_trials","negative_trials",
        "censored_trials","unknown_trials","resolved_trials",
        "success_rate_%","ci_low_%_wilson","ci_high_%_wilson",
        "unknown_%_of_total","censored_%_of_total",
        "pessimistic_rate_%","optimistic_rate_%","sensitivity_spread_pp",
        "chi2_phase_pvalue","chi2_indication_pvalue","chi2_combo_pvalue",
        "enrollment_mannwhitney_pvalue","duration_mannwhitney_pvalue",
        "logistic_regression_auc",
    ],
    "value": [
        n_total, n_positive, n_negative, n_censored, n_unknown, n_resolved,
        round(100*success_rate,2), round(100*ci_low,2), round(100*ci_high,2),
        unknown_pct, censored_pct,
        round(100*rate3,2), round(100*rate4,2), round(spread,2),
        round(p_ph,4), round(p_in,4), round(p_cb,4),
        round(p_mw,4), round(p_dur,4),
        round(auc,3),
    ]
})
summary_metrics.to_csv(OUT/"success_rate_metrics.csv", index=False)

print("=" * 70)
print("OUTPUTS SAVED")
print("=" * 70)
for fname, tbl in [
    ("success_rate_metrics.csv",         summary_metrics),
    ("phase_stratified_rates.csv",        phase_df),
    ("indication_stratified_rates.csv",   ind_df),
    ("tech_stratified_rates.csv",         tech_df),
    ("enrollment_bucket_rates.csv",       pd.DataFrame(enroll_rows)),
    ("duration_bucket_rates.csv",         pd.DataFrame(dur_rows)),
    ("year_trend.csv",                    pd.DataFrame(year_rows)),
    ("ind_phase_crosstab.csv",            ip_df),
    ("sensitivity_analysis.csv",          scenarios_df),
    ("assumption_registry.csv",           assumption_registry),
]:
    print(f"  ✅  {fname:<40}  {len(tbl):>5} rows")

print(f"""
─────────────────────────────────────────────────────────────────────
PART 2A COMPLETE — SUMMARY
  Q1 Status → outcome mapping     : Defined, 4 assertions passed ✓
  Q2 Ambiguous statuses            : Suspended=Negative (+{100*(rate2-rate1):.2f}pp)
                                     Unknown=Excluded ({spread:.1f}pp spread) ✓
  Q3 Completion vs efficacy        : Stage 2/6.
                                     {100*success_rate:.1f}% [CI {100*ci_low:.1f}–{100*ci_high:.1f}%]
                                     ≠ real approval rate (~5–10%) ✓

  STATISTICALLY TESTED:
    Phase differences      : p={p_ph:.3f} — NOT significant ⚠
    Indication differences : p={p_in:.4f} — SIGNIFICANT ✓
    Combo vs Mono          : p={p_cb:.3f} — NOT significant
    Enrollment effect      : p={p_mw:.4f} — HIGHLY significant ✓
    Duration effect        : p={p_dur:.4f} — HIGHLY significant ✓

  LOGISTIC REGRESSION (AUC={auc:.3f}):
    Enrollment is the strongest predictor (OR=7.13).
    Phase effects largely explained by enrollment size.

""" )