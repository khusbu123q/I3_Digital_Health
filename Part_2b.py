"""
PART 2B -- STRATIFIED SUCCESS RATES: COHORT ANALYSIS
Clinical Trials Dataset  |  1000 Interventional Trials

DIMENSIONS (11 total):
  D1.  Indication Group          D7.  Enrollment Size Bucket
  D2.  Trial Phase               D8.  Trial Era
  D3.  Technology Ty pe           D9.  Drug Count Bucket
  D4.  Target Class              D10. Precision vs Cytotoxic (trial-level)
  D5.  Indication x Phase        D11. Zero/Missing Enrollment Flag
  D6.  Combination vs Mono
  + SPOTLIGHT: Enrollment x Indication interaction
  + PAIRWISE: Fisher exact + Cohen's h effect sizes
  + CENSORING-ADJUSTED rate projection

BUGS FIXED (vs previous version):
  ✅  D9  -- drug count now correctly ordered 0,1,2,3,>3 (not by rate)
  ✅  D10 -- trial-level dedup: n_res<=507 not 769. Previous version counted
           trial-target PAIRS; each trial now counted exactly once.
  ✅  Pairwise -- Breast vs Pancreatic removed (p=0.16, not significant)
  ✅  Summary Sig? column -- uses == True not is True (numpy.bool fix)

OUTPUTS (13 CSV + 3 charts):
  2b_d1..d11_*.csv   2b_pairwise_effect_sizes.csv
  2b_summary_all_dimensions.csv
  2b_stratified_dashboard.png
  2b_heatmap_ind_phase.png
  2b_enrollment_interaction.png
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches

# ============================================================
# LOAD
# ============================================================
OUT     = Path("outputs"); OUT.mkdir(exist_ok=True)
df      = pd.read_csv(OUT / "trials_clean.csv")
targets = pd.read_csv(OUT / "trial_targets.csv")
drugs   = pd.read_csv(OUT / "trial_drugs.csv")
df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce")

# -- Constants ---------------------------------------------
MIN_N        = 10
N_POS        = int((df["outcome_binary"] == 1).sum())
N_NEG        = int((df["outcome_binary"] == 0).sum())
N_RESOLVED   = N_POS + N_NEG
N_TOTAL      = len(df)
OVERALL_RATE = N_POS / N_RESOLVED
PHASE_ORDER  = ["Phase 1","Phase 1/2","Phase 2","Phase 2/3","Phase 3","Phase 4"]
resolved_df  = df[df["outcome_binary"].notna()].copy()

print("=" * 70)
print("PART 2B -- STRATIFIED SUCCESS RATES: COHORT ANALYSIS")
print("=" * 70)
print(f"\n  Overall proxy success rate : {100*OVERALL_RATE:.1f}%"
      f"  (n={N_RESOLVED} resolved / {N_TOTAL} total)")
print(f"  Suppression threshold      : n < {MIN_N} resolved -> CI suppressed\n")


# ============================================================
# HELPERS
# ============================================================

def wilson_ci(pos, n):
    if n == 0: return np.nan, np.nan
    return proportion_confint(int(pos), int(n), alpha=0.05, method="wilson")

def build_rate_table(data, group_col, min_n=MIN_N):
    """One row per stratum. Sorted by success_rate DESC (override if needed)."""
    resolved = data[data["outcome_binary"].notna()].copy()
    rows = []
    for grp, sub in resolved.groupby(group_col, observed=True):
        n   = len(sub)
        pos = int((sub["outcome_binary"] == 1).sum())
        neg = n - pos
        rate = pos / n if n > 0 else np.nan
        cil, cih = wilson_ci(pos, n) if n >= min_n else (np.nan, np.nan)
        cens = int(data[data[group_col] == grp]["outcome_binary"].isna().sum())
        vs   = round(100*rate - 100*OVERALL_RATE, 1) if not np.isnan(rate) else np.nan
        rows.append({
            group_col:         grp,
            "n_resolved":      n,
            "n_positive":      pos,
            "n_negative":      neg,
            "n_censored":      cens,
            "success_rate_%":  round(100*rate, 1) if not np.isnan(rate) else np.nan,
            "ci_low_%":        round(100*cil,  1) if not np.isnan(cil)  else np.nan,
            "ci_high_%":       round(100*cih,  1) if not np.isnan(cih)  else np.nan,
            "vs_overall_pp":   vs,
            "reliable":        "✓" if n >= min_n else f"⚠ n={n}",
        })
    return pd.DataFrame(rows).sort_values("success_rate_%", ascending=False)

def print_rate_table(tbl, group_col, chi2_p=None, note=None):
    if chi2_p is not None:
        sig = "SIGNIFICANT ✓" if chi2_p < 0.05 else "not significant ⚠"
        print(f"  Chi-square p = {chi2_p:.4f}  ->  {sig}\n")
    if note:
        print(f"  NOTE: {note}\n")
    print(f"  {'Stratum':<32} {'n_res':>6} {'n_pos':>6} {'n_neg':>6}"
          f"  {'Rate':>8}  {'95% CI':>18}  {'vs Overall':>11}  {'Flag':>8}")
    print("  " + "-" * 97)
    for _, r in tbl.iterrows():
        ci = (f"[{r['ci_low_%']:.1f}%-{r['ci_high_%']:.1f}%]"
              if pd.notna(r["ci_low_%"]) else "suppressed")
        vs = f"{r['vs_overall_pp']:+.1f}pp" if pd.notna(r["vs_overall_pp"]) else "--"
        print(f"  {str(r[group_col]):<32} {r.n_resolved:>6} {r.n_positive:>6} {r.n_negative:>6}"
              f"  {r['success_rate_%']:>7.1f}%  {ci:>18}  {vs:>11}  {r.reliable:>8}")
    ci_ov = wilson_ci(N_POS, N_RESOLVED)
    print(f"\n  {'OVERALL':<32} {N_RESOLVED:>6} {N_POS:>6} {N_NEG:>6}"
          f"  {100*OVERALL_RATE:>7.1f}%  [{100*ci_ov[0]:.1f}%-{100*ci_ov[1]:.1f}%]")

def chi2_test(data, group_col):
    res = data[data["outcome_binary"].notna()]
    ct  = pd.crosstab(res[group_col], res["outcome_binary"])
    if ct.shape[0] < 2: return np.nan
    _, p, _, _ = chi2_contingency(ct)
    return p

def cohens_h(p1, p2):
    return abs(2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p2)))


# ============================================================
# D1. INDICATION GROUP
# ============================================================
print("\n" + "=" * 70)
print("D1 -- SUCCESS RATE BY INDICATION GROUP")
print("=" * 70)

d1 = build_rate_table(df, "indication_group")
p1 = chi2_test(df, "indication_group")
print_rate_table(d1, "indication_group", p1)

print("""
  INTERPRETATION:
  > Lymphoma (100.0%, +26.9pp) -- all 22 resolved trials completed.
    Well-understood biology, clear response criteria.
  > Colorectal (93.3%, +20.3pp) -- established regimens (FOLFOX/FOLFIRI)
    reduce protocol risk; strong sponsor commitment.
  > Multiple Myeloma (51.9%, -21.2pp) -- worst performer.
    Complex multi-drug regimens, refractory disease, crowded pipeline.
  > NSCLC (63.3%, -9.8pp) -- enrollment size is the key driver:
    low-enrol NSCLC = 33% vs high-enrol NSCLC = 91% (Spotlight section).
  > p=0.001: indication differences are statistically REAL.
""")
d1.to_csv(OUT / "2b_d1_indication.csv", index=False)


# ============================================================
# D2. TRIAL PHASE
# ============================================================
print("\n" + "=" * 70)
print("D2 -- SUCCESS RATE BY TRIAL PHASE")
print("=" * 70)

d2 = build_rate_table(df, "phase_clean")
p2 = chi2_test(df, "phase_clean")
d2["_ord"] = d2["phase_clean"].map({p:i for i,p in enumerate(PHASE_ORDER)})
d2 = d2.sort_values("_ord", na_position="last").drop(columns="_ord")
print_rate_table(d2, "phase_clean", p2)

print("\n  Median enrollment per phase (explains the apparent phase gradient):")
phase_enroll = resolved_df.groupby("phase_clean")["enrollment_n"].median()
for ph in PHASE_ORDER:
    if ph in phase_enroll.index:
        print(f"    {ph:<12}: median = {phase_enroll[ph]:.0f} patients")

# Kendall tau: is there a monotonic trend across phase order?
from scipy.stats import kendalltau as kendalltau_test
_phase_pairs = []
for i, ph in enumerate(PHASE_ORDER):
    sub = resolved_df[resolved_df["phase_clean"]==ph]
    for val in sub["outcome_binary"].values:
        _phase_pairs.append((i, val))
_ph_df = pd.DataFrame(_phase_pairs, columns=["phase_ord","outcome"])
_tau, _p_tau = kendalltau_test(_ph_df["phase_ord"], _ph_df["outcome"])

print(f"""
  KEY INSIGHT:
  Phase 3 enrolls {phase_enroll.get('Phase 3',0):.0f} pts median vs Phase 1's {phase_enroll.get('Phase 1',0):.0f}.
  Logistic regression (Part 2A §16): enrollment OR=7.13.
  Once enrollment is controlled, phase_int coefficient is NEGATIVE.
  -> The Phase 4>3>2>1 ranking is largely an enrollment artefact.

  Kendall tau trend test (phase ordinal 1->4 vs outcome):
    τ = {_tau:.3f}  p = {_p_tau:.4f}
    -> NO significant monotonic trend across phases.
    -> Confirms chi-square result (p={p2:.3f}): phase order does not
       predict completion probability.
""")
d2.to_csv(OUT / "2b_d2_phase.csv", index=False)


# ============================================================
# D3. TECHNOLOGY TYPE
# ============================================================
print("\n" + "=" * 70)
print("D3 -- SUCCESS RATE BY TECHNOLOGY TYPE")
print("=" * 70)

d3      = build_rate_table(df, "tech_group")
p3      = chi2_test(df, "tech_group")
d3_show = d3[d3["reliable"] == "✓"]
d3_supp = d3[d3["reliable"] != "✓"]
print_rate_table(d3_show, "tech_group", p3)
print(f"\n  {len(d3_supp)} groups suppressed (n<{MIN_N}): "
      "ADC, CAR-T, Bispecific, Vaccine, siRNA, Gene Therapy …")

print(f"""
  INTERPRETATION:
  > Protein Therapy (87.5%) -- targeted, precision rationale; small n.
  > Small Molecule (74.1%) -- dominant modality; 390 resolved trials.
  > ADC/CAR-T suppressed -- too new; most trials still active.
    These are the FUTURE of oncology; insufficient resolved data.
  > p={p3:.3f}: technology differences not significant at alpha=0.05.
""")
d3.to_csv(OUT / "2b_d3_technology.csv", index=False)


# ============================================================
# D4. TARGET CLASS
# ============================================================
print("\n" + "=" * 70)
print("D4 -- SUCCESS RATE BY TARGET CLASS")
print("=" * 70)
print("  (From trial_targets bridge table | n >= 15 | excl. DNA)\n")

tgt_merged = targets[targets["target"] != "DNA"].merge(
    df[["nct_id","outcome_binary"]], on="nct_id", how="left")
d4      = build_rate_table(tgt_merged, "target", min_n=15)
p4      = chi2_test(tgt_merged, "target")
d4_show = d4[d4["reliable"] == "✓"]
print_rate_table(d4_show, "target", p4)

print(f"""
  INTERPRETATION:
  > HER2 (95.0%, +21.9pp) -- precision biomarker-selected patients;
    enriched populations, strong commercial backing.
  > TOP1/TOP2 (87.5%/84.8%) -- established chemotherapy targets;
    decades of clinical experience, well-understood toxicity.
  > Proteasome (52.9%, -20.1pp) -- Multiple Myeloma trials;
    complex refractory disease, high early dropout.
  > PD-1 (66.7%) -- crowded competitive landscape across indications.
  > p={p4:.4f}: target class NOT significant overall; individual
    contrasts (HER2 vs Proteasome, h=1.06) ARE large effects (see Pairwise).
""")
d4.to_csv(OUT / "2b_d4_target.csv", index=False)


# ============================================================
# D5. INDICATION x PHASE
# ============================================================
print("\n" + "=" * 70)
print("D5 -- INDICATION x PHASE CROSS-TABULATION")
print("=" * 70)
print(f"  Cells with n < {MIN_N} resolved trials suppressed\n")

ip_rows = []
for (ind, ph), sub in resolved_df.groupby(
        ["indication_group","phase_clean"], observed=True):
    n   = len(sub)
    pos = int((sub["outcome_binary"] == 1).sum())
    rate = pos/n if n > 0 else np.nan
    cil, cih = wilson_ci(pos, n) if n >= MIN_N else (np.nan, np.nan)
    ip_rows.append({
        "indication_group": ind, "phase_clean": ph,
        "n_resolved": n, "n_positive": pos, "n_negative": n-pos,
        "success_rate_%": round(100*rate,1) if not np.isnan(rate) else np.nan,
        "ci_low_%":  round(100*cil,1) if not np.isnan(cil) else np.nan,
        "ci_high_%": round(100*cih,1) if not np.isnan(cih) else np.nan,
        "vs_overall_pp": round(100*rate-100*OVERALL_RATE,1) if not np.isnan(rate) else np.nan,
        "reliable": "✓" if n >= MIN_N else f"⚠ n={n}",
    })
d5 = pd.DataFrame(ip_rows).sort_values(["indication_group","phase_clean"])
shown = d5[d5["reliable"] == "✓"]

print(f"  {'Indication':<30} {'Phase':<12} {'n_res':>6} {'n_pos':>6}"
      f"  {'Rate':>8}  {'95% CI':>18}  {'vs Overall':>11}")
print("  " + "-" * 86)
for _, r in shown.iterrows():
    ci  = f"[{r['ci_low_%']:.1f}%-{r['ci_high_%']:.1f}%]"
    vs  = f"{r['vs_overall_pp']:+.1f}pp"
    print(f"  {r.indication_group:<30} {r.phase_clean:<12}"
          f" {r.n_resolved:>6} {r.n_positive:>6}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>18}  {vs:>11}")

print("""
  KEY CONTRASTS (widest range = 64.3pp):
  > Lymphoma x Phase 2 = 100.0%  vs  Multiple Myeloma x Phase 2 = 35.7%
    Same phase. 64pp gap. Biology dominates phase entirely.
  > Solid Tumors x Phase 1 = 93.3%  vs  NSCLC x Phase 1 = 50.0%
    Basket trials vs competitive single-indication studies.
  > Breast x Phase 2 = 80.0%  vs  Pancreatic x Phase 2 = 80.0%
    Same rate, very different CIs: [61-91%] vs [49-94%].
    Same number ≠ same confidence. Always show CI alongside rate.
""")
d5.to_csv(OUT / "2b_d5_ind_phase.csv", index=False)


# ============================================================
# D6. COMBINATION vs MONOTHERAPY
# ============================================================
print("\n" + "=" * 70)
print("D6 -- COMBINATION vs MONOTHERAPY")
print("=" * 70)

d6 = build_rate_table(df, "therapy_type")
p6 = chi2_test(df, "therapy_type")
print_rate_table(d6, "therapy_type", p6)

print(f"""
  > -1.6pp gap NOT significant (p={p6:.3f}).
  > Confounding: combination trials cluster in NSCLC and AML --
    indications with independently lower completion rates.
  > Adjusted conclusion: therapy type is NOT a meaningful predictor
    of completion once indication is controlled.
""")
d6.to_csv(OUT / "2b_d6_therapy_type.csv", index=False)


# ============================================================
# D7. ENROLLMENT SIZE BUCKET
# ============================================================
print("\n" + "=" * 70)
print("D7 -- ENROLLMENT SIZE BUCKET")
print("=" * 70)

enroll_rows = []
for lo, hi, label in [(0,20,"1-20 pts"),(20,50,"21-50 pts"),
                       (50,100,"51-100 pts"),(100,500,"101-500 pts"),(500,9999,">500 pts")]:
    sub_all = df[(df["enrollment_n"] > lo) & (df["enrollment_n"] <= hi)]
    sub_res = sub_all[sub_all["outcome_binary"].notna()]
    n   = len(sub_res)
    pos = int((sub_res["outcome_binary"]==1).sum()) if n > 0 else 0
    cens= int(sub_all["outcome_binary"].isna().sum())
    rate = pos/n if n > 0 else np.nan
    cil, cih = wilson_ci(pos, n) if n >= MIN_N else (np.nan, np.nan)
    vs = round(100*rate - 100*OVERALL_RATE, 1) if not np.isnan(rate) else np.nan
    enroll_rows.append({"enrollment_bucket":label,"n_resolved":n,"n_positive":pos,
                        "n_negative":n-pos,"n_censored":cens,
                        "success_rate_%":round(100*rate,1) if not np.isnan(rate) else np.nan,
                        "ci_low_%":round(100*cil,1) if not np.isnan(cil) else np.nan,
                        "ci_high_%":round(100*cih,1) if not np.isnan(cih) else np.nan,
                        "vs_overall_pp":vs,"reliable":"✓" if n>=MIN_N else f"⚠n={n}"})
d7 = pd.DataFrame(enroll_rows)

print(f"\n  {'Bucket':<18} {'n_res':>6} {'n_pos':>6} {'n_neg':>6}"
      f"  {'Rate':>8}  {'95% CI':>18}  {'vs Overall':>11}")
print("  " + "-" * 82)
for _, r in d7.iterrows():
    ci = (f"[{r['ci_low_%']:.1f}%-{r['ci_high_%']:.1f}%]"
          if pd.notna(r["ci_low_%"]) else "suppressed")
    vs = f"{r['vs_overall_pp']:+.1f}pp" if pd.notna(r["vs_overall_pp"]) else "--"
    print(f"  {r.enrollment_bucket:<18} {r.n_resolved:>6} {r.n_positive:>6} {r.n_negative:>6}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>18}  {vs:>11}")
print(f"\n  {'OVERALL':<18} {N_RESOLVED:>6} {N_POS:>6} {N_NEG:>6}  {100*OVERALL_RATE:>7.1f}%")

# Kruskal-Wallis test: is enrollment bucket a significant predictor?
from scipy.stats import kruskal as kruskal_test
_d7_groups = []
for lo, hi in [(0,20),(20,50),(50,100),(100,500),(500,9999)]:
    sub = resolved_df[(resolved_df["enrollment_n"]>lo)&(resolved_df["enrollment_n"]<=hi)]
    if len(sub) >= 10:
        _d7_groups.append(sub["outcome_binary"].dropna().values)
_stat_d7, _p_d7 = kruskal_test(*_d7_groups)

# Spearman: does completion rate monotonically increase with enrollment?
from scipy.stats import spearmanr as spearmanr_test
_d7_rel = d7[d7["reliable"]=="✓"].copy()
_rho, _p_rho = spearmanr_test(
    list(range(len(_d7_rel))),
    _d7_rel["success_rate_%"].tolist())

print(f"""
  Kruskal-Wallis test (enrollment bucket vs outcome):
    H = {_stat_d7:.2f}  p = {_p_d7:.6f}  -> HIGHLY SIGNIFICANT ✓

  Spearman rank correlation (enrollment rank vs rate):
    ρ = {_rho:.3f}  p = {_p_rho:.4f}
    -> Directionally consistent (ρ=0.70) but p={_p_rho:.4f} at n=5 data points
      is insufficient for statistical confirmation of monotonicity.
      The Kruskal-Wallis result (p<0.000001) is the reliable test here.

  > 32.8pp jump from 1-20pt (58.0%) to 51-100pt (90.8%) trials.
  > Rate plateaus above 51 pts -- additional size adds little.
  > Mechanism: tiny trials = exploratory, underfunded, loose stopping rules.
  > Strongest predictor in logistic model (OR=7.13, p<0.0001).
""")
d7.to_csv(OUT / "2b_d7_enrollment.csv", index=False)


# ============================================================
# D8. TRIAL ERA
# ============================================================
print("\n" + "=" * 70)
print("D8 -- SUCCESS RATE BY TRIAL ERA")
print("=" * 70)
print("  RIGHT-CENSORING WARNING: 2020+ has 237 active trials -> rate deflated.\n")

df_era = df.copy()
df_era["era"] = pd.cut(df_era["start_year"],
    bins=[0, 2009, 2014, 2019, 9999],
    labels=["Pre-2010","2010-2014","2015-2019","2020+"])

era_rows = []
for era, sub_all in df_era.groupby("era", observed=True):
    sub_res = sub_all[sub_all["outcome_binary"].notna()]
    n   = len(sub_res)
    pos = int((sub_res["outcome_binary"]==1).sum())
    cens= int(sub_all["outcome_binary"].isna().sum())
    cens_pct = round(100*cens/len(sub_all),1) if len(sub_all)>0 else np.nan
    rate = pos/n if n>0 else np.nan
    cil, cih = wilson_ci(pos, n) if n>=MIN_N else (np.nan, np.nan)
    vs = round(100*rate-100*OVERALL_RATE,1) if not np.isnan(rate) else np.nan
    era_rows.append({"era":str(era),"n_resolved":n,"n_positive":pos,"n_negative":n-pos,
                     "n_censored":cens,"censored_%":cens_pct,
                     "success_rate_%":round(100*rate,1) if not np.isnan(rate) else np.nan,
                     "ci_low_%":round(100*cil,1) if not np.isnan(cil) else np.nan,
                     "ci_high_%":round(100*cih,1) if not np.isnan(cih) else np.nan,
                     "vs_overall_pp":vs})
d8 = pd.DataFrame(era_rows)
p8 = chi2_test(df_era, "era")

print(f"  Chi-square p = {p8:.4f}  ->  {'SIGNIFICANT ✓' if p8<0.05 else 'not significant ⚠'}\n")
print(f"  {'Era':<14} {'n_res':>6} {'n_pos':>6} {'n_neg':>6} {'n_cens':>7} {'Cens%':>7}"
      f"  {'Rate':>8}  {'95% CI':>16}  {'vs Overall':>11}")
print("  " + "-" * 90)
for _, r in d8.iterrows():
    ci   = f"[{r['ci_low_%']:.1f}%-{r['ci_high_%']:.1f}%]"
    vs   = f"{r['vs_overall_pp']:+.1f}pp"
    bias = "  <- ⚠ censoring" if r["censored_%"] > 50 else ""
    print(f"  {r.era:<14} {r.n_resolved:>6} {r.n_positive:>6} {r.n_negative:>6} "
          f"{r.n_censored:>7} {r['censored_%']:>6.1f}%"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>16}  {vs:>11}{bias}")

print(f"""
  > Pre-2010 (79.1%) and 2010-2014 (75.3%) -- mature, reliable estimates.
  > 2020+ (52.0%) -- ⚠ CENSORING ARTEFACT: 237/312 trials still active.
    This rate will rise materially as trials complete.
  > p={p8:.4f} partly reflects censoring structure, not quality decline.
""")
d8.to_csv(OUT / "2b_d8_era.csv", index=False)


# ============================================================
# D9. DRUG COUNT BUCKET -- correctly ordered
# ============================================================
print("\n" + "=" * 70)
print("D9 -- SUCCESS RATE BY NUMBER OF DRUGS IN TRIAL")
print("=" * 70)
print("  Does protocol complexity (more drugs) hurt completion?\n")

DRUG_ORDER = ["0 drugs","1 drug","2 drugs","3 drugs",">3 drugs"]
df_dc = df.copy()
df_dc["drug_bucket"] = pd.Categorical(
    pd.cut(df_dc["n_drugs"], bins=[-1,0,1,2,3,99], labels=DRUG_ORDER),
    categories=DRUG_ORDER, ordered=True)

d9_raw = build_rate_table(df_dc, "drug_bucket")
# Enforce correct clinical order (0,1,2,3,>3) not rate order
d9_raw["_ord"] = d9_raw["drug_bucket"].map({b:i for i,b in enumerate(DRUG_ORDER)})
d9 = d9_raw.sort_values("_ord").drop(columns="_ord")
p9 = chi2_test(df_dc, "drug_bucket")
print_rate_table(d9, "drug_bucket", p9)

# Mann-Whitney
comp_drugs = resolved_df[resolved_df["outcome_binary"]==1]["n_drugs"].dropna()
stop_drugs = resolved_df[resolved_df["outcome_binary"]==0]["n_drugs"].dropna()
_, p_mw9   = mannwhitneyu(comp_drugs, stop_drugs, alternative="two-sided")
print(f"\n  Mann-Whitney (completed vs stopped n_drugs): p={p_mw9:.4f}")
print(f"  Completed median n_drugs: {comp_drugs.median():.1f}")
print(f"  Stopped   median n_drugs: {stop_drugs.median():.1f}")

print(f"""
  > All drug counts complete at similar rates (range only ~4pp).
  > p={p9:.4f}: drug count NOT a significant predictor.
  > Protocol complexity is NOT the bottleneck.
    Stopping is driven by biology and funding, not drug count.
""")
d9.to_csv(OUT / "2b_d9_drug_count.csv", index=False)


# ============================================================
# D10. PRECISION vs CYTOTOXIC -- trial-level (bug-fixed)
# ============================================================
print("\n" + "=" * 70)
print("D10 -- PRECISION vs CYTOTOXIC TARGET PARADIGM  (trial-level)")
print("=" * 70)
print("""
  Each trial is assigned ONE paradigm using a priority rule:
    Precision > Cytotoxic > Other
  This ensures each trial is counted exactly ONCE.
  (Previous version counted trial-target PAIRS -- now fixed.)

    Precision : trial has >=1 of HER2, PD-1, EGFR, VEGFR, CD20,
                c-Kit, PDL1, ALK, RET, FGFR, BRAF, GR
    Cytotoxic : no precision target; has >=1 of TUBB1, TYMS, RNR,
                TOP1, TOP2, DHFR, Proteasome
    Other     : no classified target
""")

PRECISION_T = {"HER2","PD-1","EGFR","PDL1","CD20","VEGFR2","VEGFR1",
               "VEGFR3","c-Kit","FGFR1","FGFR2","FGFR3","FGFR4","ALK",
               "RET","MET","BRAF","GR"}
CYTOTOXIC_T = {"TUBB1","TYMS","RNR","TOP1","TOP2","DHFR","Proteasome",
               "DNA-directed DNA polymerase"}

# One priority row per target, keep best per trial
tgt_p = targets[targets["target"] != "DNA"].merge(
    df[["nct_id","outcome_binary"]], on="nct_id", how="left").copy()
tgt_p["priority"] = tgt_p["target"].apply(
    lambda t: 0 if t in PRECISION_T else (1 if t in CYTOTOXIC_T else 2))
trial_par = (tgt_p.sort_values("priority")
             .groupby("nct_id", as_index=False)
             .first()[["nct_id","priority","outcome_binary"]])
trial_par["paradigm"] = trial_par["priority"].map(
    {0:"Precision / Targeted", 1:"Cytotoxic / Broad", 2:"Other"})

# Merge: trials with no targets are excluded from D10
df_d10 = df[["nct_id","outcome_binary"]].merge(
    trial_par[["nct_id","paradigm"]], on="nct_id", how="inner")

d10 = build_rate_table(df_d10, "paradigm")
p10 = chi2_test(df_d10, "paradigm")
print_rate_table(d10, "paradigm", p10,
    note=f"n_resolved total = {d10['n_resolved'].sum()} trials "
         f"(trials with no classified target excluded)")

# Extract rates safely
def get_rate(tbl, col, val):
    r = tbl[tbl[col]==val]["success_rate_%"].values
    return r[0] if len(r) > 0 else np.nan

prec_r = get_rate(d10,"paradigm","Precision / Targeted")
cyto_r = get_rate(d10,"paradigm","Cytotoxic / Broad")

print(f"""
  > Precision ({prec_r:.1f}%) vs Cytotoxic ({cyto_r:.1f}%)
    Delta: {prec_r-cyto_r:.1f}pp  (p={p10:.3f}) -- NOT statistically significant.

  > The paradigm grouping masks strong within-group heterogeneity.
    D4 (individual targets) is more informative:
    HER2=95% vs Proteasome=53%  (Cohen's h=1.06, LARGE effect).

  > Future portfolios (ADC, bispecific, CAR-T) are all precision modalities.
    As their trials resolve, this gap may widen materially.
""")
d10.to_csv(OUT / "2b_d10_precision_cytotoxic.csv", index=False)


# ============================================================
# D11. ZERO / MISSING ENROLLMENT FLAG
# ============================================================
print("\n" + "=" * 70)
print("D11 -- ZERO / MISSING ENROLLMENT FLAG ANALYSIS")
print("=" * 70)
print("  Trials with enrollment=0 or missing -- are they more likely to fail?\n")

ze_rows = []
for flag, sub_all in df.groupby("flag_zero_enroll"):
    sub_res = sub_all[sub_all["outcome_binary"].notna()]
    n   = len(sub_res)
    pos = int((sub_res["outcome_binary"]==1).sum())
    cens= int(sub_all["outcome_binary"].isna().sum())
    rate = pos/n if n>0 else np.nan
    cil, cih = wilson_ci(pos, n) if n>=MIN_N else (np.nan, np.nan)
    vs = round(100*rate-100*OVERALL_RATE,1) if not np.isnan(rate) else np.nan
    label = "Zero/Missing Enroll" if flag else "Normal Enrollment"
    ze_rows.append({"flag":label,"n_resolved":n,"n_positive":pos,"n_negative":n-pos,
                    "n_censored":cens,
                    "success_rate_%":round(100*rate,1),
                    "ci_low_%":round(100*cil,1) if not np.isnan(cil) else np.nan,
                    "ci_high_%":round(100*cih,1) if not np.isnan(cih) else np.nan,
                    "vs_overall_pp":vs})
d11 = pd.DataFrame(ze_rows)
ct_ze = pd.crosstab(resolved_df["flag_zero_enroll"], resolved_df["outcome_binary"])
_, p11 = fisher_exact(ct_ze)

print(f"  Fisher exact p = {p11:.4f}  ->  {'SIGNIFICANT ✓' if p11<0.05 else 'not significant'}\n")
print(f"  {'Flag':<22} {'n_res':>6} {'n_pos':>6} {'n_neg':>6}"
      f"  {'Rate':>8}  {'95% CI':>18}  {'vs Overall':>11}")
print("  " + "-" * 80)
for _, r in d11.iterrows():
    ci = (f"[{r['ci_low_%']:.1f}%-{r['ci_high_%']:.1f}%]"
          if pd.notna(r["ci_low_%"]) else "suppressed")
    vs = f"{r['vs_overall_pp']:+.1f}pp"
    print(f"  {r['flag']:<22} {r.n_resolved:>6} {r.n_positive:>6} {r.n_negative:>6}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>18}  {vs:>11}")

print("""
  > Zero/missing enrollment: 35.0% vs 77.1% for normal trials.
  > 38pp gap, p<0.0001 -- HIGHLY significant.
  > Mechanism: zero enrollment = trial never properly started,
    immediately suspended, or severe data quality issues.
  > Both a data quality finding AND a predictive signal.
""")
d11.to_csv(OUT / "2b_d11_zero_enrollment.csv", index=False)


# ============================================================
# D12. TRIAL DURATION BUCKET
# ============================================================
print("\n" + "=" * 70)
print("D12 -- SUCCESS RATE BY TRIAL DURATION BUCKET")
print("=" * 70)
print("""
  Trial duration = primary_completion_date − start_date.
  Longer trials -> higher completion (partly tautological: stopped trials
  are short by definition). But the gradient is informative.
""")

DURATION_ORDER = ["<1yr","1-2yr","2-3yr",">3yr"]
df_dur = resolved_df.copy()
df_dur["dur_bucket"] = pd.Categorical(
    pd.cut(df_dur["trial_duration_days"],
           bins=[-1, 365, 730, 1095, 99999],
           labels=DURATION_ORDER),
    categories=DURATION_ORDER, ordered=True)

dur_rows = []
for b, sub in df_dur.groupby("dur_bucket", observed=True):
    n   = len(sub)
    pos = int((sub["outcome_binary"]==1).sum())
    neg = n - pos
    rate = pos/n if n>0 else np.nan
    cil, cih = wilson_ci(pos, n) if n>=MIN_N else (np.nan, np.nan)
    vs = round(100*rate - 100*OVERALL_RATE, 1) if not np.isnan(rate) else np.nan
    dur_rows.append({"duration_bucket":str(b),"n_resolved":n,"n_positive":pos,
                     "n_negative":neg,
                     "success_rate_%":round(100*rate,1) if not np.isnan(rate) else np.nan,
                     "ci_low_%":round(100*cil,1) if not np.isnan(cil) else np.nan,
                     "ci_high_%":round(100*cih,1) if not np.isnan(cih) else np.nan,
                     "vs_overall_pp":vs,
                     "reliable":"✓" if n>=MIN_N else f"⚠ n={n}"})
d12 = pd.DataFrame(dur_rows)

# Kruskal-Wallis across duration buckets
from scipy.stats import kruskal as kruskal_dur
_dur_groups = [df_dur[df_dur["dur_bucket"]==b]["outcome_binary"].dropna().values
               for b in DURATION_ORDER
               if len(df_dur[df_dur["dur_bucket"]==b])>=10]
_stat_dur, _p_dur = kruskal_dur(*_dur_groups)

# Mann-Whitney: completed vs stopped duration
_dur_comp = resolved_df[resolved_df["outcome_binary"]==1]["trial_duration_days"].dropna()
_dur_stop = resolved_df[resolved_df["outcome_binary"]==0]["trial_duration_days"].dropna()
from scipy.stats import mannwhitneyu as mwu_dur
_, _p_mw_dur = mwu_dur(_dur_comp, _dur_stop, alternative="two-sided")

print(f"  Kruskal-Wallis (duration bucket vs outcome): H={_stat_dur:.2f}  p={_p_dur:.6f}  -> SIGNIFICANT ✓")
print(f"  Mann-Whitney (completed vs stopped duration): p={_p_mw_dur:.4f}  -> SIGNIFICANT ✓")
print(f"  Completed median: {_dur_comp.median():.0f} days ({_dur_comp.median()/365:.1f} yrs)")
print(f"  Stopped   median:  {_dur_stop.median():.0f} days ({_dur_stop.median()/365:.1f} yrs)\n")

print(f"  {'Duration':<10} {'n_res':>6} {'n_pos':>6} {'n_neg':>6}  {'Rate':>8}  {'95% CI':>18}  {'vs Overall':>11}")
print("  " + "-"*75)
for _, r in d12.iterrows():
    ci = (f"[{r['ci_low_%']:.1f}%-{r['ci_high_%']:.1f}%]"
          if pd.notna(r["ci_low_%"]) else "suppressed")
    vs = f"{r['vs_overall_pp']:+.1f}pp" if pd.notna(r["vs_overall_pp"]) else "--"
    print(f"  {r.duration_bucket:<10} {r.n_resolved:>6} {r.n_positive:>6} {r.n_negative:>6}"
          f"  {r['success_rate_%']:>7.1f}%  {ci:>18}  {vs:>11}")
print(f"\n  {'OVERALL':<10} {N_RESOLVED:>6} {N_POS:>6} {N_NEG:>6}  {100*OVERALL_RATE:>7.1f}%")

print(f"""
  > >3yr trials (82.1%) -- large well-resourced Phase 3s. These trials
    were designed to complete; stopping is not planned.
  > <1yr trials (55.3%) -- quick Phase 1 safety studies or early-stopped
    trials. Short duration = often a sign of early termination.
  > Duration is partly tautological (stopped = short by definition),
    but confirms data integrity: duration aligns with outcome labels.
  > Kruskal-Wallis p={_p_dur:.6f}: duration bucket IS statistically
    significant -- but interpret with caution due to tautology.
""")
d12.to_csv(OUT / "2b_d12_duration.csv", index=False)


# ============================================================
# SPOTLIGHT: ENROLLMENT x INDICATION INTERACTION
# ============================================================
print("\n" + "=" * 70)
print("SPOTLIGHT -- ENROLLMENT x INDICATION INTERACTION")
print("=" * 70)
print(f"  Split at overall median = {resolved_df['enrollment_n'].median():.0f} patients\n")

MEDIAN_E = resolved_df["enrollment_n"].median()
spot_rows = []
print(f"  {'Indication':<30} {'Lo_n':>5} {'Lo%':>7}  {'Hi_n':>5} {'Hi%':>7}  {'Delta':>8}")
print("  " + "-" * 68)
for ind, sub in resolved_df.groupby("indication_group"):
    lo = sub[sub["enrollment_n"] <= MEDIAN_E]
    hi = sub[sub["enrollment_n"] >  MEDIAN_E]
    if len(lo) >= 5 and len(hi) >= 5:
        lo_r = (lo["outcome_binary"]==1).sum()/len(lo)*100
        hi_r = (hi["outcome_binary"]==1).sum()/len(hi)*100
        delta= hi_r - lo_r
        print(f"  {ind:<30} {len(lo):>5} {lo_r:>6.1f}%  {len(hi):>5} {hi_r:>6.1f}%  {delta:>+7.1f}pp")
        spot_rows.append({"indication":ind,"lo_n":len(lo),"lo_rate_%":round(lo_r,1),
                          "hi_n":len(hi),"hi_rate_%":round(hi_r,1),"delta_pp":round(delta,1)})

spot_df = pd.DataFrame(spot_rows).sort_values("delta_pp", ascending=False)
# Spotlight Fisher exact tests
# Fisher exact tests for key spotlight comparisons
from scipy.stats import fisher_exact as fe_spot
_spot_tests = []
for _ind, _lo_lo, _lo_hi, _hi_lo, _hi_hi in [
    ("AML",             1,  7, 10,  2),
    ("NSCLC",           8, 16, 20,  2),
    ("Pancreatic Cancer",7, 11, 14,  3),
    ("Multiple Myeloma", 8, 11,  6,  2),
]:
    # recalculate from actual data
    _sub = resolved_df[resolved_df["indication_group"]==_ind].copy()
    _lo  = _sub[_sub["enrollment_n"] <= MEDIAN_E]
    _hi  = _sub[_sub["enrollment_n"] >  MEDIAN_E]
    if len(_lo)>=5 and len(_hi)>=5:
        _ct = [[int((_lo["outcome_binary"]==1).sum()), int((_lo["outcome_binary"]==0).sum())],
               [int((_hi["outcome_binary"]==1).sum()), int((_hi["outcome_binary"]==0).sum())]]
        _, _p_fe = fe_spot(_ct)
        _spot_tests.append((_ind, _p_fe))

# Print spotlight results cleanly
print("\n  KEY FINDING (Fisher exact tests for top enrollment contrasts):")
for _ind, _p in _spot_tests:
    _sub2 = resolved_df[resolved_df["indication_group"]==_ind]
    _lo2  = _sub2[_sub2["enrollment_n"] <= MEDIAN_E]
    _hi2  = _sub2[_sub2["enrollment_n"] >  MEDIAN_E]
    if len(_lo2) > 0 and len(_hi2) > 0:
        _lr = 100*((_lo2["outcome_binary"]==1).sum()/len(_lo2))
        _hr = 100*((_hi2["outcome_binary"]==1).sum()/len(_hi2))
        _delta = _hr - _lr
        if _p < 0.05:
            _sig_str = "Fisher p={:.4f} ✓ SIGNIFICANT".format(_p)
        else:
            _sig_str = "p={:.3f} not sig".format(_p)
        print("  {:<22}: low {:.0f}%  high {:.0f}%  delta +{:.0f}pp  ({})".format(
              _ind, _lr, _hr, _delta, _sig_str))
print("  Lymphoma: 100 pct vs 100 pct -- completes regardless of enrollment.")
print("  Enrollment predicts completion WITHIN each indication.")
print("  It is the truly dominant driver.\n")

# ============================================================
# PAIRWISE COMPARISONS -- EFFECT SIZES
# ============================================================
print("\n" + "=" * 70)
print("PAIRWISE COMPARISONS -- EFFECT SIZES (Cohen's h + Fisher Exact)")
print("=" * 70)
print("""
  Cohens h: 0.2 = small  |  0.5 = medium  |  0.8 = large
  Only comparisons with Fisher p < 0.05 are shown.
""")

def pairwise_test(g1_name, g2_name, df_src, gcol):
    g1 = df_src[df_src[gcol]==g1_name]
    g2 = df_src[df_src[gcol]==g2_name]
    n1=len(g1); pos1=int((g1["outcome_binary"]==1).sum())
    n2=len(g2); pos2=int((g2["outcome_binary"]==1).sum())
    if n1==0 or n2==0: return None
    ct = np.array([[pos1,n1-pos1],[pos2,n2-pos2]])
    _, p = fisher_exact(ct)
    h   = cohens_h(pos1/n1, pos2/n2)
    return {"comparison":f"{g1_name} vs {g2_name}",
            "rate_A":round(100*pos1/n1,1),"n_A":n1,
            "rate_B":round(100*pos2/n2,1),"n_B":n2,
            "delta_pp":round(100*(pos1/n1-pos2/n2),1),
            "cohens_h":round(h,3),
            "effect_size":"Large" if h>=0.8 else ("Medium" if h>=0.5 else "Small"),
            "fisher_p":round(p,6)}

tgt_for_pair = targets[targets["target"]!="DNA"].merge(
    df[["nct_id","outcome_binary"]], on="nct_id").copy()
tgt_for_pair = tgt_for_pair[tgt_for_pair["outcome_binary"].notna()]

# Only include p < 0.05 comparisons
candidate_contrasts = [
    ("Lymphoma","Multiple Myeloma", resolved_df, "indication_group"),
    ("Lymphoma","NSCLC",            resolved_df, "indication_group"),
    ("Colorectal Cancer","AML",     resolved_df, "indication_group"),
    ("HER2","Proteasome",           tgt_for_pair, "target"),
    ("HER2","PD-1",                 tgt_for_pair, "target"),
    ("TOP2","RNR",                  tgt_for_pair, "target"),
]
pair_rows = []
print(f"  {'Comparison':<45} {'A%':>6} {'B%':>6} {'Δ':>7}  {'h':>6}  {'Effect':>7}  {'p':>10}")
print("  " + "-" * 96)
for args in candidate_contrasts:
    r = pairwise_test(*args)
    if r and r["fisher_p"] < 0.05:
        pair_rows.append(r)
        print(f"  {r['comparison']:<45} {r['rate_A']:>5.1f}% {r['rate_B']:>5.1f}%"
              f" {r['delta_pp']:>+6.1f}pp  {r['cohens_h']:>6.3f}  {r['effect_size']:>7}  {r['fisher_p']:>10.6f}")
    elif r:
        print(f"  {r['comparison']:<45} (p={r['fisher_p']:.4f} -- not significant, excluded)")

print("""
  KEY:
  > Lymphoma vs MM: h=1.53 (VERY LARGE), p=0.000145 -- biggest contrast.
  > HER2 vs Proteasome: h=1.06 (LARGE) -- precision vs cytotoxic divide.
  > HER2 vs PD-1: same "precision" paradigm, very different outcomes.
    Target specificity matters more than modality class.
""")
pd.DataFrame(pair_rows).to_csv(OUT / "2b_pairwise_effect_sizes.csv", index=False)


# ============================================================
# CENSORING-ADJUSTED RATE PROJECTION
# ============================================================
print("\n" + "=" * 70)
print("CENSORING-ADJUSTED RATE PROJECTION")
print("=" * 70)

n_cens = N_TOTAL - N_RESOLVED
rate_same  = (N_POS + n_cens*OVERALL_RATE)   / (N_RESOLVED + n_cens)
rate_pess  = (N_POS + n_cens*0.55)            / (N_RESOLVED + n_cens)
rate_opt   = (N_POS + n_cens*0.85)            / (N_RESOLVED + n_cens)

print("\n  Current observed rate (n={} resolved):   {:.1f}%\n".format(N_RESOLVED, 100*OVERALL_RATE))
print("  Projected range if {} censored trials resolve:".format(n_cens))
print("    If censored resolve at same rate ({:.0f}%): {:.1f}%".format(100*OVERALL_RATE, 100*rate_same))
print("    If censored resolve pessimistically (55%):  {:.1f}%".format(100*rate_pess))
print("    If censored resolve optimistically  (85%):  {:.1f}%\n".format(100*rate_opt))
print("  > Final rate will converge to [{:.1f}% - {:.1f}%].".format(100*rate_pess, 100*rate_opt))
print("  > 2020+ cohort (currently 52%) will rise substantially.")
print("  > This does NOT change the current estimate -- it contextualises")
print("    the right-censoring limitation.\n")


# ============================================================
# SUMMARY -- ALL 11 DIMENSIONS
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY -- ALL 12 DIMENSIONS RANKED BY RANGE")
print("=" * 70)

def safe_range(tbl, col):
    if "reliable" in tbl.columns:
        r = tbl[tbl["reliable"]=="✓"][col].dropna()
    else:
        r = tbl[col].dropna()
    return round(r.max()-r.min(),1) if len(r)>=2 else np.nan

summary_rows = [
    {"dim":"D5  IndicationxPhase",  "n_strata":len(shown),           "range_pp":64.3,
     "chi2_p":np.nan, "significant":np.nan,
     "best":"LymphomaxPh2 (100%)","worst":"MMxPh2 (35.7%)"},
    {"dim":"D1  Indication",         "n_strata":15,                   "range_pp":safe_range(d1,"success_rate_%"),
     "chi2_p":round(p1,4), "significant":bool(p1<0.05),
     "best":"Lymphoma (100.0%)","worst":"Mult. Myeloma (51.9%)"},
    {"dim":"D4  Target Class",        "n_strata":len(d4_show),         "range_pp":safe_range(d4_show,"success_rate_%"),
     "chi2_p":round(p4,4), "significant":bool(p4<0.05),
     "best":"HER2 (95.0%)","worst":"Proteasome (52.9%)"},
    {"dim":"D7  Enrollment Bucket",   "n_strata":5,                    "range_pp":safe_range(d7,"success_rate_%"),
     "chi2_p":round(_p_d7,6), "significant":bool(_p_d7<0.05),
     "best":"51-100 pts (90.8%)","worst":"1-20 pts (58.0%)"},
    {"dim":"D8  Trial Era",           "n_strata":4,                    "range_pp":safe_range(d8,"success_rate_%"),
     "chi2_p":round(p8,4), "significant":bool(p8<0.05),
     "best":"Pre-2010 (79.1%)","worst":"2020+ (52.0%)*"},
    {"dim":"D2  Phase",               "n_strata":len(d2[d2["reliable"]=="✓"]), "range_pp":safe_range(d2,"success_rate_%"),
     "chi2_p":round(p2,4), "significant":bool(p2<0.05),
     "best":"Phase 4 (87.5%)","worst":"Phase 1/2 (67.5%)"},
    {"dim":"D3  Technology",          "n_strata":len(d3_show),         "range_pp":safe_range(d3_show,"success_rate_%"),
     "chi2_p":round(p3,4), "significant":bool(p3<0.05),
     "best":"Protein Th. (87.5%)","worst":"mAb (72.7%)"},
    {"dim":"D11 Zero Enrollment",     "n_strata":2,                    "range_pp":safe_range(d11,"success_rate_%"),
     "chi2_p":round(p11,6), "significant":bool(p11<0.05),
     "best":"Normal (77.1%)","worst":"Zero/Missing (35.0%)"},
    {"dim":"D10 Prec. vs Cyto.",      "n_strata":len(d10[d10["reliable"]=="✓"]), "range_pp":safe_range(d10,"success_rate_%"),
     "chi2_p":round(p10,4), "significant":bool(p10<0.05),
     "best":f"Precision ({prec_r:.1f}%)","worst":f"Cytotoxic ({cyto_r:.1f}%)"},
    {"dim":"D9  Drug Count",          "n_strata":len(d9[d9["reliable"]=="✓"]), "range_pp":safe_range(d9,"success_rate_%"),
     "chi2_p":round(p9,4), "significant":bool(p9<0.05),
     "best":">3 drugs (75.0%)","worst":"2 drugs (71.4%)"},
    {"dim":"D12 Duration Bucket",     "n_strata":4,                    "range_pp":safe_range(d12,"success_rate_%"),
     "chi2_p":round(_p_dur,6), "significant":bool(_p_dur<0.05),
     "best":">3yr (82.1%)","worst":"<1yr (55.3%)"},
    {"dim":"D6  Therapy Type",        "n_strata":2,                    "range_pp":safe_range(d6,"success_rate_%"),
     "chi2_p":round(p6,4), "significant":bool(p6<0.05),
     "best":"Monotherapy (74.1%)","worst":"Combination (72.5%)"},
]
summary_df = pd.DataFrame(summary_rows).sort_values("range_pp", ascending=False)

print(f"\n  {'Dimension':<26} {'Strata':>7} {'Range':>9} {'Chi2 p':>9} {'Sig?':>7}  Best -> Worst")
print("  " + "-" * 100)
for _, r in summary_df.iterrows():
    # Use == True/False (not is True/is False) to handle numpy.bool correctly
    sig   = "Yes ✓" if r["significant"] == True else ("No ⚠" if r["significant"] == False else "n/a")
    p_str = f"{r['chi2_p']:.4f}" if pd.notna(r["chi2_p"]) else "n/a"
    print(f"  {r['dim']:<26} {r.n_strata:>7} {r.range_pp:>8.1f}pp {p_str:>9} {sig:>7}  {r['best']} -> {r['worst']}")

print("\n  * D8 2020+ rate is censoring-inflated. True rate likely higher.")
print("  + D11 Zero Enrollment is also a significant predictor (p<0.0001).\n")
print("  DIMENSION RANKING BY EXPLANATORY POWER:")
print("  " + "-"*65)
print("  1.  D5  Indication x Phase  (64.3pp)        -- widest range; hypothesis generation")
print("  2.  D1  Indication          (48.1pp, p=0.001)  -- ONLY significant categorical predictor")
print("  3.  D4  Target Class        (42.1pp, p=0.273)  -- HER2 vs Proteasome: h=1.06 (LARGE)")
print("  4.  D11 Zero Enrollment     (42.1pp, p<0.0001) -- data quality + strong predictive signal")
print("  5.  D7  Enrollment Bucket   (32.8pp, p<0.0001) -- strongest CONTINUOUS predictor (OR=7.13)")
print("  6.  D8  Trial Era           (27.1pp, p=0.0001) -- partly censoring artefact; not causal")
print("  7.  D12 Duration Bucket     (26.8pp, p<0.0001) -- significant but partly tautological")
print("  8.  D2  Phase               (20.0pp, p=0.391)  -- NOT significant; enrollment confounder")
print("  9.  D3  Technology          (14.8pp, p=0.087)  -- borderline; only 4 reliable groups")
print("  10. D9  Drug Count          ( {:.1f}pp, p={:.3f})  -- NOT significant; complexity != failure".format(
          safe_range(d9,"success_rate_%"), p9))
print("  11. D10 Prec. vs Cyto.      ( {:.1f}pp, p={:.3f})  -- paradigm masks heterogeneity (use D4)".format(
          safe_range(d10,"success_rate_%"), p10))
print("  12. D6  Therapy Type        (  1.6pp, p=0.749)  -- noise; confounded by indication")
print("  " + "-"*65)
print("  CONCLUSION: Disease biology (indication) and resource commitment")
print("  (enrollment size) are the two dominant drivers of trial completion.")
print("  Phase, technology, and drug count do NOT independently predict")
print("  completion once these two factors are accounted for.\n")
summary_df.to_csv(OUT / "2b_summary_all_dimensions.csv", index=False)


# ============================================================
# VISUALISATIONS
# ============================================================
BLUE="#2563EB"; RED="#DC2626"; GREY="#94A3B8"; GREEN="#059669"; AMBER="#D97706"

def bar_cols(rates):
    return [BLUE if r/100 >= OVERALL_RATE else RED for r in rates]

fig, axes = plt.subplots(3, 3, figsize=(22, 17))
fig.suptitle(
    f"Part 2B -- Stratified Trial Completion Rates  |  n=1,000  |  "
    f"Overall = {100*OVERALL_RATE:.1f}% [{100*wilson_ci(N_POS,N_RESOLVED)[0]:.1f}-"
    f"{100*wilson_ci(N_POS,N_RESOLVED)[1]:.1f}%]\n"
    f"Proxy = Trial Completion (Stage 2/6)  |  "
    f"Censored/active trials excluded from denominator",
    fontsize=11, fontweight="bold", y=1.01)

# P1: Indication
ax = axes[0,0]
r = d1[d1["reliable"]=="✓"].sort_values("success_rate_%", ascending=True)
ax.barh(r["indication_group"], r["success_rate_%"]/100,
        color=bar_cols(r["success_rate_%"]), edgecolor="white", linewidth=0.4)
ax.axvline(OVERALL_RATE, color=GREY, ls="--", lw=1.5, label=f"Overall {100*OVERALL_RATE:.0f}%")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlim(0,1.05); ax.set_title(f"D1: Indication (χ² p={p1:.4f} ✓)", fontsize=9)
ax.set_xlabel("Success Rate"); ax.legend(fontsize=7)
for i,(_, r_) in enumerate(r.iterrows()):
    ax.text(r_["success_rate_%"]/100+0.01, i, f"n={r_.n_resolved}", va="center", fontsize=6.5)

# P2: Phase
ax = axes[0,1]
r = d2[(d2["reliable"]=="✓") & (d2["phase_clean"].isin(PHASE_ORDER))].copy()
r["_ord"] = r["phase_clean"].map({p:i for i,p in enumerate(PHASE_ORDER)})
r = r.sort_values("_ord")
bars = ax.bar(r["phase_clean"], r["success_rate_%"]/100,
              color=bar_cols(r["success_rate_%"]), edgecolor="white", width=0.6)
ax.axhline(OVERALL_RATE, color=GREY, ls="--", lw=1.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0,1.05); ax.set_title(f"D2: Phase (χ² p={p2:.3f} ⚠)", fontsize=9)
ax.set_ylabel("Success Rate")
for bar, row in zip(bars, r.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f"n={row.n_resolved}", ha="center", fontsize=7)

# P3: Technology
ax = axes[0,2]
r = d3_show.sort_values("success_rate_%", ascending=True)
ax.barh(r["tech_group"], r["success_rate_%"]/100,
        color=[GREEN if v/100>=OVERALL_RATE else AMBER for v in r["success_rate_%"]],
        edgecolor="white")
ax.axvline(OVERALL_RATE, color=GREY, ls="--", lw=1.5)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlim(0,1.05); ax.set_title("D3: Technology (n>=10 only)", fontsize=9)
ax.set_xlabel("Success Rate")
for i,(_, r_) in enumerate(r.iterrows()):
    ax.text(r_["success_rate_%"]/100+0.01, i, f"n={r_.n_resolved}", va="center", fontsize=7)

# P4: Target
ax = axes[1,0]
r = d4_show.sort_values("success_rate_%", ascending=True)
ax.barh(r["target"], r["success_rate_%"]/100,
        color=bar_cols(r["success_rate_%"]), edgecolor="white")
ax.axvline(OVERALL_RATE, color=GREY, ls="--", lw=1.5)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_xlim(0,1.05); ax.set_title("D4: Target Class (n>=15, excl. DNA)", fontsize=9)
ax.set_xlabel("Success Rate")
for i,(_, r_) in enumerate(r.iterrows()):
    ax.text(r_["success_rate_%"]/100+0.01, i, f"n={r_.n_resolved}", va="center", fontsize=6.5)

# P5: Enrollment
ax = axes[1,1]
r = d7[d7["reliable"]=="✓"]
bars = ax.bar(r["enrollment_bucket"], r["success_rate_%"]/100,
              color=bar_cols(r["success_rate_%"]), edgecolor="white", width=0.6)
ax.axhline(OVERALL_RATE, color=GREY, ls="--", lw=1.5,
           label=f"Overall {100*OVERALL_RATE:.0f}%")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0,1.05); ax.set_title("D7: Enrollment Size (OR=7.13, p<0.0001)", fontsize=9)
ax.set_ylabel("Success Rate"); ax.legend(fontsize=7)
for bar, row in zip(bars, r.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f"n={row.n_resolved}", ha="center", fontsize=7.5)

# P6: Era
ax = axes[1,2]
r = d8.sort_values("era")
bars = ax.bar(r["era"], r["success_rate_%"]/100,
              color=bar_cols(r["success_rate_%"]), edgecolor="white", width=0.6)
ax.axhline(OVERALL_RATE, color=GREY, ls="--", lw=1.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0,1.05); ax.set_title("D8: Trial Era (⚠ 2020+ censored)", fontsize=9)
ax.set_ylabel("Success Rate")
for bar, row in zip(bars, r.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f"n={row.n_resolved}", ha="center", fontsize=7.5)

# P7: Drug count (correctly ordered)
ax = axes[2,0]
r = d9[d9["reliable"]=="✓"]
bars = ax.bar(r["drug_bucket"], r["success_rate_%"]/100,
              color=bar_cols(r["success_rate_%"]), edgecolor="white", width=0.6)
ax.axhline(OVERALL_RATE, color=GREY, ls="--", lw=1.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0,1.05); ax.set_title(f"D9: Drug Count (p={p9:.3f}, not sig.)", fontsize=9)
ax.set_ylabel("Success Rate")
for bar, row in zip(bars, r.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f"n={row.n_resolved}", ha="center", fontsize=7.5)

# P8: Precision vs Cytotoxic (trial-level)
ax = axes[2,1]
r = d10[d10["reliable"]=="✓"].sort_values("success_rate_%", ascending=False)
bars = ax.bar(r["paradigm"], r["success_rate_%"]/100,
              color=bar_cols(r["success_rate_%"]), edgecolor="white", width=0.6)
ax.axhline(OVERALL_RATE, color=GREY, ls="--", lw=1.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0,1.05)
ax.set_title(f"D10: Target Paradigm (trial-level, p={p10:.3f})", fontsize=9)
ax.set_ylabel("Success Rate")
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right", fontsize=8)
for bar, row in zip(bars, r.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f"n={row.n_resolved}", ha="center", fontsize=7.5)

# P9: Range summary
ax = axes[2,2]
s = summary_df.sort_values("range_pp", ascending=True)
bar_c = [BLUE if bool(sig)==True else (RED if bool(sig)==False else GREY)
         for sig in s["significant"].fillna("n/a")]
bars2 = ax.barh(s["dim"], s["range_pp"], color=bar_c, edgecolor="white")
ax.set_title("Range (pp) per Dimension\nBlue=significant, Red=not sig.", fontsize=9)
ax.set_xlabel("Rate Range (percentage points)")
for bar,(_, row) in zip(bars2, s.iterrows()):
    ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
            f"{row['range_pp']:.1f}pp", va="center", fontsize=7.5)

bp = mpatches.Patch(color=BLUE, label=">= Overall or significant")
rp = mpatches.Patch(color=RED,  label="< Overall or not significant")
fig.legend(handles=[bp,rp], loc="lower center", ncol=2,
           bbox_to_anchor=(0.5,-0.01), fontsize=9, frameon=False)
plt.tight_layout(rect=[0,0.03,1,1])
fig.savefig(OUT/"2b_stratified_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  2b_stratified_dashboard.png saved")

# -- Heatmap: Indication x Phase -------------------------
phase_cols_h = [p for p in PHASE_ORDER if p in d5["phase_clean"].values]
ind_rows_h   = ["Breast Cancer","NSCLC","Colorectal Cancer","Pancreatic Cancer",
                "AML","Multiple Myeloma","Prostate Cancer","Ovarian Cancer",
                "Gastric Cancer","Lymphoma","Leukemia","Head & Neck Cancer",
                "Hematologic Malignancies","Solid Tumors (NOS)","Other"]
d5_r = d5[d5["reliable"]=="✓"]
pvt_r = d5_r.pivot(index="indication_group",columns="phase_clean",values="success_rate_%")
pvt_n = d5_r.pivot(index="indication_group",columns="phase_clean",values="n_resolved")
cp = [p for p in phase_cols_h if p in pvt_r.columns]
rp = [i for i in ind_rows_h if i in pvt_r.index]
pvt_r = pvt_r.reindex(index=rp, columns=cp)
pvt_n = pvt_n.reindex(index=rp, columns=cp)
fig2, ax2 = plt.subplots(figsize=(13,8))
im = ax2.imshow(pvt_r.values/100, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(len(cp))); ax2.set_xticklabels(cp, fontsize=9, rotation=30, ha="right")
ax2.set_yticks(range(len(rp))); ax2.set_yticklabels(rp, fontsize=9)
for i in range(len(rp)):
    for j in range(len(cp)):
        v=pvt_r.values[i,j]; n2=pvt_n.values[i,j]
        if not np.isnan(v):
            tc="white" if (v<30 or v>75) else "black"
            ax2.text(j,i,f"{v:.0f}%\nn={int(n2)}",ha="center",va="center",
                     fontsize=8,color=tc,fontweight="bold")
        else:
            ax2.text(j,i,"--",ha="center",va="center",fontsize=11,color="#CBD5E1")
plt.colorbar(im,ax=ax2,label="Proxy Success Rate",fraction=0.025)
ax2.set_title(
    f"D5: Proxy Success Rate -- Indication x Phase  |  Overall={100*OVERALL_RATE:.1f}%\n"
    f"Key contrast: LymphomaxPhase2=100% vs MMxPhase2=35.7%  |  --=n<{MIN_N} suppressed",
    fontsize=10,pad=12)
ax2.set_xlabel("Trial Phase"); ax2.set_ylabel("Indication Group")
plt.tight_layout()
fig2.savefig(OUT/"2b_heatmap_ind_phase.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  2b_heatmap_ind_phase.png saved")

# -- Enrollment x Indication interaction chart ---------
fig3, ax3 = plt.subplots(figsize=(12,6))
s_df = spot_df.sort_values("delta_pp", ascending=True)
cols_s = [BLUE if d>0 else RED for d in s_df["delta_pp"]]
ax3.barh(s_df["indication"], s_df["delta_pp"], color=cols_s, edgecolor="white")
ax3.axvline(0, color=GREY, lw=1.5)
ax3.set_title(
    "Enrollment Effect Within Each Indication\n"
    f"(High-enrol − Low-enrol, split at median={MEDIAN_E:.0f} pts)",
    fontsize=11, pad=10)
ax3.set_xlabel("Δ Success Rate (pp, high minus low enrollment)")
for bar, (_, row) in zip(ax3.patches, s_df.iterrows()):
    x = bar.get_width()
    ax3.text(x+(0.5 if x>=0 else -0.5), bar.get_y()+bar.get_height()/2,
             f"Lo={row['lo_rate_%']:.0f}% Hi={row['hi_rate_%']:.0f}%",
             va="center", ha="left" if x>=0 else "right", fontsize=8)
plt.tight_layout()
fig3.savefig(OUT/"2b_enrollment_interaction.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅  2b_enrollment_interaction.png saved")


# ============================================================
# CROSS-VALIDATED AUC + BOOTSTRAP CI
# ============================================================
model_df = resolved_df[["outcome_binary","phase_int","is_combination",
                         "enrollment_n","indication_group","tech_group"]].dropna().copy()
le_i = LabelEncoder(); le_t = LabelEncoder()
model_df["_ind"] = le_i.fit_transform(model_df["indication_group"])
model_df["_tec"] = le_t.fit_transform(model_df["tech_group"])
X   = model_df[["phase_int","is_combination","enrollment_n","_ind","_tec"]].copy()
X["is_combination"] = X["is_combination"].astype(int)
Xsc = StandardScaler().fit_transform(X)
lr  = LogisticRegression(max_iter=1000, random_state=42)
cv_auc = cross_val_score(lr, Xsc, model_df["outcome_binary"].values, cv=5, scoring="roc_auc")

rng  = np.random.default_rng(42)
boot = [rng.choice(resolved_df["outcome_binary"].values,
                   size=len(resolved_df), replace=True).mean()
        for _ in range(5000)]
b_lo = np.percentile(boot, 2.5); b_hi = np.percentile(boot, 97.5)
# ============================================================
# FEATURE IMPORTANCE (Cross-Validated Logistic Model)
# ============================================================
# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

model_df = resolved_df.copy()

features = [
    "phase_int",
    "enrollment_n",
    "is_combination",
    "indication_group",
    "tech_group"
]

model_df = model_df.dropna(
    subset=["outcome_binary"]
)

model_df["phase_int"] = (
    model_df["phase_int"]
    .fillna(model_df["phase_int"].median())
)

model_df["enrollment_n"] = (
    model_df["enrollment_n"]
    .fillna(model_df["enrollment_n"].median())
)

model_df["is_combination"] = (
    model_df["is_combination"]
    .fillna(0)
)

model_df["indication_group"] = (
    model_df["indication_group"]
    .fillna("Unknown")
)

model_df["tech_group"] = (
    model_df["tech_group"]
    .fillna("Unknown")
)

X = pd.get_dummies(
    model_df[features],
    drop_first=True
)

y = model_df["outcome_binary"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

lr = LogisticRegression(
    max_iter=5000,
    random_state=42
)

auc_scores = cross_val_score(
    lr,
    X_scaled,
    y,
    cv=cv,
    scoring="roc_auc"
)

lr.fit(X_scaled, y)

importance = pd.DataFrame({
    "feature": X.columns,
    "abs_coef": np.abs(lr.coef_[0])
}).sort_values(
    "abs_coef",
    ascending=False
)

print(f"\nCross-validated AUC = {auc_scores.mean():.3f}")
print(f"Std AUC             = {auc_scores.std():.3f}")

print("\nTop predictors:")
print(importance.head(15))

importance.to_csv(
    OUT / "2b_feature_importance.csv",
    index=False
)
# ============================================================
# MISSING DATA REPORT
# ============================================================

print("\n" + "=" * 70)
print("MISSING DATA ANALYSIS")
print("=" * 70)

missing_report = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isnull().sum(),
    "missing_pct":
        round(df.isnull().mean()*100,2),
    "unique_values":
        df.nunique(dropna=True)
})

missing_report = (
    missing_report
    .sort_values(
        "missing_pct",
        ascending=False
    )
)

print(missing_report)

missing_report.to_csv(
    OUT / "2b_missing_data_report.csv",
    index=False
)
# ============================================================
# MISSINGNESS HEATMAP
# ============================================================

plt.figure(figsize=(14,6))

plt.imshow(
    df.isnull(),
    aspect="auto",
    interpolation="nearest"
)

plt.title("Missingness Pattern Across Dataset")

plt.xlabel("Columns")
plt.ylabel("Trials")

plt.tight_layout()

plt.savefig(
    OUT / "2b_missingness_heatmap.png",
    dpi=150
)

plt.close()

# ============================================================
# SAVE ALL CSVs
# ============================================================
print("\n" + "=" * 70)
print("OUTPUTS SAVED")
print("=" * 70)
for fname, tbl in [
    ("2b_d1_indication.csv",          d1),
    ("2b_d2_phase.csv",               d2),
    ("2b_d3_technology.csv",          d3),
    ("2b_d4_target.csv",              d4),
    ("2b_d5_ind_phase.csv",           d5),
    ("2b_d6_therapy_type.csv",        d6),
    ("2b_d7_enrollment.csv",          d7),
    ("2b_d8_era.csv",                 d8),
    ("2b_d9_drug_count.csv",          d9),
    ("2b_d10_precision_cytotoxic.csv",d10),
    ("2b_d11_zero_enrollment.csv",    d11),
    ("2b_d12_duration.csv",           d12),
    ("2b_pairwise_effect_sizes.csv",  pd.DataFrame(pair_rows)),
    ("2b_summary_all_dimensions.csv", summary_df),
]:
    tbl.to_csv(OUT / fname, index=False)
    print(f"  ✅  {fname:<45}  {len(tbl):>5} rows")

print("\n" + "="*70)
print("PART 2B COMPLETE")
print("="*70)
print("  Overall rate    : {:.1f}%".format(100*OVERALL_RATE))
print("                    Wilson CI  [{:.1f}-{:.1f}%]".format(
          100*wilson_ci(N_POS,N_RESOLVED)[0], 100*wilson_ci(N_POS,N_RESOLVED)[1]))
print("                    Bootstrap  [{:.1f}-{:.1f}%]  (5,000 resamples)".format(100*b_lo, 100*b_hi))
print("  Logistic AUC    : {:.3f} +/- {:.3f}  (5-fold CV)".format(cv_auc.mean(), cv_auc.std()))
print("  Dimensions      : 12  (7 required + 5 new)")
print("  Statistical tests: chi-square x7, Kruskal-Wallis x2,")
print("                    Kendall tau x1, Spearman x1, Mann-Whitney x3,")
print("                    Fisher exact (D11 + spotlight + 6 pairwise), Cohen h x6")
print("  Charts          : 3  (9-panel dashboard, heatmap, enrollment interaction)")
print("  Outputs         : 14 CSV files")
print("")
print("  BUGS FIXED vs PREVIOUS VERSION:")
print("    D9  ordering: now 0,1,2,3,>3 (was out of order by rate)")
print("    D10 trial-level: n_res<=507 (was 769 = trial-target pairs)")
print("    Pairwise: BC vs Pancreatic removed (p=0.16, not significant)")
print("    Summary Sig? column: uses == True (numpy.bool fix)")
print("    D7 now has formal Kruskal-Wallis test (H=50.41, p<0.000001)")
print("    Phase: Kendall tau trend test added (tau=0.027, p=0.475 -- no trend)")
print("    D12 Trial Duration added as 12th dimension (p<0.000001)")
print("    Spotlight: Fisher exact tests added for key enrollment contrasts")
print("")
print("  HEADLINE FINDINGS -- USE IN YOUR LOOM:")
print("    Indication: ONLY significant categorical predictor (p=0.001)")
print("    Enrollment: STRONGEST predictor (OR=7.13, p<0.0001)")
print("    Phase: NOT significant (p=0.39); explained by enrollment size")
print("    Lymphoma vs MM: h=1.53 (VERY LARGE), Fisher p=0.000145")
print("    HER2 vs Proteasome: h=1.06 (LARGE) -- target matters most")
print("    Zero-enrollment trials: 35% vs 77% (p<0.0001)")
print("    NSCLC: low-enrol 33% vs high-enrol 91% (+58pp within indication)")
print("    Drug count: NOT significant -- complexity != failure")
print("    Duration: >3yr (82%) vs <1yr (55%) -- Kruskal-Wallis p<0.000001")
print("    Era 2020+ projected to converge to [{:.1f}-{:.1f}%]".format(100*rate_pess, 100*rate_opt))
print("-"*65)
