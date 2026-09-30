# HF-RISK preprint v2 — Results & Discussion draft (MIMIC-IV GCS extension)

> **Status:** numbers are final for the seed-17 run of `scripts/09_mimic_native_gcs_retrain.py` (400 trees, 5-fold GroupKFold, 1,000 paired bootstraps) and `scripts/09c_zhang_mimic_shap_correlation.py`. Prose is draft. Every claim maps to a row in `results/tables/mimic_gcs_retrain_results.csv`, `results/mimic_gcs_run_log.md`, or `results/shap_cross_cohort_note.md`. **Nik must review every number and claim before submission.**
>
> Design rule applied throughout: null results are reported as nulls, and caveats sit next to the claim they qualify — not in a limitations appendix.

---

## 1. Study design (summary)

We studied 42,990 hospital admissions with heart failure (ICD-10 I50*) from 18,890 unique patients in MIMIC-IV v3.1. Because 8,006 patients have more than one HF admission in the cohort (maximum 59), every evaluation uses **patient-level grouped splits** (GroupShuffleSplit for the 80/20 holdout, GroupKFold for cross-validation) and the analysis script asserts zero patient overlap before scoring. A row-level random split — as used by the historical benchmark — can place the same patient in train and test via different admissions and therefore overstates generalisation.

Predictors are available at the expanded baseline of 34 clinical features (demographics, comorbidity flags, CCI, chemistry, haematology, cardiac markers). GCS enters as its three components (eye opening, verbal response, movement). Median imputation for non-GCS predictors is fit on training rows only. GCS is deliberately **not** imputed: 72% of admissions have no GCS because they were never neurologically assessed (ward-only stays), and imputing a score would assert a clinical observation that was never made; XGBoost's native missing-value handling is used instead.

Two encoding decisions were pre-registered in `results/mimic_gcs_run_log.md` before the numbers were produced:

1. `verbal_response == 0` is MIMIC's "untestable (intubated)" code, not a measurement, and is treated as **missing** in the main analysis. The alternative 0→1 remap is carried as a sensitivity arm.
2. `gcs_total` is excluded as a feature: for all 1,111 rows with an untestable verbal component it reads 15 regardless of profoundly abnormal components (e.g. eyes=1, motor=1) — a placeholder artefact, not a measurement.

## 2. Results

### 2.1 Cohort and outcome rates

| Outcome | Prevalence | Test positives (n = 8,749 holdout) |
|---|---:|---:|
| 28-day mortality | 10.3% | 874 |
| 6-month mortality | 22.0% | 1,830 |

GCS was charted for 12,083 admissions (28.1%); 12,110 admissions (28.2%) had at least one true ICU stay (flag taken from MIMIC-IV `icustays`, independent of any charting).

### 2.2 GCS improves discrimination at both horizons

Primary quantity: the **paired AUROC delta** (with-GCS vs baseline, same test patients) with a 1,000-iteration paired bootstrap 95% CI.

**28-day mortality** (full cohort):

| Model | Features | CV AUROC | Test AUROC | Δ vs baseline [95% CI] |
|---|---:|---:|---:|---|
| Expanded baseline | 34 | 0.9002 ± 0.003 | 0.8834 | — |
| **Baseline + GCS** | 37 | 0.9146 ± 0.003 | **0.8972** | **+0.0138 [+0.009, +0.019]** |
| Baseline + ICU-stay flag (control) | 35 | 0.9068 ± 0.004 | 0.8915 | +0.0080 [+0.004, +0.013] |
| Baseline + ICU flag + GCS | 38 | 0.9155 ± 0.003 | 0.9000 | +0.0166 [+0.012, +0.022] |

**6-month mortality** (full cohort):

| Model | Features | CV AUROC | Test AUROC | Δ vs baseline [95% CI] |
|---|---:|---:|---:|---|
| Expanded baseline | 34 | 0.8318 ± 0.007 | 0.8266 | — |
| **Baseline + GCS** | 37 | 0.8393 ± 0.006 | **0.8352** | **+0.0086 [+0.005, +0.012]** |
| Baseline + ICU-stay flag (control) | 35 | 0.8326 ± 0.006 | 0.8280 | **+0.0014 [−0.001, +0.004] — null** |
| Baseline + ICU flag + GCS | 38 | 0.8393 ± 0.007 | 0.8370 | +0.0104 [+0.007, +0.014] |

All four with-GCS improvements have CIs excluding zero at both horizons.

### 2.3 The ICU-admission control: what it shows, including the null

Because GCS is charted almost exclusively for ICU patients, "adding GCS" also adds information about who was critically ill. To separate the two, a control model adds only a true ICU-stay indicator.

- **At 6 months the control is a null result:** the ICU flag alone produces no detectable improvement (+0.0014, 95% CI [−0.0012, +0.0040]). We report this as *"no detectable improvement at this sample size"*, not as a gain.
- GCS remains significant *after* the ICU flag is in the model (+0.0104 [+0.007, +0.014]), i.e. the GCS signal is **not reducible to ICU admission**.
- At 28 days both contribute: the ICU flag alone adds +0.0080 [+0.004, +0.013], and GCS adds further on top (+0.0138 alone; +0.0166 combined with the flag). The relative contribution of "was critically ill" vs "was neurologically impaired" therefore differs by prediction horizon, and we say so rather than pooling the two.

### 2.4 ICU-only subgroup (descriptive; selected on a collider)

Among the 12,110 ICU admissions (6-month mortality prevalence 35.5%), adding GCS raised test AUROC from 0.8557 to 0.8766 (Δ +0.0208 [+0.014, +0.027]). The effect is larger where GCS was actually measured. **Caveat stated explicitly:** ICU admission depends on severity, so this subgroup is selected on a collider; the number is descriptive and must not be compared directly against the full-cohort AUROCs.

### 2.5 Calibration and sensitivity analyses

With-GCS calibration: 28-day AUPRC 0.7032 / Brier 0.0487 / ECE 0.0127; 6-month AUPRC 0.6678 / Brier 0.1122 / ECE 0.0112.

- **Verbal-0 encoding sensitivity:** replacing missing with the 0→1 remap changes AUROC by +0.0003 [−0.002, +0.002] (28-day) and +0.0002 [−0.002, +0.002] (6-month). The encoding choice is immaterial to the results; the main analysis keeps the clinically honest "untestable = missing" convention.
- **Class weighting:** `scale_pos_weight` moves 6-month AUROC from 0.8352 to 0.8357 (28-day: 0.8972 → 0.8962). Prevalence is 10–22%, not an extreme-imbalance setting; the unweighted model is the headline.

### 2.6 Model explanations (SHAP) and the cross-cohort picture

SHAP was computed for the MIMIC models (TreeSHAP exact values), closing the gap that explanations existed only for the Zhang cohort.

**MIMIC full cohort (6-month, with GCS):** top features are rdw (mean |SHAP| 0.402), CCI score (0.367), age (0.300), white blood cell count (0.193), **eye opening (0.181, rank 5)**, chloride (0.179), urea (0.150), platelet (0.142), **movement (0.138, rank 9)**. Verbal response ranks 26 — its contribution is diluted by the 72% structural missingness, not absent.

**MIMIC ICU-only (6-month):** **verbal response is the single most important feature** (0.489, rank 1), with movement rank 6 and eye opening rank 7. Where neurological assessment exists, it dominates.

**Cross-cohort comparison (Zhang ↔ MIMIC).** The defensible statement is **qualitative**: the GCS family ranks in the top tier of both cohorts *independently* — Zhang's model ranks GCS total first (eye opening 7th, movement 18th of 143), while MIMIC's ranks eye opening 5th and movement 9th of 37, and all three GCS components in its ICU-only top 7. Beyond the GCS family the two cohorts do **not** preserve feature order (e.g. congestive heart failure is Zhang's 6th-ranked feature but MIMIC's 37th), consistent with the already-observed external-validation collapse (Zhang→MIMIC AUROC 0.5235 / 0.5995).

> **Explicit caveat on the rank statistic:** with only 7 shared features inside Zhang's preserved top-20 (of 143 features), a cross-cohort rank correlation is not evidence either way — its sign can flip with a single feature entering or leaving the overlap. We therefore do **not** report or interpret any cross-cohort Spearman rho; the qualitative top-tier agreement of the GCS family is the claim. Within-MIMIC rank stability (full cohort vs ICU-only, all 37 features) is high (Spearman ρ = 0.863) and is reported as an internal-consistency check only.

## 3. Discussion

### 3.1 GCS adds prognostic information that ICU admission does not capture

The central finding is incremental and modest in size but consistent: adding GCS components improves discrimination for post-discharge mortality at both horizons, and the improvement survives adjustment for a true ICU-stay indicator. The 6-month control arm is decisive in one direction only — the ICU indicator *alone* is a null result at this sample size, while GCS is not. Neurological status at the end of the ICU stay therefore carries risk information beyond the fact of critical illness itself.

### 3.2 Which components matter, and where

Eye opening and movement drive the full-cohort signal, while verbal response dominates among ICU patients — precisely where it is actually observed. This pattern is consistent with the missingness structure: components that are evaluable on more patients contribute more in the full cohort; the most informative component contributes most where measurement exists. It also motivates the encoding choice: an intubated patient's untestable verbal response is missing data, and the sensitivity analysis shows the results do not hinge on that choice.

### 3.3 Relationship to the historical MIMIC benchmark — open question

The historical MIMIC-only benchmark (0.801 AUROC for 6-month, 0.867 for 28-day mortality) is **not comparable** to the present numbers, for two reasons that cannot currently be disentangled: (i) it used a row-level split on a cohort with 8,006 repeat patients, which leaks patients across train and test and inflates scores; (ii) it used a different 24-feature set, of which only 10 columns exist in the present extract (the comparison baseline here is a 34-feature superset and must not be labelled "the 0.801 model"). Notably, the present grouped-split numbers are *higher* than the historical ones despite de-leakage — which is unexpected and almost certainly reflects the feature-set difference, but **we do not claim to know this**. Decomposing the two effects requires re-extracting the missing legacy columns and re-running the legacy model under grouped splits; until then this remains an open question and no attribution is made.

### 3.4 Cross-cohort claims stay modest

Feature-level agreement across cohorts is limited to the GCS family, and external validation of the Zhang models on this cohort already collapsed to near-chance (0.5235 / 0.5995 AUROC). The honest synthesis is: (a) both cohorts' models independently leaned on neurological status, which supports GCS as a portable risk signal; (b) the rest of the feature hierarchy is cohort-specific, so transferability of the models themselves is poor. We report (a) and (b) together; (a) without (b) would overstate the case.

### 3.5 Limitations (each stated where it bites, restated here)

1. Single-centre MIMIC-IV cohort; the Zhang comparison is limited by the Zhang SHAP artefact preserving only its top-20 features (full comparison would require re-running SHAP on the frozen Zhang models).
2. Case definition is ICD-10 I50* only (ICD-9 428* admissions are outside the cohort).
3. GCS is structurally missing for 72% of admissions (never assessed), not missing at random; XGBoost's native handling represents this honestly but cannot recover unobserved assessments.
4. The ICU-only subgroup is selected on severity (a collider) and is descriptive. Its delta is also estimated from a single train/test split rather than the 5-fold cross-validation used for the full-cohort models (`cv_auc_std` is NaN in the results table), so +0.0208 should be treated as more provisional than the full-cohort numbers.
5. Outcomes are post-discharge windows computed from MIMIC death records; the 6-month readmission flag is binary (verified against BigQuery, 60/60 agreement, minimum 3-day gap).
6. Research prototype only; not clinically validated; not for patient-care use.

### 3.6 Claims checklist

| Claim | Status |
|---|---|
| GCS components improve AUROC for 28-day and 6-month mortality on grouped splits | **Defensible** (CIs exclude 0 at both horizons) |
| The gain is not reducible to ICU admission | **Defensible** (6-month: ICU flag null, GCS-beyond-ICU significant) |
| GCS gain is larger in the ICU subgroup | **Defensible but descriptive** (collider caveat) |
| Verbal-0 encoding choice drives the result | **No — explicitly refuted** by the sensitivity arm (Δ ≈ 0.0002–0.0003, CIs cross 0) |
| Cross-cohort rank correlation between Zhang and MIMIC feature importances | **Do not claim** (n = 7; sign unstable) |
| GCS family is top-tier in both cohorts independently | **Defensible** (qualitative, both rankings documented) |
| These models beat / match / fail to match the historical 0.801 | **Do not claim** (feature sets and split protocols differ; decomposition pending) |
| Models transfer across cohorts | **Do not claim** (external AUROC 0.5235 / 0.5995) |

## 4. Provenance

| Artifact | Path |
|---|---|
| Retrain + SHAP script (method notes 1–7) | `scripts/09_mimic_native_gcs_retrain.py` |
| True ICU-stay flag extraction | `scripts/09a_icu_stay_flags.py` |
| Cross-cohort SHAP comparison | `scripts/09c_zhang_mimic_shap_correlation.py` |
| Pre-registered run log (expectations written before results) | `results/mimic_gcs_run_log.md` |
| All AUROCs, deltas, CIs | `results/tables/mimic_gcs_retrain_results.csv` |
| MIMIC SHAP importances | `results/tables/shap_importance_full_cohort.csv`, `shap_importance_icu_only.csv` |
| Cross-cohort note + table + figure | `results/shap_cross_cohort_note.md`, `results/tables/shap_cross_cohort_correlation.csv`, `results/figures/shap_cross_cohort_ranks.png` |
| Legacy-24 availability audit | `results/tables/legacy_24_features.json` |
| Historical benchmarks (leakage-inflated; context only) | `results/tables/mimic_only_xgboost_results.csv`, `external_validation_results.csv` |
