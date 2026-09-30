# MIMIC GCS retrain - run log

Produced by `scripts/09_mimic_native_gcs_retrain.py` (seed=17, patient-level grouped split, n_estimators=400, bootstrap=1000, quick=False).

## Pre-registered expectations (text fixed in the script before any of these numbers were produced)

- The historical MIMIC benchmark (0.801 AUROC for 6m, 0.867 for 28d; `results/tables/mimic_only_xgboost_results.csv`) came from `09_mimic_only_xgboost.py` using a ROW-level `train_test_split` on a cohort where 8,006 patients have several HF admissions. The same patient could land in train and test via different admissions, so those numbers are leakage-inflated. EXPECT the grouped-split AUROCs below to be LOWER; that is a correction, not a regression.
- Zhang-to-MIMIC external validation already collapsed (0.5235 / 0.5995 AUROC; `results/tables/external_validation_results.csv`), so cross-cohort claims must stay modest regardless of what this run shows.
- The primary quantity is the PAIRED DELTA (with-GCS vs baseline on the same test patients), not the absolute AUROC.

## Data and design checks

- **42,990 admissions** from **18,890 unique patients**; **8,006** patients have more than one HF admission (max 59). All splits are grouped on `subject_id`, and the script asserts zero patient overlap before scoring.
- **12,110** admissions had a true ICU stay (flag from `icustays` via `09a_icu_stay_flags.py`); GCS was charted for **12,083**. GCS is left as native NaN for the rest - they were never assessed, and imputing a score would fabricate a clinical claim.
- `verbal_response == 0` (MIMIC's "untestable/intubated" code) is treated as **missing** in **1,111** rows (1,107 flagged intubated, 4 not). The old 0 -> 1 remap is a fabricated value and survives only as the `expanded_plus_gcs_v0as1` sensitivity arm. `gcs_total` is excluded as a feature entirely: for all 1,111 of these rows it reads 15 regardless of the components (e.g. eyes=1, motor=1), i.e. a placeholder artefact.
- Legacy 24-feature list **recovered** from `09_mimic_only_xgboost.py` and saved to `results/tables/legacy_24_features.json`.
- Only **10/24** of those columns exist in `mimic_hf_with_gcs.csv`. Missing: length_of_stay, discharge_day, platelets, wbc, magnesium, phosphorus, urea_nitrogen, glucose, alt, ast, alkaline_phosphatase, chronic_kidney_disease, hypertension, atrial_fibrillation. A faithful legacy-24 rerun is therefore not possible on this cohort, so the comparison baseline is the **expanded baseline** (34 shared features). Its AUROC is a superset model and must **not** be described as 'the 0.801 model'.
- `28d_death` verbal-0 encoding sensitivity: 0->1 remap vs missing gives delta +0.0003 (95% CI [-0.0020, +0.0024]). GCS-beyond-ICU comparison: `expanded_plus_icu_flag_plus_gcs` 0.9000 vs `baseline_plus_icu_flag` 0.8915.
- `28d_death` full cohort: expanded baseline AUROC 0.8834; +GCS 0.8972 (delta +0.0138); baseline+`had_icu_stay` control 0.8915 (delta +0.0080).
- `28d_death` with-GCS calibration: AUPRC 0.7032, Brier 0.0487, ECE(10 bins) 0.0127.
- `28d_death` scale_pos_weight sensitivity (spw=8.67): with-GCS AUROC 0.8962 vs 0.8972 unweighted. Prevalence is only 10.3%, so weighting is not necessary; the unweighted number is the headline.
- `6m_death` verbal-0 encoding sensitivity: 0->1 remap vs missing gives delta +0.0002 (95% CI [-0.0016, +0.0020]). GCS-beyond-ICU comparison: `expanded_plus_icu_flag_plus_gcs` 0.8370 vs `baseline_plus_icu_flag` 0.8280.
- `6m_death` full cohort: expanded baseline AUROC 0.8266; +GCS 0.8352 (delta +0.0086); baseline+`had_icu_stay` control 0.8280 (delta +0.0014).
- `6m_death` with-GCS calibration: AUPRC 0.6678, Brier 0.1122, ECE(10 bins) 0.0112.
- `6m_death` scale_pos_weight sensitivity (spw=3.50): with-GCS AUROC 0.8357 vs 0.8352 unweighted. Prevalence is only 22.0%, so weighting is not necessary; the unweighted number is the headline.
- **ICU-only subgroup** (`6m_death`, true ICU flag, n=12,110, prevalence 35.5%, GCS charted for 12,083): no-GCS AUROC 0.8557 vs with-GCS 0.8766, delta +0.0208 (95% CI [+0.0141, +0.0274]). Different population from the full cohort - do not compare the two AUROCs directly. ICU admission depends on severity, so this subgroup is selected on a collider; treat it as descriptive.
- SHAP full cohort (`6m_death`): GCS components rank eye_opening, movement, verbal_response of 37; strongest GCS feature is `eye_opening` (mean |SHAP| 0.1815).

## Results

```
                      model_tag           target_col   cohort  n_features  cv_auc_mean  cv_auc_std  test_auc  auc_delta_vs_baseline  delta_95ci_low  delta_95ci_high  n_test  n_positive_test
              expanded_baseline  death_within_28days     full          34       0.9002      0.0033    0.8834                    NaN             NaN              NaN    8749              874
              expanded_plus_gcs  death_within_28days     full          37       0.9146      0.0025    0.8972                 0.0138          0.0090           0.0194    8749              874
         baseline_plus_icu_flag  death_within_28days     full          35       0.9068      0.0037    0.8915                 0.0080          0.0037           0.0125    8749              874
expanded_plus_icu_flag_plus_gcs  death_within_28days     full          38       0.9155      0.0033    0.9000                 0.0166          0.0119           0.0220    8749              874
        expanded_plus_gcs_v0as1  death_within_28days     full          37       0.9150      0.0018    0.8975                 0.0141          0.0091           0.0195    8749              874
              expanded_baseline death_within_6months     full          34       0.8318      0.0069    0.8266                    NaN             NaN              NaN    8749             1830
              expanded_plus_gcs death_within_6months     full          37       0.8393      0.0058    0.8352                 0.0086          0.0054           0.0120    8749             1830
         baseline_plus_icu_flag death_within_6months     full          35       0.8326      0.0064    0.8280                 0.0014         -0.0012           0.0040    8749             1830
expanded_plus_icu_flag_plus_gcs death_within_6months     full          38       0.8393      0.0065    0.8370                 0.0104          0.0073           0.0136    8749             1830
        expanded_plus_gcs_v0as1 death_within_6months     full          37       0.8393      0.0060    0.8354                 0.0088          0.0057           0.0119    8749             1830
     expanded_baseline_icu_only death_within_6months icu_only          34          NaN         NaN    0.8557                    NaN             NaN              NaN    2514              876
     expanded_plus_gcs_icu_only death_within_6months icu_only          37          NaN         NaN    0.8766                 0.0208          0.0141           0.0274    2514              876
```

## How to read this

- Every AUROC uses a patient-level split on `subject_id`; the script asserts zero patient overlap before scoring.
- "expanded_baseline" is a superset model, **not** the historical 0.801 model (see the legacy-24 note above).
- GCS is never imputed; the ~72% missingness is left to XGBoost's native NaN handling.
- `baseline_plus_icu_flag` is the control, using the TRUE ICU-stay flag from `icustays` (`09a_icu_stay_flags.py`), independent of GCS charting. If it matches `expanded_plus_gcs`, any gain is about ICU admission, not neurological status; `expanded_plus_icu_flag_plus_gcs` answers whether GCS adds anything BEYOND ICU admission.
- `expanded_plus_gcs_v0as1` is an encoding sensitivity arm (verbal 0 -> 1 remap). The main arm treats verbal_response == 0 as missing.
- `full` and `icu_only` rows describe different populations (the ICU subgroup is selected on severity - a collider). Do not compare those AUROCs to each other.
- A delta whose 95% CI crosses zero is a null result. Report it as 'no detectable improvement at this sample size', not as a gain.
