# Zhang vs MIMIC SHAP comparison

Produced by `scripts/09c_zhang_mimic_shap_correlation.py`. This addresses the review finding that SHAP had only been run on the Zhang cohort.

## Headline (lead with this)

- **Qualitative claim - the defensible cross-cohort statement:** the GCS family ranks in the top tier of BOTH cohorts independently. Zhang ranks `GCS` #1, `eye_opening` #7, `movement` #18; MIMIC full cohort ranks `eye_opening` #5, `movement` #9, `verbal_response` #26 (of 37), and in the MIMIC ICU-only model the GCS components are the #1, #6 and #7 features overall.
- `GCS` (Zhang total) has no MIMIC counterpart feature by design - `gcs_total` is excluded because it is a placeholder artefact where verbal is untestable (method note 5 in script 09), so the comparison uses the three components.
- Outside the GCS family the cohorts do NOT preserve feature order (e.g. `congestive_heart_failure` is Zhang #6 but MIMIC #37); consistent with the external-validation collapse (0.5235 / 0.5995 AUROC).

## Rank statistics (secondary - do NOT quote the cross-cohort rho)

- Cross-cohort Spearman on the 7 shared features: -0.286 (vs MIMIC full), -0.393 (vs MIMIC ICU-only). **n = 7 is too small for this to be evidence in either direction - the sign can flip with one feature entering or leaving the overlap.** A reviewer will do the n-arithmetic in seconds; do not present these values as agreement or disagreement. Recorded for completeness only.
- Within-MIMIC stability IS reportable (full 37-feature rankings, not limited by the Zhang truncation): **Spearman rho = 0.863** (full cohort vs ICU-only).

## Per-feature table

```\n                            zhang_feature             mimic_feature  zhang_rank  zhang_mean_abs_shap  mimic_rank_full  mimic_mean_abs_shap_full  mimic_rank_icu  mimic_mean_abs_shap_icu                                                                                                                 note
                                      GCS                      None           1             0.750208              NaN                       NaN             NaN                      NaN MIMIC excludes gcs_total (placeholder artefact where verbal is untestable); MIMIC uses the three components instead.
moderate_to_severe_chronic_kidney_disease    moderate_to_severe_ckd           2             0.647674             21.0                  0.071437            22.0                 0.090149                                                                                                                     
                            liver_disease             liver_disease           5             0.366646             31.0                  0.007836            34.0                 0.006523                                                                                                                     
                 congestive_heart_failure  congestive_heart_failure           6             0.357083             37.0                  0.000000            36.0                 0.000000                                                                                                                     
                              eye_opening               eye_opening           7             0.337409              5.0                  0.181492             7.0                 0.273795                                                                                                                     
                brain_natriuretic_peptide brain_natriuretic_peptide          13             0.186233             24.0                  0.054001            25.0                 0.050299                                                                                                                     
                                   gender                    gender          14             0.169888             27.0                  0.034294            30.0                 0.022595                                                                                                                     
                                 movement                  movement          18             0.132879              9.0                  0.138339             6.0                 0.291104                                                                                                                     \n```\n\n## Limitations (read before quoting this)

1. `results/tables/shap_importance.csv` holds only Zhang's TOP 20 of 143 features. Ranks below #20 are unknown, so the cross-cohort rho uses only the shared subset of those 20 and is **descriptive, not inferential**.
2. Mean |SHAP| values are NOT comparable across cohorts (different models, different outcome prevalence, different units). Only RANKS are compared.
3. A feature can be absent from the shared set purely because the two cohorts measured different things (Zhang has echo and blood-gas variables; MIMIC has MIMIC labs). Absence is not disagreement.
4. External validation AUROC already collapsed (0.5235 / 0.5995), so any cross-cohort narrative must stay modest: the defensible claim is about which features each cohort's model leaned on, not that they agree.
5. For a full-feature comparison, re-run SHAP on the Zhang models (`models/*.pkl`) with `dat.csv` from the dissertation bundle and replace `shap_importance.csv` with the full 143-feature ranking.
