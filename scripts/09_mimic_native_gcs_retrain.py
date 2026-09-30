"""
09_mimic_native_gcs_retrain.py
===============================
MIMIC-IV native heart-failure model, with and without GCS, plus SHAP.

This closes the "SHAP was only run on the Zhang cohort" gap and produces
the feature-level cross-cohort comparison for the preprint.

READS : data/mimic_hf_with_gcs.csv       (produced by 08_mimic_gcs_extraction.py)
        data/mimic_icu_stay_flags.csv    (produced by 09a_icu_stay_flags.py)
WRITES: outputs/mimic_gcs/                       (git-ignored)
        results/tables/mimic_gcs_retrain_results.csv
        results/tables/shap_importance_full_cohort.csv
        results/tables/shap_importance_icu_only.csv
        results/tables/legacy_24_features.json
        results/figures/shap_bar_full_cohort.png      shap_beeswarm_full_cohort.png
        results/figures/shap_bar_icu_only.png         shap_beeswarm_icu_only.png
        results/mimic_gcs_run_log.md

    RUNTIME: this is a heavy job (see "How long does it take?" at the
    bottom). Run it in Colab, not on a laptop. The Colab entry point is
    scripts/run_gcs_colab.py.

USAGE
-----
    python scripts/09_mimic_native_gcs_retrain.py                 # full run
    python scripts/09_mimic_native_gcs_retrain.py --quick         # ~25% smoke test
    python scripts/09_mimic_native_gcs_retrain.py --no-cv \
        --bootstrap 200 --n-estimators 200                         # fast pass

--------------------------------------------------------------------------
METHOD NOTES -- the decisions that matter, and why
--------------------------------------------------------------------------
1. GROUPED SPLIT (not train_test_split).

   The cohort is 42,990 ADMISSIONS but only 18,890 unique patients:
   8,006 patients have more than one HF admission (the busiest has 59).
   A row-level random split would put the same patient in both train and
   test via different admissions, letting the model recognise patients
   rather than learn generalisable risk. Every AUROC below is therefore
   computed on a patient-level split: GroupShuffleSplit for the holdout,
   GroupKFold for cross-validation. The script ASSERTS zero patient
   overlap and aborts if that fails.

   This is a structural fix. Re-running with a different random seed does
   NOT fix leakage and must not be used as a substitute.

2. GCS IS LEFT AS NATIVE NaN, NEVER IMPUTED.

   GCS is missing for ~72% of the cohort because MIMIC derives it from ICU
   chartevents, and most HF admissions here are ward-only. That is
   structural: these patients were never assessed. Median-imputing GCS=15
   would assert "this patient was neurologically normal" about people who
   were never examined. XGBoost's native missing-value handling learns a
   separate branch instead, which is the honest representation.

3. ALL OTHER IMPUTATION IS MEDIAN, FIT ON TRAIN ONLY.

   The medians are computed on the training rows and then applied to the
   test rows. Computing them on the full cohort would leak test-set
   information into training, which is the same class of error as the
   patient leakage above, just smaller.

4. HAD_ICU_STAY CONTROL - TRUE ICU FLAG, NOT "GCS OBSERVED".

   Because ~72% of GCS is missing, "adding GCS" also adds a near-perfect
   proxy for "was this patient ever in the ICU". If the gain is really
   about ICU admission rather than neurological status, the control model
   (baseline + had_icu_stay) will match the with-GCS model. We report
   that control whichever way it comes out.

   The flag must be INDEPENDENT of GCS charting to be a valid control.
   The first draft used `gcs_total.notna()`, which is degenerate: it is
   exactly "a GCS score was charted". The LLM-council review flagged this,
   so `had_icu_stay` now comes from MIMIC-IV's `icustays` table via
   scripts/09a_icu_stay_flags.py - one honest 0/1 per admission. The
   `expanded_plus_icu_flag_plus_gcs` variant then answers the real
   question: does GCS add anything BEYOND ICU admission? The ICU-only
   subgroup is likewise defined by this true flag.

5. gcs_total IS NOT USED AS A FEATURE.

   Verified against the extracted data: for all 1,111 rows where
   verbal_response == 0, gcs_total is 15 regardless of the components
   (e.g. eyes=1, motor=1, verbal=0/1). MIMIC uses gcs_unable as a
   placeholder for "verbal untestable (intubated)", not as a real score,
   so gcs_total=15 there is not a measurement. Using it would hand the
   model a fabricated "normal neurology" value for the sickest patients.
   We use the three components only. verbal_response == 0 is MIMIC's
   "untestable (intubated)" code, not a measurement, so the MAIN encoding
   treats it as missing (native NaN). The older 0 -> 1 remap is a fabricated
   value ("worst verbal score") and survives only as a sensitivity arm
   (`expanded_plus_gcs_v0as1`); if that arm matches the main arm, the
   encoding choice is immaterial and we say so. The extraction script 08 is
   left untouched because its SQL is a faithful passthrough of the MIMIC
   derived table; the correction belongs here, in the modelling layer.

6. NO SMOTE.

   6-month mortality prevalence here is ~22%, not the ~2.8% of the Zhang
   cohort. Synthetic oversampling would treat a problem this cohort does
   not have. Class weighting (scale_pos_weight) is reported as a
   sensitivity check instead, with the unweighted result as the headline.

7. FULL COHORT AND ICU-ONLY ARE DIFFERENT POPULATIONS.
   The full-cohort model answers "does adding GCS help on all HF
   admissions, where GCS is mostly absent?". The ICU-only model answers
   "does GCS help among patients who were actually assessed?". Both are
   legitimate; they must not be conflated or compared directly.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import xgboost as xgb
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SEED = 17
TEST_SIZE = 0.20

DATA_CSV = os.path.join("data", "mimic_hf_with_gcs.csv")
OUT_DIR = os.path.join("outputs", "mimic_gcs")
RES_TABLES = os.path.join("results", "tables")
RES_FIGS = os.path.join("results", "figures")
RUN_LOG = os.path.join("results", "mimic_gcs_run_log.md")
LEGACY_JSON = os.path.join(RES_TABLES, "legacy_24_features.json")

# The 24 features used by the original MIMIC-only benchmark
# (09_mimic_only_xgboost.py), the script that produced the historical
# 0.801 AUROC. Recovered from that script's FEATURES list.
LEGACY_24 = [
    "age", "length_of_stay", "discharge_day", "creatinine", "potassium",
    "sodium", "hematocrit", "hemoglobin", "platelets", "wbc", "magnesium",
    "phosphorus", "urea_nitrogen", "glucose", "calcium", "albumin", "alt",
    "ast", "alkaline_phosphatase", "chronic_kidney_disease", "liver_disease",
    "diabetes", "hypertension", "atrial_fibrillation",
]

GCS_COMPONENTS = ["eye_opening", "verbal_response", "movement"]
# Sensitivity encoding for verbal_response == 0 ("untestable/intubated"):
# remapped to 1 ("no response") instead of missing. Method note 5.
GCS_V0AS1 = ["eye_opening", "verbal_response_v0as1", "movement"]

SHARED_FEATURES = [
    "age", "gender", "myocardial_infarction", "congestive_heart_failure",
    "peripheral_vascular_disease", "cerebrovascular_disease", "dementia",
    "copd", "peptic_ulcer_disease", "diabetes", "moderate_to_severe_ckd",
    "liver_disease", "solid_tumor", "aids", "hemiplegia", "cci_score",
    "creatinine", "urea", "sodium", "potassium", "chloride", "bicarbonate",
    "albumin", "calcium", "anion_gap", "hemoglobin", "platelet",
    "white_blood_cell", "hematocrit", "mcv", "rdw", "red_blood_cell",
    "brain_natriuretic_peptide", "troponin_t",
]

TARGETS = {
    "28d_death": "death_within_28days",
    "6m_death": "death_within_6months",
}

_T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------
def prepare_data(quick=False, log_lines=None):
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(
            f"{DATA_CSV} not found. Run 08_mimic_gcs_extraction.py first "
            f"(or let scripts/run_gcs_colab.py fetch it)."
        )
    df = pd.read_csv(DATA_CSV)
    log(f"Loaded {DATA_CSV}: {df.shape[0]:,} rows x {df.shape[1]} cols")

    if quick:
        # Subsample by PATIENT, not by row, so the grouped-split logic is
        # still exercised the same way it will be on the full cohort.
        rng = np.random.RandomState(SEED)
        patients = df["subject_id"].unique()
        keep_patients = set(rng.choice(patients, size=int(0.5 * len(patients)),
                                       replace=False))
        df = df[df["subject_id"].isin(keep_patients)].reset_index(drop=True)
        log(f"--quick: patient-level 50% subsample -> {len(df):,} rows")

    # verbal_response == 0 is MIMIC's "untestable (intubated)" code (method
    # note 5). MAIN encoding: treat as missing. The 0 -> 1 remap is kept only
    # as the sensitivity column `verbal_response_v0as1`.
    mask_v0 = df["verbal_response"] == 0
    n_v0 = int(mask_v0.sum())
    n_intub = int((mask_v0 & (df["gcs_intubated_flag"] == 1)).sum())
    n_other = n_v0 - n_intub
    df["verbal_response_v0as1"] = df["verbal_response"].replace(0, 1)
    df.loc[mask_v0, "verbal_response"] = np.nan
    log(f"verbal_response == 0 -> NaN (untestable) in {n_v0:,} rows "
        f"({n_intub:,} flagged intubated, {n_other:,} not); "
        f"0 -> 1 remap kept as sensitivity column verbal_response_v0as1")

    # True ICU-stay flag from icustays (method note 4). This is the honest
    # control: independent of whether a GCS score was charted.
    flags_path = os.path.join("data", "mimic_icu_stay_flags.csv")
    if not os.path.exists(flags_path):
        raise FileNotFoundError(
            f"{flags_path} not found. Run scripts/09a_icu_stay_flags.py first "
            f"(one small BigQuery pull). The old gcs_total.notna() shortcut is "
            f"degenerate and is no longer used.")
    icu_flags = pd.read_csv(flags_path)
    n_before = len(df)
    df = df.merge(icu_flags[["hadm_id", "had_icu_stay"]], on="hadm_id", how="left")
    assert len(df) == n_before, "ICU flag merge changed the row count"
    assert df["had_icu_stay"].notna().all(), "missing had_icu_stay for some admissions"
    df["had_icu_stay"] = df["had_icu_stay"].astype(int)
    gcs_observed = df["gcs_total"].notna()
    assert (df.loc[gcs_observed, "had_icu_stay"] == 1).all(), \
        "GCS observed on an admission with no ICU stay - extraction mismatch"
    log(f"had_icu_stay (true, from icustays): {int(df['had_icu_stay'].sum()):,} "
        f"ICU admissions ({100 * df['had_icu_stay'].mean():.1f}%); "
        f"GCS charted on {int(gcs_observed.sum()):,}")

    if log_lines is not None:
        n_pat = df["subject_id"].nunique()
        vc = df["subject_id"].value_counts()
        log_lines.append(
            f"- **{len(df):,} admissions** from **{n_pat:,} unique patients**; "
            f"**{int((vc > 1).sum()):,}** patients have more than one HF admission "
            f"(max {int(vc.max())}). All splits are grouped on `subject_id`, and "
            f"the script asserts zero patient overlap before scoring."
        )
        n_icu = int(df["had_icu_stay"].sum())
        n_gcs_obs = int(df["gcs_total"].notna().sum())
        log_lines.append(
            f"- **{n_icu:,}** admissions had a true ICU stay (flag from "
            f"`icustays` via `09a_icu_stay_flags.py`); GCS was charted for "
            f"**{n_gcs_obs:,}**. GCS is left as native NaN for the rest - "
            f"they were never assessed, and imputing a score would fabricate a "
            f"clinical claim."
        )
        log_lines.append(
            f"- `verbal_response == 0` (MIMIC's \"untestable/intubated\" code) is "
            f"treated as **missing** in **{n_v0:,}** rows ({n_intub:,} flagged "
            f"intubated, {n_other:,} not). The old 0 -> 1 remap is a fabricated "
            f"value and survives only as the `expanded_plus_gcs_v0as1` "
            f"sensitivity arm. `gcs_total` is excluded as a feature entirely: "
            f"for all {n_v0:,} of these rows it reads 15 regardless of the "
            f"components (e.g. eyes=1, motor=1), i.e. a placeholder artefact."
        )
    return df


# --------------------------------------------------------------------------
# Modelling helpers
# --------------------------------------------------------------------------
def build_model(n_estimators, scale_pos_weight=None, random_state=SEED):
    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
    )
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


def impute_train_only(X_train_raw, X_other_raw, features, gcs_features):
    """
    Median-impute every non-GCS column, with medians computed on the
    TRAINING rows only (method note 3). GCS columns are left untouched so
    their NaN reaches XGBoost's native missing-value handling.
    """
    impute_cols = [c for c in features if c not in gcs_features]
    medians = X_train_raw[impute_cols].median()
    X_train = X_train_raw.copy()
    X_other = X_other_raw.copy()
    for c in impute_cols:
        X_train[c] = X_train[c].fillna(medians[c])
        X_other[c] = X_other[c].fillna(medians[c])
    return X_train, X_other, medians


def fit_predict(X_train, y_train, X_test, n_estimators, scale_pos_weight=None):
    model = build_model(n_estimators, scale_pos_weight=scale_pos_weight)
    model.fit(X_train, y_train)
    return model, model.predict_proba(X_test)[:, 1]


def cv_auc(X_raw, y, groups, n_estimators, gcs_features, n_splits=5):
    """
    GroupKFold CV so no patient spans folds. Median imputation is refit
    INSIDE each fold (train rows only), so a CV validation row never
    influences its own imputation medians - the same rule as the holdout.
    """
    if n_splits < 2:
        return np.nan, np.nan
    gkf = GroupKFold(n_splits=n_splits)
    features = list(X_raw.columns)
    scores = []
    for tr_i, va_i in gkf.split(X_raw, y, groups=groups):
        Xtr, Xva, _ = impute_train_only(X_raw.iloc[tr_i], X_raw.iloc[va_i],
                                        features, gcs_features)
        model = build_model(n_estimators)
        model.fit(Xtr, y.iloc[tr_i])
        p = model.predict_proba(Xva)[:, 1]
        if len(np.unique(y.iloc[va_i])) > 1:
            scores.append(roc_auc_score(y.iloc[va_i], p))
    return float(np.mean(scores)), float(np.std(scores))


def bootstrap_delta_ci(y_test, prob_base, prob_new, n_boot, seed=SEED):
    """95% CI on AUROC(new) - AUROC(base) via a PAIRED bootstrap of the test set."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y_test)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], prob_new[idx])
                      - roc_auc_score(y[idx], prob_base[idx]))
    if not deltas:
        return np.nan, np.nan, 0
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), len(deltas)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (y_prob >= lo) & ((y_prob < hi) if i < n_bins - 1 else (y_prob <= hi))
        if m.sum() == 0:
            continue
        e += (m.sum() / len(y_true)) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(e)


def split_grouped(df, y, seed=SEED, test_size=TEST_SIZE):
    """GroupShuffleSplit on subject_id, with a hard leakage assertion."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, y, groups=df["subject_id"]))
    overlap = set(df["subject_id"].iloc[tr]) & set(df["subject_id"].iloc[te])
    assert not overlap, f"PATIENT LEAKAGE: {len(overlap)} patients in both splits"
    return tr, te


def md_table(df):
    """Markdown table if tabulate is installed, else a plain fixed-width dump."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def tree_shap_values(model, X):
    """
    Exact TreeSHAP values, shape (n_rows, n_features).

    Uses XGBoost's own `pred_contribs=True`, which is the most portable route
    across shap/xgboost version combinations. (shap 0.49 + xgboost 3.2 raises
    "could not convert string to float" inside TreeExplainer because of the
    vector-valued base_score, so shap.TreeExplainer is kept only as a
    fallback.) The final column is the bias term and is dropped.
    """
    booster = model.get_booster()
    try:
        contribs = np.asarray(
            booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)),
                            pred_contribs=True))
        return contribs[:, :-1], contribs[:, -1]
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"  [shap] native pred_contribs failed ({exc}); using TreeExplainer")
        sv = np.asarray(shap.TreeExplainer(booster)(X))
        return sv, np.zeros(len(sv))


def shap_importance(model, X, features):
    """TreeSHAP mean |SHAP| per feature, sorted high to low."""
    sv, base = tree_shap_values(model, X)
    return sv, base, pd.DataFrame({
        "feature": list(features),
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)


def plot_shap(sv, base, X, features, title, tag):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap.Explanation(values=sv, base_values=base, data=X), X,
                      feature_names=list(features), show=False, max_display=15)
    plt.title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_FIGS, f"shap_beeswarm_{tag}.png"),
                dpi=130, bbox_inches="tight")
    plt.close()

    imp = pd.DataFrame({
        "feature": list(features),
        "mean_abs_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=True).tail(15)
    colors = ["#e76f51" if f in GCS_COMPONENTS else "#2a9d8f" for f in imp.feature]
    plt.figure(figsize=(10, 6))
    plt.barh(imp.feature, imp.mean_abs_shap, color=colors)
    plt.xlabel("mean |SHAP value|")
    plt.title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_FIGS, f"shap_bar_{tag}.png"),
                dpi=130, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="MIMIC-native GCS retrain + SHAP")
    p.add_argument("--quick", action="store_true", help="patient-level 50%% subsample")
    p.add_argument("--no-cv", action="store_true", help="skip GroupKFold CV (faster)")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--n-estimators", type=int, default=400)
    args = p.parse_args()

    for d in (OUT_DIR, RES_TABLES, RES_FIGS):
        os.makedirs(d, exist_ok=True)

    log("=" * 72)
    log("09 - MIMIC-IV NATIVE GCS RETRAIN + SHAP")
    log(f"  quick={args.quick}  seed={SEED}  test_size={TEST_SIZE}  "
        f"n_estimators={args.n_estimators}  bootstrap={args.bootstrap}  "
        f"cv={'off' if args.no_cv else str(args.cv_folds) + ' GroupKFold'}")
    log("=" * 72)

    log_lines = []
    df = prepare_data(quick=args.quick, log_lines=log_lines)

    # ---- legacy 24 availability (plan Option A vs B) ----------------------
    legacy_ok = [f for f in LEGACY_24 if f in df.columns]
    legacy_missing = [f for f in LEGACY_24 if f not in df.columns]
    with open(LEGACY_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "legacy_24": LEGACY_24,
            "recovered_from": "scripts/09_mimic_only_xgboost.py (FEATURES list)",
            "available_in_mimic_hf_with_gcs": legacy_ok,
            "missing_in_mimic_hf_with_gcs": legacy_missing,
            "n_available": len(legacy_ok),
            "n_missing": len(legacy_missing),
            "decision": (
                f"Only {len(legacy_ok)} of 24 legacy columns exist in the GCS "
                f"extract, so a faithful legacy-24 rerun is impossible on this "
                f"cohort without re-running the extraction. We use the EXPANDED "
                f"BASELINE instead and do not label it 'the 0.801 model'."
            ),
        }, f, indent=2)
    log(f"\nLegacy 24-feature list -> {LEGACY_JSON}")
    log(f"  available in new extract: {len(legacy_ok)}/24")
    log(f"  missing in new extract : {len(legacy_missing)}/24")
    log("  => comparison baseline is the EXPANDED BASELINE (not the 0.801 model).")
    log_lines.append(
        f"- Legacy 24-feature list **recovered** from `09_mimic_only_xgboost.py` "
        f"and saved to `results/tables/legacy_24_features.json`."
    )
    log_lines.append(
        f"- Only **{len(legacy_ok)}/24** of those columns exist in "
        f"`mimic_hf_with_gcs.csv`. Missing: {', '.join(legacy_missing)}. A faithful "
        f"legacy-24 rerun is therefore not possible on this cohort, so the "
        f"comparison baseline is the **expanded baseline** "
        f"({len(SHARED_FEATURES)} shared features). Its AUROC is a superset model "
        f"and must **not** be described as 'the 0.801 model'."
    )

    results = []

    for target_label, target_col in TARGETS.items():
        y = df[target_col].astype(int)
        log(f"\n{'-' * 72}")
        log(f"OUTCOME {target_label}   prevalence {y.mean() * 100:.2f}%")

        tr, te = split_grouped(df, y)
        log(f"  train {len(tr):,} / test {len(te):,} rows | "
            f"{df['subject_id'].nunique():,} patients | LEAKAGE CHECK PASSED")

        variants = {
            "expanded_baseline": (SHARED_FEATURES, ()),
            "expanded_plus_gcs": (SHARED_FEATURES + GCS_COMPONENTS,
                                  tuple(GCS_COMPONENTS)),
            "baseline_plus_icu_flag": (SHARED_FEATURES + ["had_icu_stay"], ()),
            "expanded_plus_icu_flag_plus_gcs": (
                SHARED_FEATURES + ["had_icu_stay"] + GCS_COMPONENTS,
                tuple(GCS_COMPONENTS)),
            "expanded_plus_gcs_v0as1": (SHARED_FEATURES + GCS_V0AS1,
                                        tuple(GCS_V0AS1)),
        }

        probs, aucs = {}, {}
        for tag, (feats, gcs_feats) in variants.items():
            Xtr, Xte, _ = impute_train_only(
                df[feats].iloc[tr], df[feats].iloc[te], feats, gcs_feats)
            ytr, yte = y.iloc[tr], y.iloc[te]
            model, prob = fit_predict(Xtr, ytr, Xte, args.n_estimators)
            auc = roc_auc_score(yte, prob)
            probs[tag], aucs[tag] = prob, auc
            if args.no_cv:
                cv_m = cv_s = np.nan
            else:
                cv_m, cv_s = cv_auc(df[feats].iloc[tr], y.iloc[tr],
                                    df["subject_id"].iloc[tr],
                                    args.n_estimators, gcs_feats,
                                    args.cv_folds)
            results.append({
                "model_tag": tag, "target_col": target_col, "cohort": "full",
                "n_features": len(feats),
                "cv_auc_mean": None if np.isnan(cv_m) else round(cv_m, 4),
                "cv_auc_std": None if np.isnan(cv_s) else round(cv_s, 4),
                "test_auc": round(auc, 4),
                "auc_delta_vs_baseline": None, "delta_95ci_low": None,
                "delta_95ci_high": None,
                "n_test": int(len(yte)), "n_positive_test": int(yte.sum()),
            })
            log(f"  {tag:24s} n_feat={len(feats):2d}  "
                f"CV={'  n/a ' if np.isnan(cv_m) else f'{cv_m:.3f}'}  "
                f"test_auc={auc:.4f}")

        yte = y.iloc[te]
        base_auc = aucs["expanded_baseline"]

        for tag in ["expanded_plus_gcs", "baseline_plus_icu_flag",
                    "expanded_plus_icu_flag_plus_gcs", "expanded_plus_gcs_v0as1"]:
            lo, hi, n_ok = bootstrap_delta_ci(
                yte, probs["expanded_baseline"], probs[tag], args.bootstrap)
            delta = aucs[tag] - base_auc
            for r in results:
                if r["model_tag"] == tag and r["target_col"] == target_col:
                    r["auc_delta_vs_baseline"] = round(delta, 4)
                    r["delta_95ci_low"] = round(lo, 4)
                    r["delta_95ci_high"] = round(hi, 4)
            sig = "" if lo <= 0 <= hi else "  <- CI excludes 0"
            log(f"  delta {tag:32s} {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]{sig}")

        # Encoding sensitivity: does the verbal-0 remap change anything?
        lo_s, hi_s, _ = bootstrap_delta_ci(
            yte, probs["expanded_plus_gcs"], probs["expanded_plus_gcs_v0as1"],
            args.bootstrap)
        d_s = aucs["expanded_plus_gcs_v0as1"] - aucs["expanded_plus_gcs"]
        sig_s = "" if lo_s <= 0 <= hi_s else "  <- CI excludes 0"
        log(f"  [sensitivity] verbal 0->1 remap vs NaN: {d_s:+.4f}  "
            f"95% CI [{lo_s:+.4f}, {hi_s:+.4f}]{sig_s}")
        log_lines.append(
            f"- `{target_label}` verbal-0 encoding sensitivity: 0->1 remap vs "
            f"missing gives delta {d_s:+.4f} (95% CI [{lo_s:+.4f}, {hi_s:+.4f}]). "
            f"GCS-beyond-ICU comparison: `expanded_plus_icu_flag_plus_gcs` "
            f"{aucs['expanded_plus_icu_flag_plus_gcs']:.4f} vs "
            f"`baseline_plus_icu_flag` {aucs['baseline_plus_icu_flag']:.4f}."
        )

        log_lines.append(
            f"- `{target_label}` full cohort: expanded baseline AUROC "
            f"{base_auc:.4f}; +GCS {aucs['expanded_plus_gcs']:.4f} "
            f"(delta {aucs['expanded_plus_gcs'] - base_auc:+.4f}); "
            f"baseline+`had_icu_stay` control {aucs['baseline_plus_icu_flag']:.4f} "
            f"(delta {aucs['baseline_plus_icu_flag'] - base_auc:+.4f})."
        )
        yte_np = np.asarray(yte)
        pg = probs["expanded_plus_gcs"]
        log_lines.append(
            f"- `{target_label}` with-GCS calibration: AUPRC "
            f"{average_precision_score(yte_np, pg):.4f}, Brier "
            f"{brier_score_loss(yte_np, pg):.4f}, ECE(10 bins) "
            f"{expected_calibration_error(yte_np, pg):.4f}."
        )

        # scale_pos_weight sensitivity (no SMOTE - see method note 6)
        feats = SHARED_FEATURES + GCS_COMPONENTS
        Xtr, Xte, _ = impute_train_only(
            df[feats].iloc[tr], df[feats].iloc[te], feats, GCS_COMPONENTS)
        ytr = y.iloc[tr]
        spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
        _, prob_spw = fit_predict(Xtr, ytr, Xte, args.n_estimators,
                                  scale_pos_weight=spw)
        auc_spw = roc_auc_score(yte, prob_spw)
        log(f"  [sensitivity] scale_pos_weight={spw:.2f}: "
            f"with-GCS AUROC {auc_spw:.4f} vs unweighted {aucs['expanded_plus_gcs']:.4f}")
        log_lines.append(
            f"- `{target_label}` scale_pos_weight sensitivity "
            f"(spw={spw:.2f}): with-GCS AUROC {auc_spw:.4f} vs "
            f"{aucs['expanded_plus_gcs']:.4f} unweighted. Prevalence is only "
            f"{y.mean() * 100:.1f}%, so weighting is not necessary; the "
            f"unweighted number is the headline."
        )

        # ---- ICU-only subgroup (6m_death only) ----------------------------
        if target_label == "6m_death":
            icu = df[df["had_icu_stay"] == 1].reset_index(drop=True)
            y_icu = icu[target_col].astype(int)
            n_gcs_icu = int(icu["gcs_total"].notna().sum())
            log(f"\n  --- ICU-only subgroup (true ICU flag): {len(icu):,} "
                f"admissions, prevalence {y_icu.mean() * 100:.2f}%, "
                f"GCS charted for {n_gcs_icu:,} ---")
            tr2, te2 = split_grouped(icu, y_icu)
            log(f"    train {len(tr2):,} / test {len(te2):,} | LEAKAGE CHECK PASSED")

            f_gcs = SHARED_FEATURES + GCS_COMPONENTS
            A, B, _ = impute_train_only(icu[f_gcs].iloc[tr2], icu[f_gcs].iloc[te2],
                                       f_gcs, GCS_COMPONENTS)
            _, p_gcs = fit_predict(A, y_icu.iloc[tr2], B, args.n_estimators)
            auc_icu_gcs = roc_auc_score(y_icu.iloc[te2], p_gcs)

            f_base = SHARED_FEATURES
            A2, B2, _ = impute_train_only(icu[f_base].iloc[tr2], icu[f_base].iloc[te2],
                                          f_base, GCS_COMPONENTS)
            _, p_base = fit_predict(A2, y_icu.iloc[tr2], B2, args.n_estimators)
            auc_icu_base = roc_auc_score(y_icu.iloc[te2], p_base)

            lo2, hi2, _ = bootstrap_delta_ci(
                y_icu.iloc[te2], p_base, p_gcs, args.bootstrap)
            d2 = auc_icu_gcs - auc_icu_base
            log(f"    ICU-only: no-GCS={auc_icu_base:.4f}  with-GCS={auc_icu_gcs:.4f}"
                f"  delta={d2:+.4f}  95% CI [{lo2:+.4f}, {hi2:+.4f}]")

            results.append({
                "model_tag": "expanded_baseline_icu_only", "target_col": target_col,
                "cohort": "icu_only", "n_features": len(f_base),
                "cv_auc_mean": None, "cv_auc_std": None,
                "test_auc": round(auc_icu_base, 4),
                "auc_delta_vs_baseline": None, "delta_95ci_low": None,
                "delta_95ci_high": None,
                "n_test": int(len(te2)), "n_positive_test": int(y_icu.iloc[te2].sum()),
            })
            results.append({
                "model_tag": "expanded_plus_gcs_icu_only", "target_col": target_col,
                "cohort": "icu_only", "n_features": len(f_gcs),
                "cv_auc_mean": None, "cv_auc_std": None,
                "test_auc": round(auc_icu_gcs, 4),
                "auc_delta_vs_baseline": round(d2, 4),
                "delta_95ci_low": round(lo2, 4), "delta_95ci_high": round(hi2, 4),
                "n_test": int(len(te2)), "n_positive_test": int(y_icu.iloc[te2].sum()),
            })
            log_lines.append(
                f"- **ICU-only subgroup** (`6m_death`, true ICU flag, "
                f"n={len(icu):,}, prevalence {y_icu.mean() * 100:.1f}%, "
                f"GCS charted for {n_gcs_icu:,}): no-GCS AUROC "
                f"{auc_icu_base:.4f} vs with-GCS {auc_icu_gcs:.4f}, delta "
                f"{d2:+.4f} (95% CI [{lo2:+.4f}, {hi2:+.4f}]). Different "
                f"population from the full cohort - do not compare the two "
                f"AUROCs directly. ICU admission depends on severity, so this "
                f"subgroup is selected on a collider; treat it as descriptive."
            )

            # SHAP: ICU-only, with-GCS, on the test rows
            mdl_icu, _ = fit_predict(A, y_icu.iloc[tr2], B, args.n_estimators)
            sv_icu, base_icu, imp_icu = shap_importance(mdl_icu, B, f_gcs)
            imp_icu.to_csv(os.path.join(RES_TABLES, "shap_importance_icu_only.csv"),
                           index=False)
            plot_shap(sv_icu, base_icu, B, f_gcs,
                      f"6-month mortality - MIMIC ICU-only (n={len(icu):,})", "icu_only")
            log("    SHAP top 5 (ICU-only):")
            for _, r in imp_icu.head(5).iterrows():
                log(f"      {r['feature']:28s} {r['mean_abs_shap']:.4f}")

    # ---- SHAP: full cohort, 6m_death, with-GCS ----------------------------
    log(f"\n{'-' * 72}\nSHAP: 6m_death, with-GCS, FULL cohort")
    feats = SHARED_FEATURES + GCS_COMPONENTS
    y6 = df["death_within_6months"].astype(int)
    tr3, te3 = split_grouped(df, y6)
    Ftr, Fte, _ = impute_train_only(
        df[feats].iloc[tr3], df[feats].iloc[te3], feats, GCS_COMPONENTS)
    mdl_full, _ = fit_predict(Ftr, y6.iloc[tr3], Fte, args.n_estimators)
    sv_full, base_full, imp_full = shap_importance(mdl_full, Fte, feats)
    imp_full.to_csv(os.path.join(RES_TABLES, "shap_importance_full_cohort.csv"),
                    index=False)
    plot_shap(sv_full, base_full, Fte, feats,
              f"6-month mortality - MIMIC full cohort (n={len(df):,})", "full_cohort")
    log("  top 8 features (full cohort):")
    for _, r in imp_full.head(8).iterrows():
        star = "  <- GCS" if r["feature"] in GCS_COMPONENTS else ""
        log(f"    {r['feature']:28s} {r['mean_abs_shap']:.4f}{star}")
    gcs_ranks = [f for f in imp_full.feature if f in GCS_COMPONENTS]
    if gcs_ranks:
        strongest = imp_full[imp_full.feature.isin(GCS_COMPONENTS)].iloc[0]
        log_lines.append(
            f"- SHAP full cohort (`6m_death`): GCS components rank "
            f"{', '.join(gcs_ranks)} of {len(feats)}; strongest GCS feature is "
            f"`{strongest['feature']}` (mean |SHAP| {strongest['mean_abs_shap']:.4f})."
        )

    # ---- save -------------------------------------------------------------
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT_DIR, "mimic_gcs_retrain_results.csv"), index=False)
    res.to_csv(os.path.join(RES_TABLES, "mimic_gcs_retrain_results.csv"), index=False)

    log(f"\n{'=' * 72}\nRESULTS\n{res.to_string(index=False)}\n{'=' * 72}")

    with open(RUN_LOG, "w", encoding="utf-8") as f:
        f.write("# MIMIC GCS retrain - run log\n\n")
        f.write(f"Produced by `scripts/09_mimic_native_gcs_retrain.py` "
                f"(seed={SEED}, patient-level grouped split, "
                f"n_estimators={args.n_estimators}, bootstrap={args.bootstrap}, "
                f"quick={args.quick}).\n\n")
        f.write("## Pre-registered expectations (text fixed in the script "
                "before any of these numbers were produced)\n\n")
        f.write("- The historical MIMIC benchmark (0.801 AUROC for 6m, 0.867 "
                "for 28d; `results/tables/mimic_only_xgboost_results.csv`) "
                "came from `09_mimic_only_xgboost.py` using a ROW-level "
                "`train_test_split` on a cohort where 8,006 patients have "
                "several HF admissions. The same patient could land in train "
                "and test via different admissions, so those numbers are "
                "leakage-inflated. EXPECT the grouped-split AUROCs below to "
                "be LOWER; that is a correction, not a regression.\n")
        f.write("- Zhang-to-MIMIC external validation already collapsed "
                "(0.5235 / 0.5995 AUROC; "
                "`results/tables/external_validation_results.csv`), so "
                "cross-cohort claims must stay modest regardless of what "
                "this run shows.\n")
        f.write("- The primary quantity is the PAIRED DELTA (with-GCS vs "
                "baseline on the same test patients), not the absolute "
                "AUROC.\n")
        f.write("\n## Data and design checks\n\n")
        for line in log_lines:
            f.write(line + "\n")
        f.write("\n## Results\n\n")
        f.write(md_table(res))
        f.write("\n\n## How to read this\n\n")
        f.write("- Every AUROC uses a patient-level split on `subject_id`; the "
                "script asserts zero patient overlap before scoring.\n")
        f.write('- "expanded_baseline" is a superset model, **not** the historical '
                "0.801 model (see the legacy-24 note above).\n")
        f.write("- GCS is never imputed; the ~72% missingness is left to XGBoost's "
                "native NaN handling.\n")
        f.write("- `baseline_plus_icu_flag` is the control, using the TRUE "
                "ICU-stay flag from `icustays` (`09a_icu_stay_flags.py`), "
                "independent of GCS charting. If it matches "
                "`expanded_plus_gcs`, any gain is about ICU admission, not "
                "neurological status; `expanded_plus_icu_flag_plus_gcs` "
                "answers whether GCS adds anything BEYOND ICU admission.\n")
        f.write("- `expanded_plus_gcs_v0as1` is an encoding sensitivity arm "
                "(verbal 0 -> 1 remap). The main arm treats "
                "verbal_response == 0 as missing.\n")
        f.write("- `full` and `icu_only` rows describe different populations "
                "(the ICU subgroup is selected on severity - a collider). "
                "Do not compare those AUROCs to each other.\n")
        f.write("- A delta whose 95% CI crosses zero is a null result. Report it as "
                "'no detectable improvement at this sample size', not as a gain.\n")

    log(f"Saved: {os.path.join(OUT_DIR, 'mimic_gcs_retrain_results.csv')}")
    log(f"Saved: {os.path.join(RES_TABLES, 'shap_importance_full_cohort.csv')}")
    log(f"Saved: {os.path.join(RES_TABLES, 'shap_importance_icu_only.csv')}")
    log(f"Saved: {RUN_LOG}")


# --------------------------------------------------------------------------
# How long does it take?
# --------------------------------------------------------------------------
# Full run on Colab (CPU, ~2 vCPU): 5 variants x 5 CV folds per outcome
# (2 outcomes), plus the ICU-only pair, the sensitivity arm and 2 TreeSHAP
# passes. Expect roughly 20-50 minutes, dominated by the 5-fold CV and the
# 1,000-iteration paired bootstraps.
#
# If you only want the headline numbers fast:
#     python scripts/09_mimic_native_gcs_retrain.py --no-cv --bootstrap 200 \
#         --n-estimators 200
# If you only want to prove the code path works:
#     python scripts/09_mimic_native_gcs_retrain.py --quick --no-cv \
#         --bootstrap 50 --n-estimators 100
if __name__ == "__main__":
    main()
