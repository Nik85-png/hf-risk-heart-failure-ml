"""
09c_zhang_mimic_shap_correlation.py
===================================
Cross-cohort SHAP comparison: Zhang cohort vs MIMIC-IV, feature by feature.

Why this script exists
----------------------
The preprint's stated goal is cross-cohort validation. Zhang -> MIMIC
external AUROC collapsed (0.5235 / 0.5995), and SHAP had only been run on
the Zhang cohort, so there was NO feature-level comparison at all. Script 09
now produces MIMIC SHAP importances; this script aligns the two feature name
spaces and compares where the shared features land in each ranking.

LIMITATION (stated up front, and repeated in the output note)
------------------------------------------------------------
`results/tables/shap_importance.csv` preserves only the TOP 20 Zhang
features out of 143. Features Zhang ranked below #20 have unknown Zhang rank
and are EXCLUDED from the correlation. The Spearman rho therefore runs on
the handful of shared features inside Zhang's top 20. It is descriptive,
not inferential - no p-value claims. (A full comparison would require
re-running SHAP on the Zhang models and data in the dissertation bundle.)

Name alignment is explicit and small on purpose:
  - exact matches (gender, liver_disease, eye_opening, movement, ...)
  - one known rename: moderate_to_severe_chronic_kidney_disease
    (Zhang) -> moderate_to_severe_ckd (MIMIC)
  - GCS (Zhang total score) has NO MIMIC counterpart feature, because
    gcs_total is excluded from MIMIC modelling by design (placeholder
    artefact - see method note 5 in script 09). It is reported as a
    note row only.

READS : results/tables/shap_importance.csv               (Zhang top-20)
        results/tables/shap_importance_full_cohort.csv   (MIMIC, from 09)
        results/tables/shap_importance_icu_only.csv      (MIMIC ICU, from 09)
WRITES: results/tables/shap_cross_cohort_correlation.csv
        results/figures/shap_cross_cohort_ranks.png
        results/shap_cross_cohort_note.md

USAGE
-----
    python scripts/09c_zhang_mimic_shap_correlation.py
"""

import os
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES_TABLES = os.path.join("results", "tables")
RES_FIGS = os.path.join("results", "figures")
ZHANG_CSV = os.path.join(RES_TABLES, "shap_importance.csv")
MIMIC_FULL_CSV = os.path.join(RES_TABLES, "shap_importance_full_cohort.csv")
MIMIC_ICU_CSV = os.path.join(RES_TABLES, "shap_importance_icu_only.csv")
OUT_CSV = os.path.join(RES_TABLES, "shap_cross_cohort_correlation.csv")
OUT_FIG = os.path.join(RES_FIGS, "shap_cross_cohort_ranks.png")
OUT_NOTE = os.path.join("results", "shap_cross_cohort_note.md")

# Known renames between Zhang canonical names and MIMIC extract names.
ALIASES = {
    "moderate_to_severe_chronic_kidney_disease": "moderate_to_severe_ckd",
}

# Zhang features with no MIMIC model-feature counterpart (documented, not
# silently dropped).
NO_COUNTERPART = {
    "GCS": "MIMIC excludes gcs_total (placeholder artefact where verbal is "
           "untestable); MIMIC uses the three components instead.",
}


def spearman(a, b):
    """Rank correlation without a scipy dependency (rank-then-Pearson)."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def load_ranked(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/09_mimic_native_gcs_retrain.py "
            f"first (it produces the MIMIC SHAP tables).")
    df = pd.read_csv(path).sort_values("mean_abs_shap", ascending=False)
    df = df.reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    print(f"  {label}: {len(df)} features")
    return df


def main():
    os.makedirs(RES_TABLES, exist_ok=True)
    os.makedirs(RES_FIGS, exist_ok=True)

    print("Loading SHAP importances...")
    zhang = load_ranked(ZHANG_CSV, "Zhang (top-20 only, of 143 features)")
    mfull = load_ranked(MIMIC_FULL_CSV, "MIMIC full cohort (6m_death)")
    micu = load_ranked(MIMIC_ICU_CSV, "MIMIC ICU-only (6m_death)")

    # ---- align Zhang -> MIMIC ------------------------------------------
    rows = []
    for _, zr in zhang.iterrows():
        zf = zr["feature"]
        if zf in NO_COUNTERPART:
            rows.append({
                "zhang_feature": zf, "mimic_feature": None,
                "zhang_rank": int(zr["rank"]),
                "zhang_mean_abs_shap": float(zr["mean_abs_shap"]),
                "mimic_rank_full": None, "mimic_mean_abs_shap_full": None,
                "mimic_rank_icu": None, "mimic_mean_abs_shap_icu": None,
                "note": NO_COUNTERPART[zf],
            })
            continue
        mf = ALIASES.get(zf, zf)
        mrow = mfull[mfull["feature"] == mf]
        if mrow.empty:
            continue  # no MIMIC counterpart feature - outside the shared set
        irow = micu[micu["feature"] == mf]
        rows.append({
            "zhang_feature": zf, "mimic_feature": mf,
            "zhang_rank": int(zr["rank"]),
            "zhang_mean_abs_shap": float(zr["mean_abs_shap"]),
            "mimic_rank_full": int(mrow["rank"].iloc[0]),
            "mimic_mean_abs_shap_full": float(mrow["mean_abs_shap"].iloc[0]),
            "mimic_rank_icu": (int(irow["rank"].iloc[0]) if not irow.empty else None),
            "mimic_mean_abs_shap_icu": (float(irow["mean_abs_shap"].iloc[0])
                                        if not irow.empty else None),
            "note": "",
        })

    shared = pd.DataFrame(rows)
    comparable = shared.dropna(subset=["mimic_rank_full"]).copy()

    rho_zhang_full = spearman(comparable["zhang_rank"],
                              comparable["mimic_rank_full"])
    rho_zhang_icu = spearman(comparable["zhang_rank"],
                             comparable["mimic_rank_icu"]) \
        if comparable["mimic_rank_icu"].notna().all() else float("nan")

    # Within-MIMIC stability: full cohort vs ICU-only rankings (full lists,
    # so this correlation is NOT limited by the Zhang top-20 truncation).
    both = mfull.merge(micu, on="feature", suffixes=("_full", "_icu"))
    rho_mimic_int = spearman(both["rank_full"], 1.0 * both["rank_icu"])

    print(f"\nShared features inside Zhang top-20 : {len(comparable)}")
    print(comparable[["zhang_feature", "mimic_feature", "zhang_rank",
                      "mimic_rank_full"]].to_string(index=False))
    print(f"\n[NOT REPORTABLE AS EVIDENCE] cross-cohort Spearman: "
          f"{rho_zhang_full:.3f} (vs MIMIC full), {rho_zhang_icu:.3f} "
          f"(vs MIMIC ICU-only). n={len(comparable)} is too small - the sign "
          f"can flip with one feature in or out of the overlap. The defensible "
          f"claim is qualitative: the GCS family ranks top-tier in BOTH "
          f"cohorts independently.")
    print(f"[reportable] Spearman(mimic_full_rank, mimic_icu_rank) = "
          f"{rho_mimic_int:.3f} (n={len(both)} - within-MIMIC stability)")

    # GCS-family positions (the headline cross-cohort statement)
    gcs_zhang = zhang[zhang["feature"].isin(["GCS", "eye_opening",
                                             "verbal_response", "movement"])]
    gcs_mimic = mfull[mfull["feature"].isin(["eye_opening",
                                             "verbal_response", "movement"])]

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    ax.scatter(comparable["zhang_rank"], comparable["mimic_rank_full"],
               s=60, color="#2a9d8f")
    for _, r in comparable.iterrows():
        ax.annotate(r["mimic_feature"],
                    (r["zhang_rank"], r["mimic_rank_full"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    top = max(int(comparable["zhang_rank"].max()),
              int(comparable["mimic_rank_full"].max())) + 2
    ax.plot([0, top], [0, top], ls="--", c="grey", lw=1)
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("Zhang SHAP rank (1 = most important)")
    ax.set_ylabel("MIMIC full-cohort SHAP rank")
    ax.set_title("Zhang vs MIMIC feature ranks (qualitative only)\n"
                 f"shared features in Zhang top-20, n={len(comparable)} - "
                 "no rank statistic is reportable at this n",
                 fontsize=10)

    ax = axes[1]
    ax.scatter(both["rank_full"], both["rank_icu"], s=40, color="#e76f51")
    for _, r in both.iterrows():
        if r["rank_full"] <= 10 or r["rank_icu"] <= 10:
            ax.annotate(r["feature"], (r["rank_full"], r["rank_icu"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    top2 = max(int(both["rank_full"].max()), int(both["rank_icu"].max())) + 2
    ax.plot([0, top2], [0, top2], ls="--", c="grey", lw=1)
    ax.set_xlim(0, top2); ax.set_ylim(0, top2)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("MIMIC full-cohort SHAP rank")
    ax.set_ylabel("MIMIC ICU-only SHAP rank")
    ax.set_title(f"MIMIC internal stability\n"
                 f"Spearman rho = {rho_mimic_int:.2f} (n={len(both)})",
                 fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    plt.close()

    # ---- outputs --------------------------------------------------------
    shared.to_csv(OUT_CSV, index=False)

    def fmt_rank(df, col):
        return ", ".join(f"`{r['feature']}` #{int(r[col])}"
                         for _, r in df.iterrows())

    with open(OUT_NOTE, "w", encoding="utf-8") as f:
        f.write("# Zhang vs MIMIC SHAP comparison\n\n")
        f.write("Produced by `scripts/09c_zhang_mimic_shap_correlation.py`. "
                "This addresses the review finding that SHAP had only been "
                "run on the Zhang cohort.\n\n")
        chf = comparable.loc[comparable["mimic_feature"] ==
                             "congestive_heart_failure", "mimic_rank_full"]
        chf_txt = f"#{int(chf.iloc[0])}" if not chf.empty else "low"
        f.write("## Headline (lead with this)\n\n")
        f.write("- **Qualitative claim - the defensible cross-cohort "
                "statement:** the GCS family ranks in the top tier of BOTH "
                f"cohorts independently. Zhang ranks "
                f"{fmt_rank(gcs_zhang, 'rank')}; MIMIC full cohort ranks "
                f"{fmt_rank(gcs_mimic, 'rank')} (of {len(mfull)}), and in "
                "the MIMIC ICU-only model the GCS components are the #1, "
                "#6 and #7 features overall.\n")
        f.write("- `GCS` (Zhang total) has no MIMIC counterpart feature by "
                "design - `gcs_total` is excluded because it is a "
                "placeholder artefact where verbal is untestable (method "
                "note 5 in script 09), so the comparison uses the three "
                "components.\n")
        f.write(f"- Outside the GCS family the cohorts do NOT preserve "
                f"feature order (e.g. `congestive_heart_failure` is Zhang "
                f"#6 but MIMIC {chf_txt}); consistent with the "
                f"external-validation collapse (0.5235 / 0.5995 AUROC).\n\n")
        f.write("## Rank statistics (secondary - do NOT quote the "
                "cross-cohort rho)\n\n")
        f.write(f"- Cross-cohort Spearman on the {len(comparable)} shared "
                f"features: {rho_zhang_full:.3f} (vs MIMIC full), "
                f"{rho_zhang_icu:.3f} (vs MIMIC ICU-only). **n = "
                f"{len(comparable)} is too small for this to be evidence in "
                f"either direction - the sign can flip with one feature "
                f"entering or leaving the overlap.** A reviewer will do the "
                f"n-arithmetic in seconds; do not present these values as "
                f"agreement or disagreement. Recorded for completeness only.\n")
        f.write(f"- Within-MIMIC stability IS reportable (full 37-feature "
                f"rankings, not limited by the Zhang truncation): **Spearman "
                f"rho = {rho_mimic_int:.3f}** (full cohort vs ICU-only).\n\n")
        f.write("## Per-feature table\n\n")
        f.write("```\\n" + shared.to_string(index=False) + "\\n```\\n\\n")
        f.write("## Limitations (read before quoting this)\n\n")
        f.write("1. `results/tables/shap_importance.csv` holds only Zhang's "
                "TOP 20 of 143 features. Ranks below #20 are unknown, so the "
                "cross-cohort rho uses only the shared subset of those 20 and "
                "is **descriptive, not inferential**.\n")
        f.write("2. Mean |SHAP| values are NOT comparable across cohorts "
                "(different models, different outcome prevalence, different "
                "units). Only RANKS are compared.\n")
        f.write("3. A feature can be absent from the shared set purely "
                "because the two cohorts measured different things (Zhang "
                "has echo and blood-gas variables; MIMIC has MIMIC labs). "
                "Absence is not disagreement.\n")
        f.write("4. External validation AUROC already collapsed "
                "(0.5235 / 0.5995), so any cross-cohort narrative must stay "
                "modest: the defensible claim is about which features each "
                "cohort's model leaned on, not that they agree.\n")
        f.write("5. For a full-feature comparison, re-run SHAP on the Zhang "
                "models (`models/*.pkl`) with `dat.csv` from the dissertation "
                "bundle and replace `shap_importance.csv` with the full "
                "143-feature ranking.\n")

    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {OUT_FIG}")
    print(f"Saved: {OUT_NOTE}")


if __name__ == "__main__":
    main()
