"""
09a_icu_stay_flags.py
=====================
One small BigQuery pull: the TRUE "was this hospital admission ever in the
ICU" indicator, used as the control variable in 09_mimic_native_gcs_retrain.py.

Why this script exists
----------------------
The first draft of script 09 derived `had_icu_stay` as `gcs_total.notna()`.
That is degenerate: it is exactly "a GCS score was charted", which is already
implied by "adding GCS", so the control model could not separate "ICU
admission" from "GCS was measured". The LLM-council review flagged this. The
honest control is the real ICU admission flag from MIMIC-IV's `icustays`
table, which is independent of any charting. This script pulls it ONCE, so
script 09 can keep running offline from local CSVs.

READS : physionet-data.mimiciv_3_1_icu.icustays   (BigQuery, ADC auth)
        data/mimic_hf_with_gcs.csv                (cohort alignment + checks)
WRITES: data/mimic_icu_stay_flags.csv             (git-ignored)
        columns: hadm_id, had_icu_stay (0/1), n_icu_stays

AUTH: same as 08_mimic_gcs_extraction.py - application-default credentials
for project gen-lang-client-0398113511 (the account with PhysioNet access).

USAGE
-----
    python scripts/09a_icu_stay_flags.py
"""

import os
import sys

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "gen-lang-client-0398113511"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COHORT_CSV = os.path.join(_SCRIPT_DIR, "..", "data", "mimic_hf_with_gcs.csv")
OUT_CSV = os.path.join(_SCRIPT_DIR, "..", "data", "mimic_icu_stay_flags.csv")

QUERY = """
SELECT
  hadm_id,
  COUNT(*) AS n_icu_stays
FROM `physionet-data.mimiciv_3_1_icu.icustays`
WHERE hadm_id IS NOT NULL
GROUP BY hadm_id
"""


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    cohort = pd.read_csv(COHORT_CSV)
    print(f"Cohort: {len(cohort):,} admissions from "
          f"{cohort['subject_id'].nunique():,} patients")

    print(f"Connecting to BigQuery project: {PROJECT_ID}")
    try:
        client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"Failed to create BigQuery client: {e}", file=sys.stderr)
        print("Hint: run `gcloud auth application-default login` "
              "(account with PhysioNet MIMIC-IV access)", file=sys.stderr)
        sys.exit(1)

    print("Querying icustays (small table, seconds)...")
    try:
        icu = client.query(QUERY).to_dataframe()
    except Exception as e:
        print(f"Query failed: {e}", file=sys.stderr)
        sys.exit(1)
    icu["hadm_id"] = icu["hadm_id"].astype(int)
    icu["n_icu_stays"] = icu["n_icu_stays"].astype(int)
    icu["had_icu_stay"] = 1
    print(f"  icustays: {len(icu):,} hadm_ids with at least one ICU stay")

    out = cohort[["hadm_id"]].drop_duplicates().merge(
        icu[["hadm_id", "had_icu_stay", "n_icu_stays"]], on="hadm_id", how="left")
    out["had_icu_stay"] = out["had_icu_stay"].fillna(0).astype(int)
    out["n_icu_stays"] = out["n_icu_stays"].fillna(0).astype(int)
    assert len(out) == cohort["hadm_id"].nunique(), "cohort alignment broke"

    # Sanity: every GCS-observed admission must have an ICU stay, because the
    # GCS extraction (08) joins through icustays. A violation means the two
    # extracts disagree and neither should be trusted silently.
    gcs_obs = set(cohort.loc[cohort["gcs_total"].notna(), "hadm_id"])
    icu_set = set(out.loc[out["had_icu_stay"] == 1, "hadm_id"])
    violation = gcs_obs - icu_set
    print(f"\nICU admissions in cohort  : {int(out['had_icu_stay'].sum()):,} "
          f"({100 * out['had_icu_stay'].mean():.1f}%)")
    print(f"GCS charted in cohort     : {len(gcs_obs):,}")
    print(f"GCS-without-ICU violations: {len(violation)}")
    assert not violation, \
        f"GCS observed without ICU stay for {len(violation)} admissions"

    out = out.sort_values("hadm_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out):,} rows -> {os.path.normpath(OUT_CSV)}")


if __name__ == "__main__":
    main()
