"""
09b_readmission_sanity_check.py
==============================
Manual sanity check of `readmission_within_6months` (43.92% in
mimic_hf_with_gcs.csv) for a handful of hadm_ids, straight from BigQuery.

Why: the two mortality outcomes (28d 10.27%, 6m 21.95%) reproduce the
dissertation's original external-validation numbers exactly, which validates
the cohort SQL. The readmission outcome was never externally validated, so it
needs its own check before it is reported anywhere.

The specific risk being tested: does the join double-count, or pick up
same-day transfers / in-hospital readmissions as "readmissions"?

Writes: outputs/mimic_gcs/readmission_sanity_check.csv
Read-only: makes no writes to BigQuery, prints a comparison table.
"""

import os
import sys

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "gen-lang-client-0398113511"
COHORT_CSV = os.path.join("data", "mimic_hf_with_gcs.csv")
OUT_DIR = os.path.join("outputs", "mimic_gcs")

# Sample: a mix of readmission=1 and 0 so both branches get exercised.
N_SAMPLE = 60

QUERY = """
SELECT
  b.hadm_id,
  b.readmission_flag_bq            AS bq_flag,
  b.days_to_next_bq                AS bq_days,
  b.next_hadm_id_bq                AS bq_next_hadm,
  b.n_readmits_180d_bq             AS bq_n_readmits_180d,
  b.next_admission_type_bq         AS bq_next_admission_type,
  b.next_within_1day_bq            AS bq_next_within_1day
FROM (
  SELECT
    b.hadm_id,
    IF(MIN(DATE_DIFF(DATE(a2.admittime), DATE(b.dischtime), DAY)) BETWEEN 1 AND 180, 1, 0)
      AS readmission_flag_bq,
    MIN(DATE_DIFF(DATE(a2.admittime), DATE(b.dischtime), DAY)) AS days_to_next_bq,
    MIN(IF(a2.admittime > b.dischtime, a2.hadm_id, NULL)) AS next_hadm_id_bq,
    COUNTIF(a2.admittime > b.dischtime
            AND DATE_DIFF(DATE(a2.admittime), DATE(b.dischtime), DAY) BETWEEN 1 AND 180)
      AS n_readmits_180d_bq,
    MIN(IF(a2.admittime > b.dischtime, a2.admission_type, NULL)) AS next_admission_type_bq,
    COUNTIF(DATE_DIFF(DATE(a2.admittime), DATE(b.dischtime), DAY) BETWEEN 0 AND 1)
      AS next_within_1day_bq
  FROM `physionet-data.mimiciv_3_1_hosp.admissions` b
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a2
    ON b.subject_id = a2.subject_id
    AND a2.hadm_id != b.hadm_id
    AND a2.admittime > b.dischtime
    AND DATE_DIFF(DATE(a2.admittime), DATE(b.dischtime), DAY) BETWEEN 0 AND 180
  WHERE b.hadm_id IN (
    SELECT DISTINCT hadm_id
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE icd_code LIKE 'I50%' AND icd_version = 10
  )
    AND b.hadm_id IN UNNEST(@sample_ids)
  GROUP BY b.hadm_id
) b
"""


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    if not os.path.exists(COHORT_CSV):
        print(f"Missing {COHORT_CSV}. Run 08_mimic_gcs_extraction.py first.")
        sys.exit(1)

    df = pd.read_csv(COHORT_CSV, usecols=["hadm_id", "subject_id", "readmission_within_6months"])
    pos = df[df.readmission_within_6months == 1].head(N_SAMPLE // 2)
    neg = df[df.readmission_within_6months == 0].head(N_SAMPLE // 2)
    sample = pd.concat([pos, neg]).reset_index(drop=True)
    sample_ids = [int(x) for x in sample.hadm_id]

    print(f"Sampling {len(sample_ids)} hadm_ids "
          f"({len(pos)} readmission=1, {len(neg)} readmission=0) for BigQuery verification...")

    client = bigquery.Client(project=PROJECT_ID)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("sample_ids", "INT64", sample_ids)
        ]
    )
    bq = client.query(QUERY, job_config=job_config).to_dataframe()

    m = sample.merge(bq, on="hadm_id", how="left")
    m["flag_agrees"] = m["readmission_within_6months"] == m["bq_flag"]
    m["csv_flag"] = m["readmission_within_6months"].astype(int)
    m["bq_flag"] = m["bq_flag"].astype("Int64")

    out_path = os.path.join(OUT_DIR, "readmission_sanity_check.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    m.to_csv(out_path, index=False)

    print(f"\nRows compared           : {len(m)}")
    print(f"CSV vs BigQuery agree   : {m.flag_agrees.sum()}/{len(m)}")
    if not m.flag_agrees.all():
        print("Disagreements:")
        print(m.loc[~m.flag_agrees, ["hadm_id", "csv_flag", "bq_flag", "bq_days"]].to_string(index=False))

    print(f"\nAdmissions with >=2 readmits in 180d (double-count check): "
          f"{(m.bq_n_readmits_180d > 1).sum()}")
    print(f"Next admission within 1 day (transfer artefact check): "
          f"{m.bq_next_within_1day.gt(0).sum()}")

    pos_bq = m.loc[m.csv_flag == 1, "bq_days"]
    print(f"\nDays-to-next-admission for CSV-flagged readmissions:")
    print(f"  min={pos_bq.min()}  median={pos_bq.median()}  max={pos_bq.max()}")
    if (pos_bq < 1).any():
        print("  WARNING: some flagged readmissions are <1 day after discharge.")

    print(f"\nNext admission_type for flagged readmissions:")
    print(m.loc[m.csv_flag == 1, "bq_next_admission_type"].value_counts().to_string())

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
