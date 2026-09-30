"""
08_mimic_gcs_extraction.py
==========================
Extracts the full MIMIC-IV heart failure cohort (n~42,990) with GCS components
added as new features, for the Phase 1 -> Phase 3 future-work pipeline.

What this produces
------------------
mimic_hf_with_gcs.csv — ~42,990 rows x 48 columns:
  - Demographics     : hadm_id, subject_id, age, gender
  - Outcomes         : death_within_28days, death_within_6months, readmission_within_6months
  - Comorbidities    : 14 Charlson flags + CCI score
  - Labs             : creatinine, urea, sodium, potassium, chloride, bicarbonate,
                       albumin, calcium, anion_gap
  - CBC              : hemoglobin, platelet, white_blood_cell, hematocrit, mcv, rdw, red_blood_cell
  - Cardiac markers  : brain_natriuretic_peptide (NT-proBNP), troponin_t
  - *** NEW ***      : gcs_total, eye_opening, verbal_response, movement, gcs_intubated_flag
  - Anthropometrics  : height_cm, weight_kg, bmi

Requirements
------------
pip install google-cloud-bigquery db-dtypes pandas pyarrow

Authentication
--------------
Run once before this script:
    gcloud auth application-default login
(Use the nikpro0805@gmail.com account — the one with PhysioNet MIMIC-IV access)
    gcloud auth application-default set-quota-project gen-lang-client-0398113511

Usage
-----
python scripts/08_mimic_gcs_extraction.py              # full 42k extract
python scripts/08_mimic_gcs_extraction.py --dry-run    # LIMIT 5 validation only
python scripts/08_mimic_gcs_extraction.py --dry-run --limit 20

Output goes to ./mimic_hf_with_gcs.csv (same folder as the script) or data/.
"""

import argparse
import os
import sys

from google.cloud import bigquery
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ID = "gen-lang-client-0398113511"
OUTPUT_FILE = "mimic_hf_with_gcs.csv"

# ── Query ────────────────────────────────────────────────────────────────────
# Derived tables are from physionet-data.mimiciv_3_1_derived
# NOTE: hf_hadm uses ICD-10 I50* only to match the existing 42,990 cohort's
# definition. If you need ICD-9 (428*) as well, add:
#   OR (icd_code LIKE '428%' AND icd_version = 9)
QUERY_TEMPLATE = """
WITH
hf_hadm AS (
  SELECT DISTINCT hadm_id
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
  WHERE icd_code LIKE 'I50%' AND icd_version = 10
),
base AS (
  SELECT
    a.hadm_id, a.subject_id, a.admittime, a.dischtime,
    p.gender, p.dod,
    p.anchor_age + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
    a.admission_type, a.race
  FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON a.subject_id = p.subject_id
  WHERE a.hadm_id IN (SELECT hadm_id FROM hf_hadm)
),
outcomes AS (
  SELECT
    b.hadm_id,
    CASE WHEN b.dod IS NOT NULL
         AND DATE_DIFF(b.dod, DATE(b.dischtime), DAY) BETWEEN 0 AND 28
         THEN 1 ELSE 0 END AS death_within_28days,
    CASE WHEN b.dod IS NOT NULL
         AND DATE_DIFF(b.dod, DATE(b.dischtime), DAY) BETWEEN 0 AND 180
         THEN 1 ELSE 0 END AS death_within_6months,
    CASE WHEN MIN(DATE_DIFF(DATE(a2.admittime), DATE(b.dischtime), DAY))
              BETWEEN 1 AND 180
         THEN 1 ELSE 0 END AS readmission_within_6months
  FROM base b
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a2
    ON b.subject_id = a2.subject_id
    AND a2.hadm_id != b.hadm_id
    AND a2.admittime > b.dischtime
    AND DATE_DIFF(DATE(a2.admittime), DATE(b.dischtime), DAY) BETWEEN 1 AND 180
  GROUP BY b.hadm_id, b.dod, b.dischtime
),
comorbid AS (
  SELECT hadm_id,
    myocardial_infarct                                       AS myocardial_infarction,
    congestive_heart_failure,
    peripheral_vascular_disease,
    cerebrovascular_disease,
    dementia,
    chronic_pulmonary_disease                                AS copd,
    peptic_ulcer_disease,
    GREATEST(diabetes_without_cc, diabetes_with_cc)         AS diabetes,
    renal_disease                                            AS moderate_to_severe_ckd,
    GREATEST(mild_liver_disease, severe_liver_disease)      AS liver_disease,
    GREATEST(malignant_cancer, metastatic_solid_tumor)      AS solid_tumor,
    aids,
    paraplegia                                               AS hemiplegia,
    charlson_comorbidity_index                               AS cci_score
  FROM `physionet-data.mimiciv_3_1_derived.charlson`
),
labs AS (
  SELECT hadm_id, creatinine, bun AS urea, sodium, potassium, chloride,
         bicarbonate, albumin, calcium, aniongap AS anion_gap
  FROM (
    SELECT hadm_id, creatinine, bun, sodium, potassium, chloride,
           bicarbonate, albumin, calcium, aniongap,
           ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime DESC) AS rn
    FROM `physionet-data.mimiciv_3_1_derived.chemistry`
    WHERE hadm_id IS NOT NULL
  ) WHERE rn = 1
),
cbc AS (
  SELECT hadm_id, hemoglobin, platelet, wbc, hematocrit, mcv, rdw, rbc
  FROM (
    SELECT hadm_id, hemoglobin, platelet, wbc, hematocrit, mcv, rdw, rbc,
           ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime DESC) AS rn
    FROM `physionet-data.mimiciv_3_1_derived.complete_blood_count`
    WHERE hadm_id IS NOT NULL
  ) WHERE rn = 1
),
cardiac AS (
  SELECT hadm_id,
         ntprobnp   AS brain_natriuretic_peptide,
         troponin_t
  FROM (
    SELECT hadm_id, ntprobnp, troponin_t,
           ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY charttime DESC) AS rn
    FROM `physionet-data.mimiciv_3_1_derived.cardiac_marker`
    WHERE hadm_id IS NOT NULL
  ) WHERE rn = 1
),
gcs AS (
  -- Last GCS recording across any ICU stay linked to this admission
  SELECT i.hadm_id,
         g.gcs       AS gcs_total,
         g.gcs_eyes  AS eye_opening,
         g.gcs_verbal AS verbal_response,
         g.gcs_motor  AS movement,
         g.gcs_unable AS gcs_intubated_flag
  FROM (
    SELECT stay_id, gcs, gcs_eyes, gcs_verbal, gcs_motor, gcs_unable,
           ROW_NUMBER() OVER (PARTITION BY stay_id ORDER BY charttime DESC) AS rn
    FROM `physionet-data.mimiciv_3_1_derived.gcs`
  ) g
  JOIN `physionet-data.mimiciv_3_1_icu.icustays` i ON g.stay_id = i.stay_id
  WHERE g.rn = 1
  QUALIFY ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.outtime DESC) = 1
),
hw AS (
  SELECT i.hadm_id,
         AVG(h.height) AS height_cm,
         AVG(w.weight) AS weight_kg
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_height` h ON i.stay_id = h.stay_id
  LEFT JOIN `physionet-data.mimiciv_3_1_derived.first_day_weight` w ON i.stay_id = w.stay_id
  WHERE i.hadm_id IN (SELECT hadm_id FROM hf_hadm)
  GROUP BY i.hadm_id
)

SELECT
  b.hadm_id,
  b.subject_id,
  b.age,
  CASE WHEN b.gender = 'M' THEN 1 ELSE 0 END AS gender,
  -- Outcomes
  o.death_within_28days,
  o.death_within_6months,
  o.readmission_within_6months,
  -- Comorbidities
  c.myocardial_infarction,
  c.congestive_heart_failure,
  c.peripheral_vascular_disease,
  c.cerebrovascular_disease,
  c.dementia,
  c.copd,
  c.peptic_ulcer_disease,
  c.diabetes,
  c.moderate_to_severe_ckd,
  c.liver_disease,
  c.solid_tumor,
  c.aids,
  c.hemiplegia,
  c.cci_score,
  -- Labs
  l.creatinine,
  l.urea,
  l.sodium,
  l.potassium,
  l.chloride,
  l.bicarbonate,
  l.albumin,
  l.calcium,
  l.anion_gap,
  -- CBC
  cb.hemoglobin,
  cb.platelet,
  cb.wbc        AS white_blood_cell,
  cb.hematocrit,
  cb.mcv,
  cb.rdw,
  cb.rbc        AS red_blood_cell,
  -- Cardiac markers
  ca.brain_natriuretic_peptide,
  ca.troponin_t,
  -- *** GCS — NEW FEATURES ***
  g.gcs_total,
  g.eye_opening,
  g.verbal_response,
  g.movement,
  g.gcs_intubated_flag,
  -- Anthropometrics
  hw.height_cm,
  hw.weight_kg,
  CASE
    WHEN hw.height_cm > 0 AND hw.weight_kg > 0
    THEN ROUND(hw.weight_kg / POW(hw.height_cm / 100.0, 2), 2)
    ELSE NULL
  END AS bmi
FROM base b
LEFT JOIN outcomes o  ON b.hadm_id = o.hadm_id
LEFT JOIN comorbid c  ON b.hadm_id = c.hadm_id
LEFT JOIN labs     l  ON b.hadm_id = l.hadm_id
LEFT JOIN cbc      cb ON b.hadm_id = cb.hadm_id
LEFT JOIN cardiac  ca ON b.hadm_id = ca.hadm_id
LEFT JOIN gcs      g  ON b.hadm_id = g.hadm_id
LEFT JOIN hw          ON b.hadm_id = hw.hadm_id
{limit_clause}
"""


def build_query(limit=None):
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    return QUERY_TEMPLATE.format(limit_clause=limit_clause)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Extract MIMIC-IV HF cohort with GCS")
    parser.add_argument("--dry-run", action="store_true", help="Only fetch LIMIT rows for validation")
    parser.add_argument("--limit", type=int, default=5, help="Row limit for dry-run (default 5)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    limit = args.limit if args.dry_run else None
    query = build_query(limit=limit)

    # Resolve output path
    if args.output:
        out_path = args.output
    else:
        # default: same folder as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, "..", "data", OUTPUT_FILE) if not args.dry_run else os.path.join(script_dir, "mimic_hf_with_gcs_dryrun.csv")
        out_path = os.path.normpath(out_path)

    print(f"Connecting to BigQuery project: {PROJECT_ID}")
    print(f"Mode: {'DRY-RUN (LIMIT %d)' % limit if limit else 'FULL EXTRACT (~42,990 rows)'}")
    try:
        client = bigquery.Client(project=PROJECT_ID)
    except Exception as e:
        print(f"Failed to create BigQuery client: {e}", file=sys.stderr)
        print("Hint: run `gcloud auth application-default login` and `gcloud auth application-default set-quota-project gen-lang-client-0398113511`", file=sys.stderr)
        sys.exit(1)

    print("Running extraction query (dry-run ~5s, full ~30-60s)...")
    try:
        df = client.query(query).to_dataframe()
    except Exception as e:
        print(f"Query failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Rows extracted   : {len(df):,}")
    print(f"✓ Columns          : {len(df.columns)}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    if "death_within_28days" in df.columns:
        print("\nOutcome rates (in extracted sample):")
        for col in ["death_within_28days", "death_within_6months", "readmission_within_6months"]:
            if col in df.columns:
                rate = df[col].mean() * 100 if len(df) else 0
                n = df[col].sum() if len(df) else 0
                print(f"  {col}: {rate:.2f}%  (n={int(n):,})")

    if "gcs_total" in df.columns:
        print("\nGCS coverage (% admissions with GCS recorded):")
        gcs_coverage = df["gcs_total"].notna().mean() * 100 if len(df) else 0
        print(f"  gcs_total non-null : {gcs_coverage:.1f}%")
        if df["gcs_total"].notna().any():
            gcs15_rate = (df["gcs_total"] == 15).sum() / df["gcs_total"].notna().sum() * 100
            print(f"  GCS = 15 (max)     : {gcs15_rate:.1f}%  (Zhang cohort: 97.2%)")
            print(f"  GCS mean           : {df['gcs_total'].mean():.2f}")
            print(f"  GCS median         : {df['gcs_total'].median():.0f}")
            print(f"  GCS distribution   :\n{df['gcs_total'].value_counts().sort_index().head(20)}")
            print(f"\n  eye_opening (1-4) mean: {df['eye_opening'].mean():.2f}  non-null {df['eye_opening'].notna().mean()*100:.1f}%")
            print(f"  verbal_response (1-5) mean: {df['verbal_response'].mean():.2f}  non-null {df['verbal_response'].notna().mean()*100:.1f}%")
            print(f"  movement (1-6) mean: {df['movement'].mean():.2f}  non-null {df['movement'].notna().mean()*100:.1f}%")

    print("\nMissingness (% null per column):")
    miss = df.isnull().mean().sort_values(ascending=False)
    for col, pct in miss[miss > 0].items():
        print(f"  {col:<35} {pct*100:.1f}%")
    if (miss == 0).any():
        print(f"  ... {(miss==0).sum()} columns with 0% missing")

    if not args.dry_run:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n✓ Saved to: {out_path}")
        print(f"  File size: {os.path.getsize(out_path)/1024/1024:.1f} MB")
    else:
        print(f"\n(dry-run) Preview:")
        print(df.head().to_string())
        df.to_csv(out_path, index=False)
        print(f"\n(dry-run) Saved preview to: {out_path}")

    print("\nDone. Next: script 09 retrain with 27 features including eye_opening, verbal_response, movement.")


if __name__ == "__main__":
    main()
