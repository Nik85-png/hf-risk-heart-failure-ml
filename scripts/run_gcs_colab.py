"""
run_gcs_colab.py
================
Colab / cloud entry point for the MIMIC GCS retrain (script 09).

This exists so the heavy job runs on Colab's CPU rather than on your
laptop. Do NOT run script 09 directly on your PC unless you pass --quick.

The full run is roughly 15-40 minutes of CPU: 5-fold GroupKFold CV across
6 model fits, 2 more models for the ICU-only subgroup, 2 TreeSHAP passes,
and a 1,000-iteration paired bootstrap per comparison.

HOW TO USE (paste these cells into a Colab notebook)
----------------------------------------------------
Cell 1 - get the code
    !git clone <your-repo-url> hf-risk-heart-failure-ml
    %cd hf-risk-heart-failure-ml

Cell 2 - install
    !pip -q install google-cloud-bigquery db-dtypes pandas pyarrow \
         xgboost shap scikit-learn matplotlib tabulate

Cell 3 - get the data
    (a) Already have mimic_hf_with_gcs.csv on Drive?
        from google.colab import drive
        drive.mount('/content/drive')
        !mkdir -p data
        !cp "/content/drive/MyDrive/hf-risk/mimic_hf_with_gcs.csv" data/

    (b) Otherwise pull it fresh from BigQuery (~40 s) using script 08:
        from google.colab import auth
        auth.authenticate_user()
        import subprocess
        subprocess.run(["python", "scripts/08_mimic_gcs_extraction.py"], check=True)

Cell 4 - run the analysis
        import sys, subprocess
        args = sys.argv[1:] or []
        subprocess.run(
            [sys.executable, "scripts/09_mimic_native_gcs_retrain.py", *args],
            check=True)

        # fast smoke test first (a few minutes):
        #   subprocess.run([sys.executable,
        #     "scripts/09_mimic_native_gcs_retrain.py", "--quick", "--no-cv",
        #     "--bootstrap", "50", "--n-estimators", "100"], check=True)

Cell 5 - save the outputs to Drive
    from google.colab import drive
    drive.mount('/content/drive')
    !mkdir -p "/content/drive/MyDrive/hf-risk/outputs"
    !cp -r outputs/mimic_gcs  "/content/drive/MyDrive/hf-risk/outputs/"
    !cp -r results          "/content/drive/MyDrive/hf-risk/results/"

WHAT TO READ WHEN IT FINISHES
-----------------------------
    results/mimic_gcs_run_log.md            <- start here; plain English
    results/tables/mimic_gcs_retrain_results.csv
    results/tables/shap_importance_full_cohort.csv
    results/tables/shap_importance_icu_only.csv

Review those two files before anything goes into the preprint or the
website. The run log states which baseline was used, whether the GCS
delta's confidence interval crosses zero, and what the had_icu_stay
control showed.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DEFAULT_ARGS = []  # full run; override from the notebook


def main():
    extra = sys.argv[1:] or DEFAULT_ARGS
    target = os.path.join("scripts", "09_mimic_native_gcs_retrain.py")
    script = os.path.join(REPO, target)

    if not os.path.exists(script):
        print(f"Cannot find {script}")
        sys.exit(1)

    data_csv = os.path.join(REPO, "data", "mimic_hf_with_gcs.csv")
    if not os.path.exists(data_csv):
        print("data/mimic_hf_with_gcs.csv is missing.")
        print("Either upload it, or run scripts/08_mimic_gcs_extraction.py first.")
        sys.exit(1)

    size_mb = os.path.getsize(data_csv) / 1e6
    print(f"Found {data_csv} ({size_mb:.1f} MB)")
    print(f"Running: 09_mimic_native_gcs_retrain.py {' '.join(extra)}")
    print("This takes roughly 15-40 minutes on Colab CPU. Be patient.\n")

    rc = subprocess.call([sys.executable, script, *extra], cwd=REPO)
    if rc != 0:
        print(f"\nFailed with exit code {rc}")
        sys.exit(rc)

    print("\nDone. Read results/mimic_gcs_run_log.md first.")


if __name__ == "__main__":
    main()
