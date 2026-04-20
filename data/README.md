# Data

Raw datasets are intentionally not committed to this public repository.

Expected local files when reproducing the pipeline:

```text
data/dat.csv
data/mimic_hf_cohort.csv
```

- `dat.csv`: Zhang / PhysioNet heart failure cohort used for model development.
- `mimic_hf_cohort.csv`: formatted MIMIC-IV heart failure cohort used for external validation and MIMIC-only benchmarking.

The scripts assume these files exist locally under `data/` when run from the repository root.
