# Models

Trained `.pkl` model artifacts are intentionally not committed to this public repository.

Running the pipeline locally will create model files here, including:

```text
models/best_model_28d_death.pkl
models/best_model_3m_death.pkl
models/best_model_6m_death.pkl
models/best_model_6m_readmission.pkl
models/mice_imputer.pkl
models/cox_model.pkl
models/rsf_model.pkl
```

Model binaries can be large and may encode training-time artifacts, so the public repository focuses on source code, aggregate results, and figures.
