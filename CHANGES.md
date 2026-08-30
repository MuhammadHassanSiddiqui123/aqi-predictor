# Changes vs. the original reference project

This project started from the `AirQualityIndexPredictor-main` reference
(Open-Meteo + Hopsworks + GitHub Actions + Streamlit, 10Pearls Shine
Internship). The core design is sound and was kept as-is; the changes
below were found by actually *running* every script end-to-end against
the real 3-year Karachi dataset, not just reading the code.

## Bugs fixed

1. **`data_pipeline/pipeline_manager.py` — silent data loss on every run.**
   `run_pipeline()` overwrote `karachi_daily_features.csv` with *only*
   the newly-fetched window (`df.to_csv(FILE_PATH, index=False)`).
   Since the hourly GitHub Actions workflow calls this with `--days 1`,
   any run against the local CSV would destroy the 3 years of history
   down to a single day. Hopsworks itself was safe (feature group
   `insert()` upserts by primary key), but the local CSV — which this
   project now also relies on as a dev/offline fallback — was not.
   **Fix:** merge the newly-fetched window with existing local history,
   de-duplicate by date (new data wins on overlap), and recompute all
   derived features (lags, rolling stats, targets) across the *full*
   merged series so values at the old/new boundary stay correct.
   Verified with a mocked pipeline run: 1097 existing + 2 fetched → 1098
   after de-dup, full date range preserved.

2. **`ml_pipeline/train_model.py` / `analysis/run_shap.py` — hard
   Hopsworks dependency at import time.** Both imported
   `data_pipeline.hopsworks_connector` (which imports the `hopsworks`
   package) at module load, so neither script could even be imported,
   let alone run, without a Hopsworks account and the `hopsworks`
   package installed. This blocked any local development or testing
   before signing up.
   **Fix:** made the Hopsworks import lazy, and added a
   `fetch_training_data()` fallback that reads the local CSV (applying
   the same column-normalization Hopsworks does on ingest, so feature
   names match regardless of source) when `HOPSWORKS_API_KEY` isn't
   set or the Hopsworks call fails. Verified: full 5-model training run
   on local data reproduces the README's benchmark numbers exactly
   (Ridge 1-day R²=0.866, HGBR 0.819, RF 0.819, MLP 0.750, DT 0.705).

3. **`backend/engine.py` — Streamlit secrets check doesn't actually
   guard against the case it's meant for.** `hasattr(st, "secrets")` is
   always `True` (`st.secrets` is a lazy object); the real
   `StreamlitSecretNotFoundError` is only raised once you *access* it,
   which happened one line later inside the "guarded" block — so the
   app crashed on import in any environment without a `secrets.toml`
   (i.e. local dev, or the sandbox this was verified in).
   **Fix:** wrapped the actual access in try/except instead of relying
   on `hasattr`.

4. **`frontend/app.py` — defaulted to a non-functional mode locally.**
   `_detect_integrated_mode()` returned `False` unless
   `STREAMLIT_CLOUD=1` was set or a `secrets.toml` with
   `HOPSWORKS_API_KEY` existed. In any plain local run, this meant the
   app defaulted to "API Mode," which calls out to a FastAPI backend on
   `localhost:8000` — nothing is listening there unless you separately
   started `backend/main.py`, so the dashboard would silently show
   empty/`None` data with no error.
   **Fix:** default to Integrated Mode always (imports `AQIEngine`
   directly, which itself now falls back to local CSV/models). API Mode
   is now opt-in via `USE_API_MODE=1`, for anyone who wants the
   frontend/backend split. Verified with Streamlit's `AppTest` harness
   (executes the real script, not just an HTTP ping): zero exceptions,
   correct current-AQI/forecast/history rendering.

5. **`analysis/run_eda.py` — minor seaborn deprecation warnings** from
   passing `palette=` without `hue=`. Fixed for forward-compatibility
   with seaborn ≥0.14.

## What was *not* changed

- Feature engineering logic (`data_pipeline/engineer_features.py`) —
  ran as-is, no issues found.
- Model architectures/hyperparameters in `train_model.py` — kept
  exactly as tuned in the original, since they reproduce the documented
  benchmark numbers on real data.
- Overall system architecture (Open-Meteo → Hopsworks Feature Store →
  daily training → Hopsworks Model Registry → Streamlit) — unchanged;
  it's the right design for this project's "100% serverless" brief.
- `.github/workflows/*.yml` — unchanged; they already handle the
  Hopsworks-unavailable case reasonably (the training/registry-upload
  steps fail gracefully rather than crashing the whole job) and now
  additionally benefit from fixes #1 and #2 above.
