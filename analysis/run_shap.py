"""
run_shap.py  –  SHAP feature‑importance analysis for every trained model.

Loads each model artifact from the Hopsworks registry (via ml_pipeline),
runs SHAP on the test split, and saves:
  • a beeswarm summary plot per model × horizon
  • a consolidated text report with top‑10 features

Usage:
    python -m analysis.run_shap
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_pipeline.train_model import fetch_training_data  # Hopsworks-or-local-CSV fallback

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "shap")
LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_MAP = {
    "hgbr": "karachi_aqi_hgbr",
    "rf":   "karachi_aqi_rf",
    "ridge":"karachi_aqi_ridge",
    "dt":   "karachi_aqi_dt",
    "mlp":  "karachi_aqi_mlp",
}


def _load_model_bundle(short, registry_name):
    """Load from Hopsworks Model Registry if configured, else the local
    .pkl saved by ml_pipeline/train_model.py -- same fallback pattern
    used in train_model.py and backend/engine.py."""
    if os.getenv("HOPSWORKS_API_KEY"):
        try:
            from ml_pipeline.model_utils import load_latest_model_path
            p = load_latest_model_path(registry_name)
            if p:
                return joblib.load(p)
        except Exception as e:
            print(f"  Hopsworks load failed for {short} ({e}); trying local file.")
    local_path = os.path.join(LOCAL_MODELS_DIR, f"aqi_{short}_models.pkl")
    if os.path.exists(local_path):
        return joblib.load(local_path)
    return None
HORIZONS = ["1d", "2d", "3d"]
TARGET_COLS = ["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]
DROP_COLS   = ["event_timestamp", "karachi_id", "date", "created"]

# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_explainer(model, X_bg):
    """Pick the right SHAP explainer for the model type."""
    cls = type(model).__name__
    if cls in ("HistGradientBoostingRegressor", "RandomForestRegressor", "DecisionTreeRegressor"):
        return shap.TreeExplainer(model)
    else:
        # KernelExplainer for Ridge / MLP (use 50‑row background for speed)
        bg = shap.sample(X_bg, min(50, len(X_bg)))
        return shap.KernelExplainer(model.predict, bg)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1.  Fetch data (Hopsworks if configured, else local CSV) ───────────
    df = fetch_training_data()
    df = df.sort_values("event_timestamp").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS and c not in TARGET_COLS]
    print(f"✅ Data ready: {len(df)} rows, {len(feature_cols)} features")

    # ── 2.  Load models (registry or local) ─────────────────────────────────
    artifacts = {}
    for short, reg_name in MODEL_MAP.items():
        try:
            bundle = _load_model_bundle(short, reg_name)
            if bundle:
                artifacts[short] = bundle
                print(f"  Loaded {short}")
            else:
                print(f"  ⚠ {short}: no model found (train it first)")
        except Exception as e:
            print(f"  ⚠ {short}: {e}")

    if not artifacts:
        print("No models loaded – aborting.")
        sys.exit(1)

    # ── 3.  SHAP analysis per model × horizon ───────────────────────────────
    report_lines = ["SHAP FEATURE IMPORTANCE SUMMARY", "=" * 50, ""]

    for short, bundle in artifacts.items():
        model_features = bundle["features"]
        performance    = bundle.get("performance", {})

        for horizon in HORIZONS:
            tag = f"{short}_{horizon}"
            print(f"\n── SHAP: {tag} ──")

            target = f"target_aqi_{horizon}"
            df_h = df.dropna(subset=[target]).copy()
            X = df_h[model_features]
            y = df_h[target]

            X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

            h_meta = bundle["models"][horizon]
            model  = h_meta["model"]
            scaler = h_meta["scaler"]

            X_test_scaled  = scaler.transform(X_test)
            X_train_scaled = scaler.transform(X_train)

            X_test_df  = pd.DataFrame(X_test_scaled, columns=model_features)
            X_train_df = pd.DataFrame(X_train_scaled, columns=model_features)

            # SHAP
            explainer   = _get_explainer(model, X_train_df)
            shap_values = explainer(X_test_df)

            # Beeswarm plot
            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
            plt.title(f"SHAP – {short.upper()} ({horizon})")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"shap_{tag}.png"), dpi=150)
            plt.close()
            print(f"  → shap_{tag}.png")

            # Top‑10 features by mean |SHAP|
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[::-1][:10]

            h_perf = performance.get(horizon, {})
            header = f"{short.upper()} – {horizon}  (R²={h_perf.get('R2','?'):.3f}, MAE={h_perf.get('MAE','?'):.2f}, RMSE={h_perf.get('RMSE','?'):.2f})"
            report_lines.append(header)
            report_lines.append("-" * len(header))
            for rank, idx in enumerate(top_idx, 1):
                report_lines.append(f"  {rank:>2}. {model_features[idx]:<30s}  mean|SHAP| = {mean_abs[idx]:.4f}")
            report_lines.append("")

    # ── 4.  Write consolidated report ───────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "feature_importance_summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n✅ SHAP analysis complete!  Files → {OUTPUT_DIR}")
    print(f"📄 Consolidated report → {report_path}")


if __name__ == "__main__":
    main()
