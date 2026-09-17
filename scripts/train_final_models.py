from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import rdkit
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from cardiac_channels.baselines import XGB_COMMON, feature_matrix
from cardiac_channels.features import (
    DESCRIPTOR_NAMES,
    FEATURE_DIM,
    MACCS_BITS,
    MORGAN_BITS,
    MORGAN_RADIUS,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_root = root / "artifacts" / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    for channel in ("herg", "cav1.2", "nav1.5"):
        feature_path = root / "data" / "features" / f"{channel}_features.npz"
        data_path = root / "data" / "dev" / f"{channel}_dev.csv"
        data = np.load(feature_path)
        x = feature_matrix(data)
        yc = data["y_class"]
        yr = data["y_reg"]
        negatives = int((yc == 0).sum())
        positives = int((yc == 1).sum())
        train, validation = train_test_split(
            np.arange(len(yc)), test_size=0.2, random_state=42, stratify=yc
        )
        validation_classifier = XGBClassifier(
            **XGB_COMMON,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / max(positives, 1),
        ).fit(x[train], yc[train])
        validation_regressor = XGBRegressor(
            **XGB_COMMON,
            objective="reg:squarederror",
            eval_metric="rmse",
        ).fit(x[train], yr[train])
        class_probability = validation_classifier.predict_proba(x[validation])[:, 1]
        regression_prediction = validation_regressor.predict(x[validation])
        holdout_metrics = {
            "split": "random stratified 80/20, seed=42; smoke-test baseline only",
            "classification_roc_auc": float(roc_auc_score(yc[validation], class_probability)),
            "regression_r2": float(r2_score(yr[validation], regression_prediction)),
            "regression_rmse": float(mean_squared_error(yr[validation], regression_prediction) ** 0.5),
            "regression_mae": float(mean_absolute_error(yr[validation], regression_prediction)),
        }

        classifier = XGBClassifier(
            **XGB_COMMON,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / max(positives, 1),
        ).fit(x, yc)
        regressor = XGBRegressor(
            **XGB_COMMON,
            objective="reg:squarederror",
            eval_metric="rmse",
        ).fit(x, yr)

        destination = model_root / channel
        destination.mkdir(parents=True, exist_ok=True)
        classifier.save_model(destination / "classification.ubj")
        regressor.save_model(destination / "regression.ubj")
        metadata = {
            "channel": channel,
            "model": "xgboost",
            "task_models": {
                "classification": "classification.ubj",
                "regression": "regression.ubj",
            },
            "training_rows": int(len(yc)),
            "class_counts": {"non_blocker": negatives, "blocker": positives},
            "label_mapping": {"Non-blocker": 0, "Blocker": 1},
            "blocker_pic50_cutoff": 5.0,
            "features": {
                "morgan_radius": MORGAN_RADIUS,
                "morgan_bits": MORGAN_BITS,
                "maccs_bits": MACCS_BITS,
                "rdkit_descriptors": list(DESCRIPTOR_NAMES),
                "total_dimension": FEATURE_DIM,
            },
            "training_data_sha256": sha256(data_path),
            "versions": {
                "python": platform.python_version(),
                "rdkit": rdkit.__version__,
                "scikit_learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
            "hyperparameters": XGB_COMMON,
            "holdout_metrics": holdout_metrics,
        }
        (destination / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"saved {channel}: {len(yc)} rows -> {destination}; {holdout_metrics}", flush=True)


if __name__ == "__main__":
    main()
