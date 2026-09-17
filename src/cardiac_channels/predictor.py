from __future__ import annotations

import json
import math
from pathlib import Path

from xgboost import XGBClassifier, XGBRegressor

from .features import FEATURE_DIM, featurize_smiles


CHANNELS = ("herg", "cav1.2", "nav1.5")
CIPA_REFERENCE_DRUGS = {
    "dofetilide": {
        "smiles": "CS(=O)(=O)NCCCOc1ccc(CCNCCc2ccc(OS(=O)(=O)C)cc2)cc1",
        "expected_risk": "high",
    },
    "verapamil": {
        "smiles": "COc1ccc(CCN(C)CCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
        "expected_risk": "low",
    },
}


class CardiacChannelPredictor:
    def __init__(self, model_root: str | Path | None = None):
        root = Path(__file__).resolve().parents[2]
        self.model_root = Path(model_root) if model_root else root / "artifacts" / "models"
        self.models = {}
        self.metadata = {}
        for channel in CHANNELS:
            folder = self.model_root / channel
            classifier = XGBClassifier()
            classifier.load_model(folder / "classification.ubj")
            regressor = XGBRegressor()
            regressor.load_model(folder / "regression.ubj")
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if metadata["features"]["total_dimension"] != FEATURE_DIM:
                raise ValueError(f"Feature schema mismatch for {channel}")
            self.models[channel] = (classifier, regressor)
            self.metadata[channel] = metadata

    def predict(self, smiles: str) -> dict:
        canonical, features = featurize_smiles(smiles)
        x = features.reshape(1, -1)
        channels = {}
        for channel, (classifier, regressor) in self.models.items():
            probability = float(classifier.predict_proba(x)[0, 1])
            pic50 = float(regressor.predict(x)[0])
            channels[channel] = {
                "pIC50": pic50,
                "ic50_uM": float(10 ** (6 - pic50)),
                "blocker_probability": probability,
                "blocker": probability >= 0.5,
            }
        return {"smiles": canonical, "channels": channels}


def mice_model5(channels: dict) -> dict:
    # Kramer et al. Model 5 uses CavD = log10(IC50_Cav1.2 / IC50_hERG).
    cavd = channels["herg"]["pIC50"] - channels["cav1.2"]["pIC50"]
    coefficient = 3.1966488010380325
    intercept = -0.5463762419051946
    probability = 1.0 / (1.0 + math.exp(-(coefficient * cavd + intercept)))
    risk = "high" if cavd >= 1.0 else "intermediate" if cavd >= 0.0 else "low"
    return {
        "model": "MICE Model 5",
        "cavd": cavd,
        "difference_definition": "log10(IC50_Cav1.2 / IC50_hERG)",
        "log10_ic50_herg_over_cav12": -cavd,
        "probability": probability,
        "risk": risk,
    }


def validate_cipa_reference_drugs(predictor: CardiacChannelPredictor | None = None) -> dict:
    model = predictor or CardiacChannelPredictor()
    results = {}
    for name, reference in CIPA_REFERENCE_DRUGS.items():
        prediction = model.predict(reference["smiles"])
        tdp = mice_model5(prediction["channels"])
        results[name] = {
            "smiles": prediction["smiles"],
            "expected_risk": reference["expected_risk"],
            "predicted_risk": tdp["risk"],
            "passed": tdp["risk"] == reference["expected_risk"],
            "tdp": tdp,
            "channels": prediction["channels"],
        }
    return {
        "passed": all(item["passed"] for item in results.values()),
        "references": results,
    }
