from cardiac_channels.predictor import CardiacChannelPredictor, mice_model5


_predictor: CardiacChannelPredictor | None = None


def get_local_predictor() -> CardiacChannelPredictor:
    global _predictor

    if _predictor is None:
        print("[Local Toxicity] Loading three XGBoost channel models...")
        _predictor = CardiacChannelPredictor()
        print("[Local Toxicity] Three channel models loaded.")

    return _predictor


async def evaluate_toxicity_locally(smiles: str) -> dict:
    try:
        result = get_local_predictor().predict(smiles)
        tdp = mice_model5(result["channels"])
        result.update(
            {
                "probability": round(float(tdp["probability"]), 4),
                "is_toxic": tdp["risk"] != "low",
                "risk_level": tdp["risk"],
                "tdp": tdp,
            }
        )
        return result
    except ValueError as exc:
        return {"smiles": smiles, "error": str(exc)}
