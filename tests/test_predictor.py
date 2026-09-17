import math
import unittest

from cardiac_channels.features import FEATURE_DIM, featurize_smiles
from cardiac_channels.predictor import (
    CardiacChannelPredictor,
    mice_model5,
    validate_cipa_reference_drugs,
)


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = CardiacChannelPredictor()

    def test_invalid_smiles(self):
        with self.assertRaises(ValueError):
            self.predictor.predict("not-a-smiles")

    def test_feature_dimension(self):
        _, features = featurize_smiles("CCO")
        self.assertEqual(features.shape, (FEATURE_DIM,))

    def test_three_channel_prediction_is_stable(self):
        first = self.predictor.predict("CCO")
        second = self.predictor.predict("CCO")
        self.assertEqual(set(first["channels"]), {"herg", "cav1.2", "nav1.5"})
        self.assertEqual(first, second)
        for result in first["channels"].values():
            self.assertTrue(math.isfinite(result["pIC50"]))
            self.assertGreater(result["ic50_uM"], 0)
            self.assertGreaterEqual(result["blocker_probability"], 0)
            self.assertLessEqual(result["blocker_probability"], 1)

    def test_mice_cavd_direction(self):
        channels = {
            "herg": {"pIC50": 7.0},
            "cav1.2": {"pIC50": 5.0},
            "nav1.5": {"pIC50": 4.0},
        }
        result = mice_model5(channels)
        self.assertEqual(result["cavd"], 2.0)
        self.assertEqual(result["log10_ic50_herg_over_cav12"], -2.0)
        self.assertEqual(result["risk"], "high")

    def test_cipa_reference_drugs(self):
        validation = validate_cipa_reference_drugs(self.predictor)
        self.assertTrue(validation["passed"])
        self.assertEqual(
            validation["references"]["dofetilide"]["predicted_risk"], "high"
        )
        self.assertEqual(
            validation["references"]["verapamil"]["predicted_risk"], "low"
        )


if __name__ == "__main__":
    unittest.main()
