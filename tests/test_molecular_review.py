import unittest

from app.services.molecular_review import review_candidate


class MolecularReviewTests(unittest.TestCase):
    def test_identical_structure_passes_similarity_proxy(self):
        review = review_candidate("CCO", "CCO")
        self.assertEqual(review["structure_similarity"]["score"], 1.0)
        self.assertTrue(review["activity_retention_proxy"]["passed"])
        self.assertTrue(review["drug_likeness"]["passed"])

    def test_dissimilar_structure_fails_similarity_proxy(self):
        review = review_candidate("c1ccccc1", "CCCCCCCCCCCCCCCC")
        self.assertFalse(review["structure_similarity"]["passed"])
        self.assertFalse(review["activity_retention_proxy"]["passed"])


if __name__ == "__main__":
    unittest.main()
