import unittest
from scoring.scoring import ContrastiveScorer
import numpy as np

class TestScoring(unittest.TestCase):
    def test_final_score(self):
        scorer = ContrastiveScorer()
        score = scorer.compute_final_score(0.8, 0.9)
        self.assertAlmostEqual(score, 1.52, places=2)