import unittest
from benchmarking.retrieval_accuracy import compute_accuracy

class TestBenchmarking(unittest.TestCase):
    def test_accuracy(self):
        retrieved = [("Doc A", 1.5), ("Doc B", 1.2)]
        metrics = compute_accuracy(retrieved)
        self.assertTrue("precision" in metrics)