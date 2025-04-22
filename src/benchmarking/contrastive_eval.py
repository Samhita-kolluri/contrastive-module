import numpy as np
import logging
import json
import os

def compute_ranking_metrics(retrieved_docs, ground_truth_path=os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_docs.json")):
    logging.basicConfig(level="INFO")
    logger = logging.getLogger(__name__)
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    dcg = sum(1 / np.log2(i + 2) for i, (doc, _) in enumerate(retrieved_docs) if doc in ground_truth)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), len(retrieved_docs))))
    ndcg = dcg / idcg if idcg > 0 else 0
    mrr = next((1 / (i + 1) for i, (doc, _) in enumerate(retrieved_docs) if doc in ground_truth), 0)
    metrics = {"ndcg": ndcg, "mrr": mrr}
    logger.info(f"Ranking metrics: {metrics}")
    return metrics