import numpy as np
import logging
import json
import os

def compute_accuracy(retrieved_docs, ground_truth_path=os.path.join(os.path.dirname(__file__), "..", "..", "data", "contradicition_data.json")):
    logging.basicConfig(level="INFO")
    logger = logging.getLogger(__name__)
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    retrieved_texts = [doc for doc, _ in retrieved_docs]
    true_positives = len(set(retrieved_texts) & set(ground_truth))
    precision = true_positives / len(retrieved_docs) if retrieved_docs else 0
    recall = true_positives / len(ground_truth) if ground_truth else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    metrics = {"precision": precision, "recall": recall, "f1": f1}
    logger.info(f"Accuracy metrics: {metrics}")
    return metrics

if __name__ == "__main__":
    # Sample usage
    retrieved = [("AI improves healthcare", 1.5), ("AI harms healthcare", 1.2)]
    metrics = compute_accuracy(retrieved)
    print(metrics)