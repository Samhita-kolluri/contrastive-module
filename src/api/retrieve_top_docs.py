import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

def retrieve_top_docs(ranked_docs, n=3):
    logger.info(f"Retrieving top {n} documents from {len(ranked_docs)} ranked docs")
    sorted_docs = sorted(ranked_docs, key=lambda x: x["score"], reverse=True)
    return sorted_docs[:n]