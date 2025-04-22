import os
import yaml
import logging

from src.llm.llm_handler import LLMHandler
from src.database.vector_db import VectorDB
from src.scoring.scoring import ContrastiveScorer
from src.api.retrieve_top_docs import retrieve_top_docs

class APIServer:
    def __init__(self, config_path=None):
        # Load config file
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Logging setup
        logging_level = self.config.get("logging", {}).get("level", "INFO")
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Contrastive Search Module...")

        # Instantiate components with config
        self.llm_handler = LLMHandler(config=self.config)
        self.vector_db = VectorDB(config=self.config)
        self.scorer = ContrastiveScorer(config=self.config)

    async def contrastive_search(self, data: dict):
        input_text = data["text"]
        self.logger.info(f"Processing input: {input_text[:80]}...")

        contrastive_docs = self.llm_handler.generate_contrastive_docs(input_text)

        all_docs = [input_text] + contrastive_docs
        embeddings = self.llm_handler.embed_documents(all_docs)

        for i, (doc, emb) in enumerate(zip(all_docs, embeddings)):
            self.vector_db.add_document(f"doc_{i}", doc, emb["topic_embedding"], emb["stance_embedding"])

        query_emb = embeddings[0]
        candidates = self.vector_db.query(query_emb["topic_embedding"])
        ranked_docs = self.scorer.score_documents(query_emb, candidates)

        top_docs = retrieve_top_docs(ranked_docs)
        return {"results": top_docs}
