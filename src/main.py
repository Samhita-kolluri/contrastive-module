import yaml
import logging
from typing import List, Dict, Any, Optional
import os
from src.embeddings.vectorizer import Vectorizer
from src.database.vector_db import VectorDB
from src.llm.llm_handler import LLMHandler
from src.llm.stance_transformation import StanceTransformer
from src.llm.topic_expansion import TopicExpander
from src.scoring.scoring import ContrastiveScorer

class ContrastiveRetriever:
    """Main class for contrastive document retrieval"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the retrieval system with configurations"""
        
        # logging setup
        logging.basicConfig(
            level=logging.INFO,  # logging level
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Log the loaded configuration
        self.logger.info(f"Loaded config: {self.config}")

        # Setup logging based on config
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        self.logger.info("Initializing ContrastiveRetriever")
        
        # Initialize components
        self.vectorizer = Vectorizer(self.config)
        self.db = VectorDB(self.config)
        self.llm_handler = LLMHandler(self.config)
        self.scorer = ContrastiveScorer(self.config)
        
        # Initialize auxiliary components
        self.topic_expander = TopicExpander()
        self.stance_transformer = StanceTransformer()
        
        self.logger.info("ContrastiveRetriever initialization complete")
        
    def retrieve(self, query_text: str, n_results: Optional[int] = None, include_discarded: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve contrastive documents for a given query
        
        Args:
            query_text: The input query text
            n_results: Number of results to return (defaults to config value)
            include_discarded: Whether to include documents that didn't pass thresholds
            
        Returns:
            List of contrastive documents with metadata
        """
        if n_results is None:
            n_results = self.config["settings"]["num_contrastive_docs"]
            
        self.logger.info(f"Retrieving contrastive documents for query: {query_text}")
        
        # Get query embeddings
        query_embeddings = self.vectorizer.embed([query_text])[0]
        self.logger.info("Generated query embeddings")
        
        # Generate new contrastive documents directly
        self.logger.info(f"Generating contrastive documents")
        contrastive_docs = self.llm_handler.generate_contrastive_docs(
            input_text=query_text,
            num_docs=n_results * 2  # Generate extras to allow for filtering
        )
        
        # Embed the generated documents
        generated_embeddings = self.vectorizer.embed(contrastive_docs)
        
        # Create result documents with embeddings
        results = []
        for i, (doc, emb) in enumerate(zip(contrastive_docs, generated_embeddings)):
            doc_id = f"generated_{i}"
            results.append({
                "id": doc_id,
                "text": doc,
                "topic_embedding": emb["topic_embedding"],
                "stance_embedding": emb["stance_embedding"],
                "metadata": {"source": "generated", "query": query_text}
            })
        
        # Store in database
        try:
            self.db.add_documents_batch(
                contrastive_docs, 
                [emb["topic_embedding"] for emb in generated_embeddings],
                [emb["stance_embedding"] for emb in generated_embeddings]
            )
        except Exception as e:
            self.logger.error(f"Error storing documents: {str(e)}")
                
        # Score and filter the results
        all_scored = self.scorer.score_documents(
            query_embedding=query_embeddings,
            documents=results
        )
        
        # Sort by final score and limit to requested number
        final_results = all_scored[:n_results]
        
        self.logger.info(f"Returning {len(final_results)} contrastive documents")
        return final_results


def main(query: str = None):
    """Run the contrastive retrieval pipeline"""
    retriever = ContrastiveRetriever()

    if query is None:
        # Interactive mode
        while True:
            query = input("Enter query (or 'q' to quit): ")
            if query.lower() == 'q':
                break
                
            results = retriever.retrieve(query)
            print("\nContrastive Documents:")
            for i, doc in enumerate(results):
                print(f"\n{i+1}. \"{doc['text']}\" (Topic similarity: {doc.get('topic_similarity', 0):.2f}, " +
                      f"Stance opposition: {doc.get('stance_opposition', 0):.2f} → " +
                      f"Final score: {doc.get('score', 0):.2f})")
    else:
        # Single query mode
        results = retriever.retrieve(query)
        print(f"\n**Input**: \"{query}\"")
        print("**Output**:")
        for i, doc in enumerate(results):
            print(f"{i+1}. \"{doc['text']}\" (Topic similarity: {doc.get('topic_similarity', 0):.2f}, " +
                  f"Stance opposition: {doc.get('stance_opposition', 0):.2f} → " +
                  f"Final score: {doc.get('score', 0):.2f})")

    return results

if __name__ == "__main__":
    main()
