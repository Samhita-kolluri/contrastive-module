from sentence_transformers import SentenceTransformer
import logging
import numpy as np
import os
from typing import List, Dict, Any

class Vectorizer:
    """Text embedding class that handles both topic and stance embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vectorizer with the specified models
        
        Args:
            config: Configuration dictionary with model paths
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load topic model
        topic_model_path = self.config["paths"]["topic_model"]
        self.logger.info(f"Loading topic model from {topic_model_path}")
        self.topic_model = SentenceTransformer(topic_model_path)

        # Load stance model
        stance_model_path = self.config["paths"].get("stance_model", None)
        
        if stance_model_path:
            try:
                if os.path.exists(stance_model_path):
                    self.logger.info(f"Loading local stance model from: {stance_model_path}")
                else:
                    self.logger.info(f"Loading remote HuggingFace stance model from: {stance_model_path}")
                self.stance_model = SentenceTransformer(stance_model_path)
            except Exception as e:
                self.logger.warning(f"Failed to load stance model from {stance_model_path}. Using topic model instead. Error: {e}")
                self.stance_model = self.topic_model
        else:
            self.logger.info("No stance model path provided. Using topic model as stance model.")
            self.stance_model = self.topic_model
        
        self.logger.info("Vectorizer initialized")
        
    def embed(self, texts: List[str]) -> List[Dict[str, np.ndarray]]:
        """
        Generate both topic and stance embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of dictionaries with "topic_embedding" and "stance_embedding" keys
        """
        if not texts:
            self.logger.warning("Empty list provided to embed method")
            return []
        
        self.logger.info(f"Embedding {len(texts)} texts")
        
        # Generate topic embeddings
        topic_embeddings = self.topic_model.encode(texts, convert_to_numpy=True)
        self.logger.info(f"Generated topic embeddings with shape {topic_embeddings.shape}")
        
        # Generate stance embeddings
        stance_embeddings = self.stance_model.encode(texts, convert_to_numpy=True)
        self.logger.info(f"Generated stance embeddings with shape {stance_embeddings.shape}")
        
        # Merge into output format
        results = []
        for i in range(len(texts)):
            results.append({
                "topic_embedding": topic_embeddings[i],
                "stance_embedding": stance_embeddings[i]
            })
        
        return results
