import chromadb
import numpy as np
import logging
import yaml
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import os

class VectorDB:
    """Enhanced vector database for storing and retrieving contrastive documents"""
    
    def __init__(self, config: Dict[str, Any] = None, config_path: str = "configs/config.yaml"):
        """
        Initialize the vector database with configuration
        
        Args:
            config: Configuration dictionary (optional)
            config_path: Path to configuration file (used if config not provided)
        """
        if config is None:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing VectorDB")
        
        # Make sure the database directory exists
        db_path = self.config["paths"]["vector_db"]
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Set up collections
        self._setup_collections()
        
        self.logger.info("VectorDB initialized successfully")
        
    def _setup_collections(self):
        """Set up separate collections for topic and stance embeddings"""
        # Create or get collections for topic, stance, and combined embeddings
        self.topic_collection = self.client.get_or_create_collection(
            name="contrastive_docs_topic",
            metadata={"embedding_type": "topic"}
        )
        
        self.stance_collection = self.client.get_or_create_collection(
            name="contrastive_docs_stance",
            metadata={"embedding_type": "stance"}
        )
        
        # For backwards compatibility and combined queries
        self.combined_collection = self.client.get_or_create_collection(
            name="contrastive_docs",
            metadata={"embedding_type": "combined"}
        )
        
        self.logger.info("Collections set up: topic, stance, and combined")
        
    def add_document(self, doc_id: str, text: str, 
                    topic_embedding: np.ndarray, stance_embedding: np.ndarray,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document with topic and stance embeddings
        
        Args:
            doc_id: Unique document ID
            text: Document text
            topic_embedding: Topic embedding vector
            stance_embedding: Stance embedding vector
            metadata: Optional document metadata
            
        Returns:
            Document ID
        """
        if metadata is None:
            metadata = {}
            
        # Enhance metadata with document properties
        enhanced_metadata = {
            **metadata,
            "length": len(text.split()),
            "created_at": datetime.now().isoformat(),
        }
        
        try:
            # Convert numpy arrays to lists for ChromaDB
            topic_emb_list = topic_embedding.tolist() if isinstance(topic_embedding, np.ndarray) else topic_embedding
            stance_emb_list = stance_embedding.tolist() if isinstance(stance_embedding, np.ndarray) else stance_embedding
            
            # Add to topic collection
            self.topic_collection.add(
                ids=[doc_id],
                documents=[text],
                embeddings=[topic_emb_list],
                metadatas=[enhanced_metadata]
            )
            
            # Add to stance collection
            self.stance_collection.add(
                ids=[doc_id],
                documents=[text],
                embeddings=[stance_emb_list],
                metadatas=[enhanced_metadata]
            )
            
            # Add to combined collection
            combined_embedding = np.concatenate([topic_embedding, stance_embedding])
            combined_list = combined_embedding.tolist() if isinstance(combined_embedding, np.ndarray) else combined_embedding
            
            self.combined_collection.add(
                ids=[doc_id],
                documents=[text],
                embeddings=[combined_list],
                metadatas=[enhanced_metadata]
            )
            
            self.logger.info(f"Added document {doc_id} to all collections")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error adding document {doc_id}: {str(e)}")
            raise
            
    def add_documents_batch(self, texts: List[str], 
                           topic_embeddings: List[np.ndarray], 
                           stance_embeddings: List[np.ndarray],
                           metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add multiple documents in one batch operation
        
        Args:
            texts: List of document texts
            topic_embeddings: List of topic embeddings
            stance_embeddings: List of stance embeddings
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs
        """
        # Generate IDs if not provided
        doc_ids = [f"doc_{uuid.uuid4()}" for _ in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
            
        # Enhance all metadata
        enhanced_metadatas = []
        for i, text in enumerate(texts):
            enhanced_metadata = {
                **metadatas[i],
                "length": len(text.split()),
                "created_at": datetime.now().isoformat(),
            }
            enhanced_metadatas.append(enhanced_metadata)
            
        try:
            # Convert numpy arrays to lists
            topic_embs_list = [e.tolist() if isinstance(e, np.ndarray) else e for e in topic_embeddings]
            stance_embs_list = [e.tolist() if isinstance(e, np.ndarray) else e for e in stance_embeddings]
            
            # Add to topic collection
            self.topic_collection.add(
                ids=doc_ids,
                documents=texts,
                embeddings=topic_embs_list,
                metadatas=enhanced_metadatas
            )
            
            # Add to stance collection
            self.stance_collection.add(
                ids=doc_ids,
                documents=texts,
                embeddings=stance_embs_list,
                metadatas=enhanced_metadatas
            )
            
            # Add to combined collection
            combined_embeddings = [
                np.concatenate([t, s]).tolist() 
                for t, s in zip(topic_embeddings, stance_embeddings)
            ]
            
            self.combined_collection.add(
                ids=doc_ids,
                documents=texts,
                embeddings=combined_embeddings,
                metadatas=enhanced_metadatas
            )
            
            self.logger.info(f"Added batch of {len(texts)} documents")
            return doc_ids
            
        except Exception as e:
            self.logger.error(f"Error adding document batch: {str(e)}")
            raise
            
    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the database using a topic embedding
        
        Args:
            query_embedding: The query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of document dictionaries
        """
        try:
            # Convert numpy array to list
            query_emb_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            # Query the combined collection for simplicity
            results = self.combined_collection.query(
                query_embeddings=[query_emb_list],
                n_results=n_results,
                include=["documents", "embeddings", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:  # Check for empty results
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = {
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "embedding": results["embeddings"][0][i] if "embeddings" in results and results["embeddings"] else None,
                        "metadata": results["metadatas"][0][i] if "metadatas" in results and results["metadatas"] else {},
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                    }
                    formatted_results.append(doc)
                    
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error querying vector store: {str(e)}")
            return []
            
    def contrastive_query(self, query_text: str, 
                         topic_embedding: np.ndarray, 
                         stance_embedding: np.ndarray,
                         weights: Optional[List[float]] = None,
                         n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform weighted multi-vector retrieval for contrastive search
        
        Args:
            query_text: Original query text
            topic_embedding: Query topic embedding
            stance_embedding: Query stance embedding
            weights: List of weights [topic_weight, stance_weight]
            n_results: Number of results to return
            
        Returns:
            List of document dictionaries
        """
        if weights is None:
            weights = self.config["settings"]["weights"]
            
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
        
        try:
            # Convert numpy arrays to lists
            topic_emb_list = topic_embedding.tolist() if isinstance(topic_embedding, np.ndarray) else topic_embedding
            
            # Create an inverted stance embedding to find opposing views
            inverted_stance = -1 * stance_embedding
            stance_emb_list = inverted_stance.tolist() if isinstance(inverted_stance, np.ndarray) else inverted_stance
            
            # Get more candidates for reranking
            search_n = min(n_results * 3, 100)  # Get more results but cap at 100
            
            # Get similar topics
            topic_results = self.topic_collection.query(
                query_embeddings=[topic_emb_list],
                n_results=search_n,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Get opposing stances
            stance_results = self.stance_collection.query(
                query_embeddings=[stance_emb_list],
                n_results=search_n,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Combine and score results
            combined_scores = {}
            doc_info = {}
            
            # Process topic results
            if topic_results["ids"] and len(topic_results["ids"]) > 0 and len(topic_results["ids"][0]) > 0:
                for i, doc_id in enumerate(topic_results["ids"][0]):
                    if doc_id not in combined_scores:
                        combined_scores[doc_id] = 0
                        
                    # For topic similarity, lower distance is better
                    topic_score = 1 - topic_results["distances"][0][i]
                    combined_scores[doc_id] += weights[0] * topic_score
                    
                    # Store document info
                    doc_info[doc_id] = {
                        "id": doc_id,
                        "text": topic_results["documents"][0][i],
                        "topic_embedding": topic_results["embeddings"][0][i] if "embeddings" in topic_results and topic_results["embeddings"] else None,
                        "metadata": topic_results["metadatas"][0][i] if "metadatas" in topic_results and topic_results["metadatas"] else {},
                        "topic_similarity": topic_score
                    }
            
            # Process stance results
            if stance_results["ids"] and len(stance_results["ids"]) > 0 and len(stance_results["ids"][0]) > 0:
                for i, doc_id in enumerate(stance_results["ids"][0]):
                    if doc_id not in combined_scores:
                        combined_scores[doc_id] = 0
                        
                    # For stance opposition, we used inverted embedding, so lower distance is better
                    stance_score = 1 - stance_results["distances"][0][i]  # This will be the opposition score (higher = more opposing)
                    combined_scores[doc_id] += weights[1] * stance_score
                    
                    # Update document info if we have it
                    if doc_id in doc_info:
                        doc_info[doc_id]["stance_embedding"] = stance_results["embeddings"][0][i] if "embeddings" in stance_results and stance_results["embeddings"] else None
                        doc_info[doc_id]["stance_opposition"] = stance_score
                    else:
                        # This document was only in stance results
                        doc_info[doc_id] = {
                            "id": doc_id,
                            "text": stance_results["documents"][0][i],
                            "stance_embedding": stance_results["embeddings"][0][i] if "embeddings" in stance_results and stance_results["embeddings"] else None,
                            "metadata": stance_results["metadatas"][0][i] if "metadatas" in stance_results and stance_results["metadatas"] else {},
                            "stance_opposition": stance_score
                        }
            
            # Sort by combined score and get top results
            ranked_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:n_results]
            
            # Create final results list
            final_results = []
            for doc_id in ranked_ids:
                doc = doc_info[doc_id]
                doc["score"] = combined_scores[doc_id]
                final_results.append(doc)
                
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in contrastive query: {str(e)}")
            return []