from src.database.vector_db import VectorDB

def retrieve_embeddings(query_embedding, n_results=10):
    db = VectorDB()
    return db.query(query_embedding, n_results)