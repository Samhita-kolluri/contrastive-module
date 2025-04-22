from src.database.vector_db import VectorDB

def store_embeddings(documents, embeddings):
    db = VectorDB()
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        db.add_document(f"doc_{i}", doc, emb["topic_embedding"], emb["stance_embedding"])