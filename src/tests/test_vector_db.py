import unittest
from database.vector_db import VectorDB
from embeddings.vectorizer import Vectorizer

class TestVectorDB(unittest.TestCase):
    def test_add_and_query(self):
        db = VectorDB()
        vectorizer = Vectorizer()
        emb = vectorizer.embed_document("Test text")
        db.add_document("test1", "Test text", emb["topic_embedding"], emb["stance_embedding"])
        results = db.query(emb["topic_embedding"], n_results=1)
        self.assertEqual(results["documents"][0], ["Test text"])
