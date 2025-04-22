import unittest
from src.embeddings.vectorizer import Vectorizer

class TestEmbeddings(unittest.TestCase):
    def test_embed_shape(self):
        vectorizer = Vectorizer()
        emb = vectorizer.embed_document("Test text")
        self.assertEqual(emb["topic_embedding"].shape, (1024,))