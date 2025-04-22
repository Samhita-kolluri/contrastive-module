import unittest
from llm.llm_handler import LLMHandler

class TestLLM(unittest.TestCase):
    def test_generate_docs(self):
        handler = LLMHandler()
        docs = handler.generate_contrastive_docs("AI improves healthcare")
        self.assertEqual(len(docs), handler.num_docs)

# tests/test_llm_handler.py
import pytest
from src.llm.llm_handler import LLMHandler
from src.config import Config

@pytest.fixture
def llm_handler():
    config = Config.load("config/config.yaml")
    return LLMHandler(config)

def test_generate_contrastive_docs(llm_handler):
    docs = llm_handler.generate_contrastive_docs("Two men, one playing a purple guitar and the other playing an accordion, sitting on stone steps.")
    assert len(docs) == 5
    assert all(isinstance(doc, str) and len(doc.split()) > 20 for doc in docs)