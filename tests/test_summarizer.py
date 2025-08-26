"""
Unit tests for Summarizer module
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from summarizer import SmartSummarizer

class TestSmartSummarizer(unittest.TestCase):
    """Test cases for SmartSummarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.summarizer = SmartSummarizer()
        self.sample_text = """
        This is a sample text for testing the summarization functionality.
        It contains multiple sentences and should be long enough to generate a meaningful summary.
        The text discusses various topics including natural language processing, machine learning, and artificial intelligence.
        These technologies are becoming increasingly important in modern software development.
        Summarization is a key task in natural language processing that involves condensing text while preserving important information.
        There are two main types of summarization: extractive and abstractive.
        Extractive summarization selects important sentences from the original text.
        Abstractive summarization generates new sentences that capture the essence of the original text.
        Both approaches have their advantages and disadvantages depending on the use case.
        """
        
    def test_init(self):
        """Test SmartSummarizer initialization."""
        self.assertIsInstance(self.summarizer, SmartSummarizer)
        self.assertIn(self.summarizer.device, ['cuda', 'cpu'])
    
    def test_generate_summary_short_text(self):
        """Test summary generation with text too short."""
        short_text = "This is too short."
        result = self.summarizer.generate_summary(short_text)
        self.assertIn("too short", result.lower())
    
    def test_generate_summary_extractive(self):
        """Test extractive summarization."""
        summary = self.summarizer.generate_summary(
            self.sample_text, 
            mode="extractive", 
            max_length=100
        )
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLess(len(summary.split()), 120)  # Should respect max_length roughly
    
    def test_generate_summary_invalid_mode(self):
        """Test summary generation with invalid mode."""
        result = self.summarizer.generate_summary(
            self.sample_text, 
            mode="invalid_mode"
        )
        self.assertIn("Error", result)
    
    def test_extract_keywords_yake(self):
        """Test keyword extraction using YAKE."""
        keywords = self.summarizer.extract_keywords(
            self.sample_text, 
            num_keywords=10, 
            method="yake"
        )
        
        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 10)
        
        # Check if keywords are relevant
        text_lower = self.sample_text.lower()
        for keyword in keywords[:5]:  # Check first 5 keywords
            self.assertTrue(any(word in text_lower for word in keyword.lower().split()))
    
    def test_extract_keywords_tfidf(self):
        """Test keyword extraction using TF-IDF."""
        keywords = self.summarizer.extract_keywords(
            self.sample_text, 
            num_keywords=10, 
            method="tfidf"
        )
        
        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 10)
    
    def test_extract_keywords_empty_text(self):
        """Test keyword extraction with empty text."""
        keywords = self.summarizer.extract_keywords("", num_keywords=10)
        self.assertEqual(keywords, [])
        
        keywords = self.summarizer.extract_keywords("Short text", num_keywords=10)
        self.assertEqual(keywords, [])
    
    def test_extract_keywords_invalid_method(self):
        """Test keyword extraction with invalid method."""
        keywords = self.summarizer.extract_keywords(
            self.sample_text, 
            method="invalid_method"
        )
        # Should fallback to YAKE
        self.assertIsInstance(keywords, list)
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        dirty_text = "This  text   has\n\nexcessive\n\n\nwhitespace and hyphen- ated words."
        cleaned = self.summarizer._preprocess_text(dirty_text)
        
        self.assertNotIn('\n\n', cleaned)
        self.assertNotIn('   ', cleaned)
        self.assertIn('hyphenated', cleaned)
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        long_text = "This is a sentence. " * 100  # Create long text
        chunks = self.summarizer._chunk_text(long_text, max_length=500)
        
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        
        # Each chunk should be within size limit
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 550)  # Allow some buffer
    
    def test_post_process_summary(self):
        """Test summary post-processing."""
        # Test punctuation addition
        summary_no_punct = "This is a summary without punctuation"
        processed = self.summarizer._post_process_summary(summary_no_punct, 100)
        self.assertTrue(processed.endswith('.'))
        
        # Test length truncation
        long_summary = "Word " * 200  # 200 words
        processed = self.summarizer._post_process_summary(long_summary, 50)
        self.assertLessEqual(len(processed.split()), 60)  # Should be roughly 50 words
    
    def test_fallback_extractive_summary(self):
        """Test fallback extractive summarization."""
        summary = self.summarizer._fallback_extractive_summary(self.sample_text, 100)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLess(len(summary.split()), 120)

class TestSmartSummarizerIntegration(unittest.TestCase):
    """Integration tests for SmartSummarizer."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.summarizer = SmartSummarizer()
        # Longer, more realistic text for integration testing
        self.long_text = """
        Natural language processing (NLP) is a subfield of linguistics, computer science, 
        and artificial intelligence concerned with the interactions between computers and human language, 
        in particular how to program computers to process and analyze large amounts of natural language data.
        
        The goal is a computer capable of understanding the contents of documents, including the contextual 
        nuances of the language within them. The technology can then accurately extract information and insights 
        contained in the documents as well as categorize and organize the documents themselves.
        
        Challenges in natural language processing frequently involve speech recognition, natural language 
        understanding, and natural language generation. Modern deep learning techniques have been very successful 
        in addressing these challenges, particularly in recent years with the development of transformer architectures.
        
        Applications of NLP include sentiment analysis, machine translation, information extraction, 
        question answering systems, and text summarization. These applications have found widespread use 
        in industries ranging from healthcare and finance to entertainment and education.
        
        The field continues to evolve rapidly with advances in machine learning and the availability 
        of large-scale datasets. Current research focuses on improving model efficiency, reducing bias, 
        and developing more robust systems that can handle diverse linguistic patterns and cultural contexts.
        """ * 3  # Make it longer for better testing
    
    @unittest.skip("Requires model downloads - enable for full integration testing")
    def test_abstractive_summarization_integration(self):
        """Test abstractive summarization with model loading."""
        summary = self.summarizer.generate_summary(
            self.long_text,
            mode="abstractive",
            max_length=150
        )
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 50)
        self.assertLess(len(summary.split()), 200)
    
    def test_extractive_algorithms_comparison(self):
        """Test different extractive algorithms."""
        textrank_summary = self.summarizer._generate_extractive_summary(
            self.long_text, 100, 30, "textrank"
        )
        lsa_summary = self.summarizer._generate_extractive_summary(
            self.long_text, 100, 30, "lsa"
        )
        
        self.assertIsInstance(textrank_summary, str)
        self.assertIsInstance(lsa_summary, str)
        self.assertGreater(len(textrank_summary), 0)
        self.assertGreater(len(lsa_summary), 0)
        
        # Summaries might be different but both should be reasonable
        self.assertLess(len(textrank_summary.split()), 120)
        self.assertLess(len(lsa_summary.split()), 120)

if __name__ == '__main__':
    unittest.main()
