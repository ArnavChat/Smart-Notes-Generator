"""
Unit tests for PDF Handler module
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pdf_handler import PDFHandler

class TestPDFHandler(unittest.TestCase):
    """Test cases for PDFHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pdf_handler = PDFHandler()
        self.test_data_dir = Path(__file__).parent / "test_data"
        
    def test_init(self):
        """Test PDFHandler initialization."""
        self.assertIsInstance(self.pdf_handler, PDFHandler)
        self.assertEqual(self.pdf_handler.supported_formats, ['.pdf'])
    
    def test_validate_pdf_nonexistent_file(self):
        """Test validation with non-existent file."""
        result = self.pdf_handler.validate_pdf("nonexistent_file.pdf")
        self.assertFalse(result['is_valid'])
        self.assertIn("does not exist", result['error'])
    
    def test_extract_text_invalid_file(self):
        """Test text extraction with invalid file."""
        with self.assertRaises(FileNotFoundError):
            self.pdf_handler.extract_text("nonexistent_file.pdf")
    
    def test_extract_text_wrong_format(self):
        """Test text extraction with wrong file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(b"This is not a PDF")
            tmp_path = tmp_file.name
        
        try:
            with self.assertRaises(ValueError):
                self.pdf_handler.extract_text(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_clean_page_text(self):
        """Test page text cleaning functionality."""
        # Test basic cleaning
        dirty_text = "This  is   a   test\n\nwith\n\n\nexcessive   whitespace"
        cleaned = self.pdf_handler._clean_page_text(dirty_text)
        self.assertNotIn('\n\n', cleaned)
        self.assertNotIn('   ', cleaned)
        
        # Test hyphenated word fixing
        hyphenated_text = "This is a hyphen- ated word test"
        cleaned = self.pdf_handler._clean_page_text(hyphenated_text)
        self.assertIn('hyphenated', cleaned)
        self.assertNotIn('hyphen- ated', cleaned)
        
        # Test empty text
        self.assertEqual(self.pdf_handler._clean_page_text(""), "")
        self.assertEqual(self.pdf_handler._clean_page_text(None), "")
    
    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        test_text = "This is a test sentence. This is another sentence! And a third one?"
        stats = self.pdf_handler.get_text_statistics(test_text)
        
        self.assertIn('word_count', stats)
        self.assertIn('char_count', stats)
        self.assertIn('sentence_count', stats)
        self.assertIn('paragraph_count', stats)
        self.assertIn('avg_words_per_sentence', stats)
        self.assertIn('avg_chars_per_word', stats)
        self.assertIn('reading_time_minutes', stats)
        
        # Test with empty text
        empty_stats = self.pdf_handler.get_text_statistics("")
        self.assertEqual(empty_stats['word_count'], 0)
        self.assertEqual(empty_stats['char_count'], 0)
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        # This would require a real PDF file to test properly
        # For now, we'll test the structure
        pass
    
    def test_page_range_validation(self):
        """Test page range validation logic."""
        # This would require a real PDF file to test properly
        # We can test the error conditions
        pass

class TestPDFHandlerIntegration(unittest.TestCase):
    """Integration tests for PDFHandler (requires sample PDF files)."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.pdf_handler = PDFHandler()
        # Note: These tests would require actual PDF files
        # In a real scenario, you'd place test PDFs in tests/test_data/
    
    @unittest.skip("Requires sample PDF file")
    def test_extract_text_integration(self):
        """Test actual PDF text extraction."""
        # This test would use a real PDF file
        # sample_pdf = Path(__file__).parent / "test_data" / "sample.pdf"
        # result = self.pdf_handler.extract_text(sample_pdf)
        # self.assertIn('text', result)
        # self.assertIn('metadata', result)
        # self.assertIn('pages', result)
        pass
    
    @unittest.skip("Requires sample PDF file")
    def test_page_range_extraction(self):
        """Test page range extraction."""
        # This test would use a real PDF file
        pass

if __name__ == '__main__':
    # Create test data directory if it doesn't exist
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    
    # Run tests
    unittest.main()
