"""
Unit tests for Exporter module
"""

import unittest
import tempfile
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from exporter import NotesExporter

class TestNotesExporter(unittest.TestCase):
    """Test cases for NotesExporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exporter = NotesExporter()
        self.sample_notes_data = {
            'summary': 'This is a test summary of the document content. It covers the main points and key insights.',
            'keywords': ['artificial intelligence', 'machine learning', 'natural language processing', 'deep learning'],
            'chapters': [
                {
                    'title': 'Introduction to AI',
                    'content': 'This chapter introduces the basic concepts of artificial intelligence and its applications.'
                },
                {
                    'title': 'Machine Learning Fundamentals',
                    'content': 'This chapter covers the fundamental concepts of machine learning including supervised and unsupervised learning.'
                }
            ],
            'metadata': {
                'title': 'Test Document',
                'author': 'Test Author',
                'pages': 25,
                'subject': 'Computer Science',
                'creation_date': '2024-01-01'
            }
        }
    
    def test_init(self):
        """Test NotesExporter initialization."""
        self.assertIsInstance(self.exporter, NotesExporter)
        self.assertEqual(self.exporter.supported_formats, ['.docx', '.txt', '.md'])
    
    def test_export_notes_invalid_format(self):
        """Test export with invalid format."""
        with self.assertRaises(ValueError):
            self.exporter.export_notes(self.sample_notes_data, format_type='.pdf')
    
    def test_export_to_txt(self):
        """Test export to TXT format."""
        result = self.exporter.export_notes(
            self.sample_notes_data, 
            format_type='.txt', 
            filename='test_export'
        )
        
        self.assertIsInstance(result, bytes)
        content = result.decode('utf-8')
        
        # Check if content contains expected sections
        self.assertIn('SMART NOTES SUMMARY', content)
        self.assertIn('DOCUMENT INFORMATION', content)
        self.assertIn('SUMMARY', content)
        self.assertIn('KEY TERMS', content)
        self.assertIn('CHAPTER BREAKDOWN', content)
        
        # Check if data is present
        self.assertIn(self.sample_notes_data['summary'], content)
        self.assertIn('Test Document', content)
        self.assertIn('artificial intelligence', content)
        self.assertIn('Introduction to AI', content)
    
    def test_export_to_markdown(self):
        """Test export to Markdown format."""
        result = self.exporter.export_notes(
            self.sample_notes_data, 
            format_type='.md', 
            filename='test_export'
        )
        
        self.assertIsInstance(result, bytes)
        content = result.decode('utf-8')
        
        # Check if content contains expected Markdown elements
        self.assertIn('# Smart Notes Summary', content)
        self.assertIn('## Document Information', content)
        self.assertIn('## Summary', content)
        self.assertIn('## Key Terms', content)
        self.assertIn('### Chapter 1:', content)
        
        # Check for Markdown table
        self.assertIn('| Property | Value |', content)
        self.assertIn('|----------|-------|', content)
        
        # Check for bullet points
        self.assertIn('- `artificial intelligence`', content)
    
    @unittest.skip("Requires python-docx library")
    def test_export_to_docx(self):
        """Test export to DOCX format."""
        result = self.exporter.export_notes(
            self.sample_notes_data, 
            format_type='.docx', 
            filename='test_export'
        )
        
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 1000)  # DOCX files should be reasonably sized
    
    def test_wrap_text(self):
        """Test text wrapping functionality."""
        long_text = "This is a very long line of text that should be wrapped to multiple lines when the width limit is reached."
        wrapped = self.exporter._wrap_text(long_text, width=20)
        
        lines = wrapped.split('\n')
        self.assertGreater(len(lines), 1)  # Should create multiple lines
        
        # Each line should be within the width limit (allowing for last line)
        for line in lines[:-1]:  # All but the last line
            self.assertLessEqual(len(line), 25)  # Allow some buffer for word boundaries
    
    def test_get_export_preview_txt(self):
        """Test export preview for TXT format."""
        preview = self.exporter.get_export_preview(
            self.sample_notes_data,
            format_type='.txt',
            max_length=200
        )
        
        self.assertIsInstance(preview, str)
        self.assertLessEqual(len(preview), 203)  # Should respect max_length (plus ...)
        self.assertIn('SMART NOTES', preview)
    
    def test_get_export_preview_md(self):
        """Test export preview for Markdown format."""
        preview = self.exporter.get_export_preview(
            self.sample_notes_data,
            format_type='.md',
            max_length=200
        )
        
        self.assertIsInstance(preview, str)
        self.assertLessEqual(len(preview), 203)  # Should respect max_length (plus ...)
        self.assertIn('# Smart Notes', preview)
    
    def test_get_export_preview_unsupported(self):
        """Test export preview for unsupported format."""
        preview = self.exporter.get_export_preview(
            self.sample_notes_data,
            format_type='.docx'
        )
        
        self.assertIn('Preview not available', preview)
    
    def test_export_with_minimal_data(self):
        """Test export with minimal data."""
        minimal_data = {
            'summary': 'A simple summary.',
            'keywords': [],
            'chapters': [],
            'metadata': {}
        }
        
        result = self.exporter.export_notes(minimal_data, format_type='.txt')
        self.assertIsInstance(result, bytes)
        
        content = result.decode('utf-8')
        self.assertIn('A simple summary.', content)
    
    def test_export_with_no_chapters(self):
        """Test export when chapters are not provided."""
        data_no_chapters = self.sample_notes_data.copy()
        data_no_chapters['chapters'] = []
        
        result = self.exporter.export_notes(data_no_chapters, format_type='.txt')
        content = result.decode('utf-8')
        
        # Should still contain other sections
        self.assertIn('SUMMARY', content)
        self.assertIn('KEY TERMS', content)
        # Should not contain chapter section or should handle gracefully
    
    def test_export_with_no_keywords(self):
        """Test export when keywords are not provided."""
        data_no_keywords = self.sample_notes_data.copy()
        data_no_keywords['keywords'] = []
        
        result = self.exporter.export_notes(data_no_keywords, format_type='.txt')
        content = result.decode('utf-8')
        
        # Should still contain other sections
        self.assertIn('SUMMARY', content)
        self.assertIn('CHAPTER BREAKDOWN', content)

class TestNotesExporterIntegration(unittest.TestCase):
    """Integration tests for NotesExporter."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.exporter = NotesExporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_to_file_txt(self):
        """Test saving exported content to file."""
        sample_data = {
            'summary': 'Test summary',
            'keywords': ['test', 'export'],
            'chapters': [],
            'metadata': {'title': 'Test Doc'}
        }
        
        content = self.exporter.export_notes(sample_data, format_type='.txt')
        filepath = Path(self.temp_dir) / 'test_output.txt'
        
        self.exporter.save_to_file(content, filepath, '.txt')
        
        # Verify file was created and contains expected content
        self.assertTrue(filepath.exists())
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
        self.assertIn('Test summary', file_content)
    
    @unittest.skip("Requires python-docx library")
    def test_save_to_file_docx(self):
        """Test saving DOCX file."""
        sample_data = {
            'summary': 'Test summary',
            'keywords': ['test', 'export'],
            'chapters': [],
            'metadata': {'title': 'Test Doc'}
        }
        
        content = self.exporter.export_notes(sample_data, format_type='.docx')
        filepath = Path(self.temp_dir) / 'test_output.docx'
        
        self.exporter.save_to_file(content, filepath, '.docx')
        
        # Verify file was created
        self.assertTrue(filepath.exists())
        self.assertGreater(filepath.stat().st_size, 100)  # Should have substantial content

if __name__ == '__main__':
    unittest.main()
