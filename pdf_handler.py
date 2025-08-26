"""
PDF Handler Module
Handles PDF text extraction using PyMuPDF (fitz) with comprehensive error handling.
"""

import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import pandas as pd

logger = logging.getLogger(__name__)

class PDFHandler:
    """
    Handles PDF text extraction and metadata retrieval using PyMuPDF.
    Provides robust text extraction with preprocessing and error handling.
    """
    
    def __init__(self):
        """Initialize the PDF handler."""
        self.supported_formats = ['.pdf']
        
    def extract_text(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and page information
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {pdf_path.suffix}")
        
        try:
            # Open PDF document
            doc = fitz.open(str(pdf_path))
            
            # Extract metadata
            metadata = self._extract_metadata(doc)
            
            # Extract text from all pages
            pages_data = []
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Clean and process page text
                cleaned_text = self._clean_page_text(page_text)
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'text': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text)
                })
                
                full_text += cleaned_text + "\n\n"
            
            doc.close()
            
            # Compile final results
            result = {
                'text': full_text.strip(),
                'metadata': metadata,
                'pages': pages_data,
                'total_pages': len(pages_data),
                'total_words': sum(page['word_count'] for page in pages_data),
                'total_chars': sum(page['char_count'] for page in pages_data)
            }
            
            logger.info(f"Successfully extracted text from {pdf_path.name}: "
                       f"{result['total_pages']} pages, {result['total_words']} words")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict:
        """
        Extract metadata from PDF document.
        
        Args:
            doc: PyMuPDF Document object
            
        Returns:
            Dictionary containing document metadata
        """
        metadata = doc.metadata
        
        return {
            'title': metadata.get('title', '').strip() or 'Unknown',
            'author': metadata.get('author', '').strip() or 'Unknown',
            'subject': metadata.get('subject', '').strip() or 'Unknown',
            'creator': metadata.get('creator', '').strip() or 'Unknown',
            'producer': metadata.get('producer', '').strip() or 'Unknown',
            'creation_date': metadata.get('creationDate', '').strip() or 'Unknown',
            'modification_date': metadata.get('modDate', '').strip() or 'Unknown',
            'pages': doc.page_count,
            'encrypted': doc.is_encrypted,
            'pdf_version': getattr(doc, 'pdf_version', 'Unknown')
        }
    
    def _clean_page_text(self, text: str) -> str:
        """
        Clean and preprocess text extracted from a PDF page.
        
        Args:
            text: Raw text from PDF page
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers (simple heuristic)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove headers/footers (basic pattern)
        text = re.sub(r'^[A-Z\s]+$', '', text, flags=re.MULTILINE)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        return text.strip()
    
    def extract_page_range(self, pdf_path: Union[str, Path], 
                          start_page: int, end_page: int) -> Dict:
        """
        Extract text from a specific range of pages.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number (1-indexed)
            end_page: Ending page number (1-indexed)
            
        Returns:
            Dictionary containing extracted text and metadata for the page range
        """
        pdf_path = Path(pdf_path)
        
        try:
            doc = fitz.open(str(pdf_path))
            
            # Validate page range
            if start_page < 1 or end_page > len(doc) or start_page > end_page:
                raise ValueError(f"Invalid page range: {start_page}-{end_page} for document with {len(doc)} pages")
            
            pages_data = []
            full_text = ""
            
            for page_num in range(start_page - 1, end_page):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                cleaned_text = self._clean_page_text(page_text)
                
                pages_data.append({
                    'page_number': page_num + 1,
                    'text': cleaned_text,
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text)
                })
                
                full_text += cleaned_text + "\n\n"
            
            doc.close()
            
            return {
                'text': full_text.strip(),
                'pages': pages_data,
                'page_range': f"{start_page}-{end_page}",
                'total_words': sum(page['word_count'] for page in pages_data),
                'total_chars': sum(page['char_count'] for page in pages_data)
            }
            
        except Exception as e:
            logger.error(f"Error extracting page range {start_page}-{end_page} from {pdf_path}: {str(e)}")
            raise Exception(f"Failed to extract page range: {str(e)}")
    
    def get_page_text(self, pdf_path: Union[str, Path], page_number: int) -> str:
        """
        Extract text from a single page.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            Cleaned text from the specified page
        """
        try:
            result = self.extract_page_range(pdf_path, page_number, page_number)
            return result['text']
        except Exception as e:
            logger.error(f"Error extracting page {page_number} from {pdf_path}: {str(e)}")
            raise Exception(f"Failed to extract page {page_number}: {str(e)}")
    
    def validate_pdf(self, pdf_path: Union[str, Path]) -> Dict:
        """
        Validate PDF file and return basic information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing validation results and basic info
        """
        pdf_path = Path(pdf_path)
        
        validation_result = {
            'is_valid': False,
            'error': None,
            'file_size': 0,
            'page_count': 0,
            'is_encrypted': False,
            'has_text': False
        }
        
        try:
            if not pdf_path.exists():
                validation_result['error'] = "File does not exist"
                return validation_result
            
            validation_result['file_size'] = pdf_path.stat().st_size
            
            # Try to open the PDF
            doc = fitz.open(str(pdf_path))
            validation_result['page_count'] = len(doc)
            validation_result['is_encrypted'] = doc.is_encrypted
            
            # Check if PDF has extractable text
            if len(doc) > 0:
                sample_page = doc.load_page(0)
                sample_text = sample_page.get_text().strip()
                validation_result['has_text'] = len(sample_text) > 0
            
            doc.close()
            validation_result['is_valid'] = True
            
        except Exception as e:
            validation_result['error'] = str(e)
        
        return validation_result
    
    def get_text_statistics(self, text: str) -> Dict:
        """
        Generate statistics about extracted text.
        
        Args:
            text: Extracted text
            
        Returns:
            Dictionary containing text statistics
        """
        if not text:
            return {
                'word_count': 0,
                'char_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'avg_words_per_sentence': 0,
                'avg_chars_per_word': 0
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        # Filter out empty elements
        words = [w for w in words if w.strip()]
        sentences = [s for s in sentences if s.strip()]
        paragraphs = [p for p in paragraphs if p.strip()]
        
        word_count = len(words)
        char_count = len(text)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0,
            'avg_chars_per_word': char_count / word_count if word_count > 0 else 0,
            'reading_time_minutes': word_count / 200  # Assuming 200 words per minute
        }
