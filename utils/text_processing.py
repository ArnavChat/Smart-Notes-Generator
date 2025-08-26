"""
Text Processing Utilities
Provides text cleaning, preprocessing, and segmentation functions.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
import string
import unicodedata

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Handles text preprocessing, cleaning, and segmentation tasks.
    """
    
    def __init__(self):
        """Initialize the text processor."""
        # Compile regex patterns for efficiency
        self.patterns = {
            'whitespace': re.compile(r'\s+'),
            'line_breaks': re.compile(r'\n+'),
            'page_numbers': re.compile(r'^\d+\s*$', re.MULTILINE),
            'headers_footers': re.compile(r'^[A-Z\s]{10,}$', re.MULTILINE),
            'hyphenated': re.compile(r'(\w+)-\s*\n\s*(\w+)'),
            'excessive_punct': re.compile(r'[.]{3,}|[-]{3,}'),
            'chapter_markers': re.compile(r'(?i)(chapter|section|part)\s+(\d+|[ivx]+)', re.MULTILINE),
            'bullet_points': re.compile(r'^[\s]*[•·▪▫◦‣⁃]\s*', re.MULTILINE),
            'numbers_list': re.compile(r'^[\s]*\d+[\.\)]\s*', re.MULTILINE),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
        }
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean and preprocess text extracted from PDF.
        
        Args:
            text: Raw text to clean
            aggressive: Whether to apply aggressive cleaning
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t ')
        
        # Fix hyphenated words broken across lines
        text = self.patterns['hyphenated'].sub(r'\1\2', text)
        
        # Normalize line breaks and whitespace
        text = self.patterns['line_breaks'].sub(' ', text)
        text = self.patterns['whitespace'].sub(' ', text)
        
        # Remove page numbers (standalone numbers)
        text = self.patterns['page_numbers'].sub('', text)
        
        # Remove likely headers/footers (all caps lines)
        text = self.patterns['headers_footers'].sub('', text)
        
        # Fix excessive punctuation
        text = self.patterns['excessive_punct'].sub(lambda m: '...' if '.' in m.group() else '---', text)
        
        if aggressive:
            text = self._aggressive_cleaning(text)
        
        # Final cleanup
        text = self.patterns['whitespace'].sub(' ', text)
        text = text.strip()
        
        return text
    
    def _aggressive_cleaning(self, text: str) -> str:
        """
        Apply aggressive cleaning for better summarization.
        
        Args:
            text: Text to clean aggressively
            
        Returns:
            Aggressively cleaned text
        """
        # Remove URLs and email addresses
        text = self.patterns['url'].sub('[URL]', text)
        text = self.patterns['email'].sub('[EMAIL]', text)
        
        # Remove excessive special characters
        text = re.sub(r'[^\w\s.,!?;:()\[\]"\'-]', ' ', text)
        
        # Remove single characters (likely artifacts)
        text = re.sub(r'\b\w\b', '', text)
        
        # Remove very short words (less than 2 characters)
        words = text.split()
        words = [word for word in words if len(word) >= 2 or word in string.punctuation]
        text = ' '.join(words)
        
        return text
    
    def segment_sentences(self, text: str, min_length: int = 10) -> List[str]:
        """
        Segment text into sentences with length filtering.
        
        Args:
            text: Input text
            min_length: Minimum sentence length in characters
            
        Returns:
            List of sentence strings
        """
        if not text:
            return []
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= min_length:
                # Ensure sentence ends with punctuation
                if sentence and sentence[-1] not in '.!?':
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def segment_paragraphs(self, text: str, min_length: int = 50) -> List[str]:
        """
        Segment text into paragraphs.
        
        Args:
            text: Input text
            min_length: Minimum paragraph length in characters
            
        Returns:
            List of paragraph strings
        """
        if not text:
            return []
        
        # Split on double line breaks (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            paragraph = self.patterns['whitespace'].sub(' ', paragraph).strip()
            if len(paragraph) >= min_length:
                cleaned_paragraphs.append(paragraph)
        
        return cleaned_paragraphs
    
    def segment_chapters(self, text: str) -> List[Dict[str, str]]:
        """
        Detect and segment text into chapters.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with chapter title and content
        """
        if not text:
            return []
        
        # Find chapter markers
        chapter_matches = list(self.patterns['chapter_markers'].finditer(text))
        
        if len(chapter_matches) < 2:  # Need at least 2 chapters to segment
            return []
        
        chapters = []
        
        for i, match in enumerate(chapter_matches):
            # Extract chapter title
            title_start = match.start()
            title_end = text.find('\n', title_start)
            if title_end == -1:
                title_end = title_start + 100  # Fallback
            
            title = text[title_start:title_end].strip()
            
            # Extract chapter content
            content_start = title_end
            if i + 1 < len(chapter_matches):
                content_end = chapter_matches[i + 1].start()
            else:
                content_end = len(text)
            
            content = text[content_start:content_end].strip()
            content = self.clean_text(content)
            
            if content and len(content) > 100:  # Only include substantial chapters
                chapters.append({
                    'title': title,
                    'content': content,
                    'word_count': len(content.split()),
                    'position': i + 1
                })
        
        return chapters
    
    def extract_bullet_points(self, text: str) -> List[str]:
        """
        Extract bullet points and numbered lists from text.
        
        Args:
            text: Input text
            
        Returns:
            List of bullet point strings
        """
        if not text:
            return []
        
        bullet_points = []
        
        # Find bullet points
        bullet_matches = self.patterns['bullet_points'].findall(text)
        number_matches = self.patterns['numbers_list'].findall(text)
        
        # Process bullet points
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if (self.patterns['bullet_points'].match(line) or 
                self.patterns['numbers_list'].match(line)):
                # Clean the bullet point
                cleaned = re.sub(r'^[\s]*[•·▪▫◦‣⁃]\s*', '', line)
                cleaned = re.sub(r'^[\s]*\d+[\.\)]\s*', '', cleaned)
                if cleaned and len(cleaned) > 5:
                    bullet_points.append(cleaned.strip())
        
        return bullet_points
    
    def normalize_spacing(self, text: str) -> str:
        """
        Normalize spacing and punctuation in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized spacing
        """
        if not text:
            return ""
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)  # Single space after punctuation
        
        # Fix quotation marks
        text = re.sub(r'\s*"\s*', '"', text)
        text = re.sub(r'\s*\'\s*', "'", text)
        
        # Fix parentheses
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        # Normalize multiple spaces
        text = self.patterns['whitespace'].sub(' ', text)
        
        return text.strip()
    
    def get_text_statistics(self, text: str) -> Dict[str, int]:
        """
        Calculate various statistics about the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                'characters': 0,
                'words': 0,
                'sentences': 0,
                'paragraphs': 0,
                'unique_words': 0
            }
        
        # Basic counts
        characters = len(text)
        words = len(text.split())
        sentences = len(self.segment_sentences(text))
        paragraphs = len(self.segment_paragraphs(text))
        
        # Unique words
        word_list = [word.lower().strip(string.punctuation) for word in text.split()]
        unique_words = len(set(word for word in word_list if word))
        
        return {
            'characters': characters,
            'words': words,
            'sentences': sentences,
            'paragraphs': paragraphs,
            'unique_words': unique_words,
            'avg_words_per_sentence': words / sentences if sentences > 0 else 0,
            'avg_chars_per_word': characters / words if words > 0 else 0,
            'lexical_diversity': unique_words / words if words > 0 else 0
        }
    
    def extract_key_phrases(self, text: str, max_phrases: int = 20) -> List[str]:
        """
        Extract key phrases using simple heuristics.
        
        Args:
            text: Input text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of key phrases
        """
        if not text:
            return []
        
        phrases = []
        
        # Extract phrases in quotes
        quoted_phrases = re.findall(r'"([^"]+)"', text)
        phrases.extend([phrase.strip() for phrase in quoted_phrases if len(phrase.strip()) > 5])
        
        # Extract capitalized phrases (potential proper nouns)
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        phrases.extend(capitalized_phrases)
        
        # Extract phrases with common academic patterns
        academic_patterns = [
            r'\b(?:according to|based on|in conclusion|furthermore|however|therefore)\b[^.!?]*',
            r'\b(?:research shows|studies indicate|evidence suggests)\b[^.!?]*',
            r'\b(?:it is important|significant|crucial|essential)\b[^.!?]*'
        ]
        
        for pattern in academic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend([match.strip() for match in matches])
        
        # Remove duplicates and sort by length
        unique_phrases = list(set(phrases))
        unique_phrases.sort(key=len, reverse=True)
        
        return unique_phrases[:max_phrases]
