"""
Smart Summarizer Module
Implements both extractive and abstractive summarization with keyword extraction.
"""

import logging
import re
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

# NLP libraries
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yake

# Transformers for abstractive summarization
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Text ranking algorithms
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

logger = logging.getLogger(__name__)

class SmartSummarizer:
    """
    Advanced summarization system supporting both extractive and abstractive methods.
    Includes keyword extraction and multiple algorithm support.
    """
    
    def __init__(self):
        """Initialize the summarizer with models and configurations."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize abstractive models (lazy loading)
        self._abstractive_model = None
        self._abstractive_tokenizer = None
        
        # YAKE keyword extractor
        self.keyword_extractor = None
        
        # Sumy summarizers
        self.textrank_summarizer = TextRankSummarizer()
        self.lsa_summarizer = LsaSummarizer()
        
    def _load_abstractive_model(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Lazy load the abstractive summarization model.
        
        Args:
            model_name: HuggingFace model name for abstractive summarization
        """
        if self._abstractive_model is None:
            try:
                logger.info(f"Loading abstractive model: {model_name}")
                self._abstractive_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._abstractive_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                # Move to appropriate device
                if self.device == "cuda":
                    self._abstractive_model = self._abstractive_model.to(self.device)
                    
                logger.info("Abstractive model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load abstractive model: {str(e)}")
                raise Exception(f"Could not load abstractive model: {str(e)}")
    
    def generate_summary(self, text: str, mode: str = "extractive", 
                        max_length: int = 200, min_sentence_length: int = 30,
                        algorithm: str = "textrank") -> str:
        """
        Generate a summary using the specified mode and algorithm.
        
        Args:
            text: Input text to summarize
            mode: "extractive" or "abstractive"
            max_length: Maximum length of summary in words
            min_sentence_length: Minimum length for sentences to be considered
            algorithm: For extractive: "textrank" or "lsa"
            
        Returns:
            Generated summary text
        """
        if not text or len(text.strip()) < 100:
            return "Text too short for meaningful summarization."
        
        try:
            if mode.lower() == "extractive":
                return self._generate_extractive_summary(text, max_length, min_sentence_length, algorithm)
            elif mode.lower() == "abstractive":
                return self._generate_abstractive_summary(text, max_length)
            else:
                raise ValueError(f"Unsupported summarization mode: {mode}")
                
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _generate_extractive_summary(self, text: str, max_length: int, 
                                   min_sentence_length: int, algorithm: str) -> str:
        """
        Generate extractive summary using TextRank or LSA.
        
        Args:
            text: Input text
            max_length: Maximum summary length in words
            min_sentence_length: Minimum sentence length
            algorithm: "textrank" or "lsa"
            
        Returns:
            Extractive summary
        """
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Use Sumy for extractive summarization
        parser = PlaintextParser.from_string(cleaned_text, Tokenizer("english"))
        
        # Calculate number of sentences for summary
        total_sentences = len(list(parser.document.sentences))
        sentence_count = max(1, min(total_sentences // 4, max_length // 20))
        
        try:
            if algorithm.lower() == "textrank":
                summarizer = self.textrank_summarizer
            elif algorithm.lower() == "lsa":
                summarizer = self.lsa_summarizer
            else:
                logger.warning(f"Unknown algorithm {algorithm}, defaulting to TextRank")
                summarizer = self.textrank_summarizer
            
            # Generate summary
            summary_sentences = summarizer(parser.document, sentence_count)
            summary = " ".join([str(sentence) for sentence in summary_sentences])
            
            # Post-process and truncate if needed
            summary = self._post_process_summary(summary, max_length)
            
            return summary
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {str(e)}")
            # Fallback to simple approach
            return self._fallback_extractive_summary(text, max_length)
    
    def _generate_abstractive_summary(self, text: str, max_length: int) -> str:
        """
        Generate abstractive summary using transformer models.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Abstractive summary
        """
        # Load model if not already loaded
        self._load_abstractive_model()
        
        try:
            # Create summarization pipeline
            summarizer = pipeline(
                "summarization",
                model=self._abstractive_model,
                tokenizer=self._abstractive_tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            # Chunk text if it's too long for the model
            max_chunk_length = 1024  # BART's max input length
            chunks = self._chunk_text(text, max_chunk_length)
            
            summaries = []
            for chunk in chunks:
                if len(chunk.split()) < 50:  # Skip very short chunks
                    continue
                    
                # Generate summary for chunk
                chunk_summary = summarizer(
                    chunk,
                    max_length=min(max_length // len(chunks), 142),  # BART max output
                    min_length=30,
                    do_sample=False,
                    temperature=0.7
                )[0]['summary_text']
                
                summaries.append(chunk_summary)
            
            # Combine summaries
            combined_summary = " ".join(summaries)
            
            # Final post-processing
            final_summary = self._post_process_summary(combined_summary, max_length)
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Abstractive summarization failed: {str(e)}")
            # Fallback to extractive
            logger.info("Falling back to extractive summarization")
            return self._generate_extractive_summary(text, max_length, 30, "textrank")
    
    def extract_keywords(self, text: str, num_keywords: int = 15, 
                        method: str = "yake") -> List[str]:
        """
        Extract keywords from text using YAKE or TF-IDF.
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            method: "yake" or "tfidf"
            
        Returns:
            List of extracted keywords
        """
        if not text or len(text.strip()) < 50:
            return []
        
        try:
            if method.lower() == "yake":
                return self._extract_yake_keywords(text, num_keywords)
            elif method.lower() == "tfidf":
                return self._extract_tfidf_keywords(text, num_keywords)
            else:
                logger.warning(f"Unknown method {method}, defaulting to YAKE")
                return self._extract_yake_keywords(text, num_keywords)
                
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []
    
    def _extract_yake_keywords(self, text: str, num_keywords: int) -> List[str]:
        """Extract keywords using YAKE algorithm."""
        # Configure YAKE
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # Maximum number of words in keyphrase
            dedupLim=0.7,  # Deduplication threshold
            top=num_keywords,
            features=None
        )
        
        # Extract keywords
        keywords = kw_extractor.extract_keywords(text)
        
        # Return only the keyword strings (keywords are first element in tuple)
        return [str(kw[0]) for kw in keywords]
    
    def _extract_tfidf_keywords(self, text: str, num_keywords: int) -> List[str]:
        """Extract keywords using TF-IDF."""
        # Split text into sentences for TF-IDF
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < 2:
            return []
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=num_keywords * 3,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=1
        )
        
        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Get top keywords
        top_indices = mean_scores.argsort()[-num_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        
        return keywords
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 10]
        text = '\n'.join(lines)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\n+', ' ', text)  # Replace newlines with spaces
        
        return text.strip()
    
    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """
        Split text into chunks for processing by transformer models.
        
        Args:
            text: Input text
            max_length: Maximum length per chunk in characters
            
        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _post_process_summary(self, summary: str, max_length: int) -> str:
        """Post-process summary to ensure quality and length constraints."""
        # Remove excessive whitespace
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        # Ensure it ends with proper punctuation
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        # Truncate if too long
        words = summary.split()
        if len(words) > max_length:
            summary = ' '.join(words[:max_length])
            # Try to end at sentence boundary
            last_period = summary.rfind('.')
            if last_period > len(summary) * 0.8:  # If period is near the end
                summary = summary[:last_period + 1]
            else:
                summary += '...'
        
        return summary
    
    def _fallback_extractive_summary(self, text: str, max_length: int) -> str:
        """
        Simple fallback extractive summarization using sentence scoring.
        
        Args:
            text: Input text
            max_length: Maximum summary length
            
        Returns:
            Simple extractive summary
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= 3:
            return ' '.join(sentences)
        
        # Simple scoring based on sentence length and position
        scores = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())  # Word count
            # Boost for early sentences (introduction) and late sentences (conclusion)
            if i < len(sentences) * 0.2:  # First 20%
                score *= 1.2
            elif i > len(sentences) * 0.8:  # Last 20%
                score *= 1.1
            scores.append((score, i, sentence))
        
        # Sort by score and select top sentences
        scores.sort(reverse=True)
        summary_sentences = []
        word_count = 0
        
        for score, i, sentence in scores:
            if word_count + len(sentence.split()) <= max_length:
                summary_sentences.append((i, sentence))
                word_count += len(sentence.split())
            else:
                break
        
        # Sort selected sentences by original order
        summary_sentences.sort(key=lambda x: x[0])
        summary = '. '.join([sentence for i, sentence in summary_sentences])
        
        return summary + '.' if summary and summary[-1] != '.' else summary
