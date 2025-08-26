"""
Configuration Module
Contains configuration settings and constants for Smart Notes Generator.
"""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration settings for Smart Notes Generator."""
    
    # Application settings
    APP_NAME = "Smart Notes Generator"
    APP_VERSION = "1.0.0"
    
    # File settings
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_PDF_EXTENSIONS = ['.pdf']
    SUPPORTED_EXPORT_FORMATS = ['.docx', '.txt', '.md']
    
    # Text processing settings
    MIN_TEXT_LENGTH = 100
    MIN_SENTENCE_LENGTH = 10
    MAX_SENTENCE_LENGTH = 500
    
    # Summarization settings
    DEFAULT_SUMMARY_LENGTH = 200
    MIN_SUMMARY_LENGTH = 50
    MAX_SUMMARY_LENGTH = 1000
    
    # Keyword extraction settings
    DEFAULT_KEYWORD_COUNT = 15
    MIN_KEYWORD_COUNT = 5
    MAX_KEYWORD_COUNT = 50
    
    # Model settings
    SPACY_MODEL = "en_core_web_sm"
    DEFAULT_ABSTRACTIVE_MODEL = "facebook/bart-large-cnn"
    ALTERNATIVE_MODELS = [
        "google/pegasus-xsum",
        "t5-small",
        "t5-base"
    ]
    
    # Processing settings
    CHUNK_SIZE = 1024  # For transformer models
    MAX_PROCESSING_TIME = 300  # 5 minutes
    
    # UI settings
    STREAMLIT_CONFIG = {
        'page_title': APP_NAME,
        'page_icon': '📚',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    }
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File paths
    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory path."""
        return Path.home() / ".smart_notes_generator"
    
    @classmethod
    def get_temp_dir(cls) -> Path:
        """Get the temporary directory path."""
        return cls.get_data_dir() / "temp"
    
    @classmethod
    def get_exports_dir(cls) -> Path:
        """Get the exports directory path."""
        return cls.get_data_dir() / "exports"
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist."""
        dirs = [cls.get_data_dir(), cls.get_temp_dir(), cls.get_exports_dir()]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_type: str = "extractive") -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Args:
            model_type: "extractive" or "abstractive"
            
        Returns:
            Model configuration dictionary
        """
        if model_type == "extractive":
            return {
                'algorithms': ['textrank', 'lsa'],
                'default_algorithm': 'textrank',
                'sentence_count_ratio': 0.25,  # 25% of original sentences
                'min_sentences': 2,
                'max_sentences': 10
            }
        elif model_type == "abstractive":
            return {
                'model_name': cls.DEFAULT_ABSTRACTIVE_MODEL,
                'max_length': 142,  # BART limitation
                'min_length': 30,
                'temperature': 0.7,
                'do_sample': False,
                'chunk_overlap': 50  # Words overlap between chunks
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @classmethod
    def get_keyword_config(cls) -> Dict[str, Any]:
        """Get keyword extraction configuration."""
        return {
            'yake': {
                'lan': 'en',
                'n': 3,  # max words in keyphrase
                'dedupLim': 0.7,
                'features': None
            },
            'tfidf': {
                'max_features': cls.DEFAULT_KEYWORD_COUNT * 3,
                'stop_words': 'english',
                'ngram_range': (1, 2),
                'max_df': 0.8,
                'min_df': 1
            }
        }
    
    @classmethod
    def get_export_config(cls) -> Dict[str, Any]:
        """Get export configuration."""
        return {
            'docx': {
                'font_name': 'Calibri',
                'font_size': 11,
                'heading_size': 14,
                'line_spacing': 1.15
            },
            'txt': {
                'line_width': 80,
                'encoding': 'utf-8'
            },
            'md': {
                'encoding': 'utf-8',
                'use_tables': True
            }
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required directories
            cls.ensure_directories()
            
            # Validate numeric ranges
            assert cls.MIN_SUMMARY_LENGTH < cls.DEFAULT_SUMMARY_LENGTH < cls.MAX_SUMMARY_LENGTH
            assert cls.MIN_KEYWORD_COUNT < cls.DEFAULT_KEYWORD_COUNT < cls.MAX_KEYWORD_COUNT
            assert cls.MAX_FILE_SIZE_MB > 0
            
            # Validate file extensions
            assert all(ext.startswith('.') for ext in cls.SUPPORTED_PDF_EXTENSIONS)
            assert all(ext.startswith('.') for ext in cls.SUPPORTED_EXPORT_FORMATS)
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration."""
    LOG_LEVEL = "DEBUG"
    MAX_PROCESSING_TIME = 600  # 10 minutes for development
    
class ProductionConfig(Config):
    """Production environment configuration."""
    LOG_LEVEL = "WARNING"
    MAX_PROCESSING_TIME = 180  # 3 minutes for production

# Configuration factory
def get_config() -> Config:
    """
    Get the appropriate configuration based on environment.
    
    Returns:
        Configuration instance
    """
    env = os.getenv('SMART_NOTES_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()
