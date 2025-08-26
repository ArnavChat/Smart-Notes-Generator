# Smart Notes Generator

🚀 **Automatically generates structured notes from educational PDFs using advanced NLP techniques.**

Transform your educational PDFs into intelligent, structured notes with extractive and abstractive summarization, keyword extraction, and chapter segmentation capabilities.

## ✨ Features

- **🖥️ Interactive Streamlit Web Interface** - User-friendly UI for PDF upload and processing
- **🧠 Dual Summarization Modes**:
  - **Extractive**: TextRank and LSA algorithms for fast summarization
  - **Abstractive**: Transformer models (T5, BART) for human-like summaries
- **🔑 Advanced Keyword Extraction** - YAKE algorithm with TF-IDF fallback
- **📖 Intelligent Chapter Segmentation** - Automatic detection and organization
- **💾 Multi-Format Export** - Export to `.docx`, `.txt`, and `.md` formats
- **🧪 Comprehensive Testing** - Full test suite for reliability
- **🔧 Modular Architecture** - Easy to extend and maintain

## 🏗️ Project Architecture

```
smart_notes_generator/
├── 🎯 app.py                    # Streamlit web application (main entry point)
├── 🧠 summarizer.py             # NLP summarization engine
├── 📄 pdf_handler.py            # PDF text extraction & processing
├── 💾 exporter.py               # Multi-format export functionality
├── 📝 example_usage.py          # Usage examples and demos
├── 📋 requirements.txt          # Python dependencies
├── 🚫 .gitignore               # Git ignore rules
├── 🔧 utils/                   # Utility modules
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   └── text_processing.py     # Text cleaning & preprocessing
└── 🧪 tests/                   # Comprehensive test suite
    ├── __init__.py
    ├── test_pdf_handler.py    # PDF processing tests
    ├── test_summarizer.py     # Summarization tests
    └── test_exporter.py       # Export functionality tests

```

## 🛠️ Technologies & Libraries

### **Core Framework**

- **[Streamlit](https://streamlit.io/)** `>=1.38` - Web application framework
- **[Python](https://python.org/)** `3.10+` - Programming language

### **PDF Processing**

- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)** `>=1.24` - Primary PDF text extraction
- **[PyPDF2](https://pypdf2.readthedocs.io/)** - Backup PDF processing
- **[pdfplumber](https://github.com/jsvine/pdfplumber)** `>=0.11` - Advanced PDF analysis

### **Natural Language Processing**

- **[spaCy](https://spacy.io/)** `>=3.7` - Industrial-strength NLP library
- **[Transformers (Hugging Face)](https://huggingface.co/transformers/)** `>=4.43` - State-of-the-art NLP models
- **[YAKE](https://github.com/LIAAD/yake)** `>=0.4.8` - Keyword extraction algorithm
- **[Sumy](https://github.com/miso-belica/sumy)** `>=0.11` - Automatic text summarization

### **Machine Learning & Data Processing**

- **[PyTorch](https://pytorch.org/)** `>=2.3` - Deep learning framework
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning utilities
- **[pandas](https://pandas.pydata.org/)** `>=2.2` - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** `>=1.26` - Numerical computing

### **Document Generation**

- **[python-docx](https://python-docx.readthedocs.io/)** `>=1.1` - Microsoft Word document generation

### **Additional Dependencies**

- **[sentencepiece](https://github.com/google/sentencepiece)** `>=0.2` - Text tokenization
- **[regex](https://pypi.org/project/regex/)** `>=2024.5` - Advanced regular expressions
- **[requests](https://requests.readthedocs.io/)** `>=2.32` - HTTP library

## 📋 System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4GB RAM minimum (8GB recommended for large PDFs)
- **Storage**: 2GB free space (for models and dependencies)
- **Network**: Internet connection for initial model downloads

## 🚀 Installation & Setup

### **Step 1: Clone the Repository**

```bash
git clone <repository-url>
cd smart_notes_generator
```

### **Step 2: Create Virtual Environment (Recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

### **Step 4: Download Required Models**

```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK data (automatically handled on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### **Step 5: Verify Installation**

```bash
python example_usage.py
```

## 🎯 How to Run

### **Option 1: Web Interface (Recommended)**

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

### **Option 2: Programmatic Usage**

```python
from pdf_handler import PDFHandler
from summarizer import SmartSummarizer
from exporter import NotesExporter

# Initialize components
pdf_handler = PDFHandler()
summarizer = SmartSummarizer()
exporter = NotesExporter()

# Process PDF
result = pdf_handler.extract_text("your_document.pdf")
summary = summarizer.generate_summary(result['text'], mode="extractive")
keywords = summarizer.extract_keywords(result['text'])

# Export notes
notes_data = {
    'summary': summary,
    'keywords': keywords,
    'metadata': result['metadata']
}
exported_content = exporter.export_notes(notes_data, format_type='.docx')
```

### **Option 3: Example Demo**

```bash
python example_usage.py [optional_pdf_path]
```

## 🔄 Workflow

### **1. PDF Upload & Validation**

- Upload PDF through web interface or specify file path
- Validate file format and integrity
- Extract metadata (title, author, pages, etc.)

### **2. Text Extraction & Preprocessing**

- Extract raw text using PyMuPDF
- Clean and preprocess text (remove artifacts, fix formatting)
- Segment into sentences and paragraphs

### **3. Content Analysis**

- **Chapter Detection**: Identify chapter boundaries and titles
- **Text Statistics**: Calculate word count, reading time, etc.
- **Structure Analysis**: Identify bullet points, lists, and sections

### **4. Summarization**

- **Extractive Mode**:
  - Use TextRank or LSA algorithms
  - Select most important sentences
  - Fast processing, no GPU required
- **Abstractive Mode**:
  - Use transformer models (BART/T5)
  - Generate new sentences
  - Higher quality, requires more resources

### **5. Keyword Extraction**

- Apply YAKE algorithm for key phrase extraction
- Fallback to TF-IDF if needed
- Filter and rank keywords by relevance

### **6. Export & Download**

- Generate structured notes in chosen format
- Include summary, keywords, and metadata
- Provide download link for immediate access

## ⚙️ Configuration Options

### **Summarization Settings**

- **Mode**: Extractive (fast) or Abstractive (high-quality)
- **Length**: 50-500 words
- **Algorithm**: TextRank or LSA (extractive mode)
- **Model**: BART or T5 (abstractive mode)

### **Keyword Extraction**

- **Count**: 5-50 keywords
- **Method**: YAKE or TF-IDF
- **Phrase Length**: 1-3 words per phrase

### **Export Options**

- **Format**: DOCX, TXT, or Markdown
- **Include Chapters**: Yes/No
- **Include Keywords**: Yes/No
- **Custom Filename**: User-defined names

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_pdf_handler.py -v
python -m pytest tests/test_summarizer.py -v
python -m pytest tests/test_exporter.py -v

# Run with coverage report
pip install pytest-cov
python -m pytest tests/ --cov=. --cov-report=html
```

## 🔧 Development

### **Adding New Features**

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes in appropriate modules
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

### **Extending Summarization**

- Add new algorithms in `summarizer.py`
- Implement in `_generate_extractive_summary()` or `_generate_abstractive_summary()`
- Update configuration options in `utils/config.py`

### **Adding Export Formats**

- Extend `exporter.py` with new format methods
- Update `supported_formats` list
- Add corresponding tests

## 🚨 Troubleshooting

### **Common Issues**

**1. spaCy Model Not Found**

```bash
python -m spacy download en_core_web_sm
```

**2. NLTK Data Missing**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**3. PyTorch Installation Issues**
Visit [PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform-specific instructions.

**4. Memory Issues with Large PDFs**

- Reduce summary length
- Use extractive mode instead of abstractive
- Process PDFs in smaller chunks

### **Performance Optimization**

- Use extractive mode for faster processing
- Enable GPU acceleration for abstractive summarization
- Adjust chunk sizes for large documents

## 🔮 Future Enhancements

- **📝 Notion API Integration** - Direct export to Notion databases
- **🌍 Multi-language Support** - Process PDFs in multiple languages
- **🎯 Custom Model Fine-tuning** - Domain-specific summarization
- **📊 Batch Processing** - Handle multiple PDFs simultaneously
- **🔌 REST API** - Headless operation for integration
- **📱 Mobile App** - React Native mobile application

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and feature requests on GitHub Issues
- **Documentation**: Check the `/docs` folder for detailed guides
- **Examples**: See `example_usage.py` for usage examples

---

**Made with ❤️ by the Smart Notes Generator Team**
