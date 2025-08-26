"""
Smart Notes Generator - Streamlit Frontend
Provides an interactive UI for PDF upload, processing, and note generation.
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import logging
from datetime import datetime

# Import our custom modules
from pdf_handler import PDFHandler
from summarizer import SmartSummarizer
from exporter import NotesExporter
from utils.config import Config
from utils.text_processing import TextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Smart Notes Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SmartNotesApp:
    """Main application class for Smart Notes Generator"""
    
    def __init__(self):
        self.config = Config()
        self.pdf_handler = PDFHandler()
        self.summarizer = SmartSummarizer()
        self.exporter = NotesExporter()
        self.text_processor = TextProcessor()
        
    def render_header(self):
        """Render the main header and description"""
        st.title("📚 Smart Notes Generator")
        st.markdown("""
        Transform your educational PDFs into structured, intelligent notes using advanced NLP techniques.
        Upload a PDF, configure your preferences, and generate comprehensive summaries with keyword extraction.
        """)
        st.divider()
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        st.sidebar.header("⚙️ Configuration")
        
        # Summarization mode selection
        mode = st.sidebar.selectbox(
            "📝 Summarization Mode",
            ["MVP (Extractive)", "Advanced (Abstractive)"],
            help="MVP mode is faster, Advanced mode provides higher quality summaries"
        )
        
        # Summary length
        summary_length = st.sidebar.slider(
            "📏 Summary Length",
            min_value=50,
            max_value=500,
            value=200,
            step=25,
            help="Number of words in the generated summary"
        )
        
        # Chapter segmentation
        chapter_segmentation = st.sidebar.checkbox(
            "📖 Chapter-wise Segmentation",
            value=True,
            help="Automatically detect and segment chapters"
        )
        
        # Keyword extraction
        keyword_count = st.sidebar.slider(
            "🔑 Keywords to Extract",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="Number of keywords to extract from the document"
        )
        
        # Export format
        export_format = st.sidebar.selectbox(
            "💾 Export Format",
            [".docx", ".txt"],
            help="Choose the format for exported notes"
        )
        
        st.sidebar.divider()
        
        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            min_sentence_length = st.slider(
                "Minimum Sentence Length",
                min_value=10,
                max_value=100,
                value=30
            )
            
            similarity_threshold = st.slider(
                "Similarity Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1
            )
        
        return {
            'mode': mode,
            'summary_length': summary_length,
            'chapter_segmentation': chapter_segmentation,
            'keyword_count': keyword_count,
            'export_format': export_format,
            'min_sentence_length': min_sentence_length,
            'similarity_threshold': similarity_threshold
        }
    
    def process_pdf(self, uploaded_file, settings):
        """Process the uploaded PDF and generate notes"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Extract text from PDF
            with st.spinner("📖 Extracting text from PDF..."):
                extracted_data = self.pdf_handler.extract_text(tmp_path)
                
            if not extracted_data['text'].strip():
                st.error("❌ No text found in the PDF. The file might be image-based or corrupted.")
                return None
            
            # Process text
            with st.spinner("🔧 Processing text..."):
                processed_text = self.text_processor.clean_text(extracted_data['text'])
                
            # Generate summary
            summarization_mode = "extractive" if "MVP" in settings['mode'] else "abstractive"
            
            with st.spinner(f"🤖 Generating {summarization_mode} summary..."):
                summary_result = self.summarizer.generate_summary(
                    processed_text,
                    mode=summarization_mode,
                    max_length=settings['summary_length'],
                    min_sentence_length=settings['min_sentence_length']
                )
            
            # Extract keywords
            with st.spinner("🔑 Extracting keywords..."):
                keywords = self.summarizer.extract_keywords(
                    processed_text,
                    num_keywords=settings['keyword_count']
                )
            
            # Chapter segmentation if enabled
            chapters = None
            if settings['chapter_segmentation']:
                with st.spinner("📖 Detecting chapters..."):
                    chapters = self.text_processor.segment_chapters(processed_text)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return {
                'summary': summary_result,
                'keywords': keywords,
                'chapters': chapters,
                'metadata': extracted_data['metadata'],
                'original_text': processed_text
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            st.error(f"❌ Error processing PDF: {str(e)}")
            return None
    
    def render_results(self, results, settings):
        """Render the processing results"""
        if not results:
            return
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🔑 Keywords", "📖 Chapters", "📊 Metadata"])
        
        with tab1:
            st.subheader("Generated Summary")
            st.write(results['summary'])
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Summary Length", f"{len(results['summary'].split())} words")
            with col2:
                st.metric("Compression Ratio", f"{len(results['summary'].split()) / len(results['original_text'].split()):.2%}")
            with col3:
                st.metric("Mode", settings['mode'])
        
        with tab2:
            st.subheader("Extracted Keywords")
            
            # Display keywords as badges
            keyword_html = ""
            for keyword in results['keywords']:
                keyword_html += f'<span style="background-color: #407a94; padding: 4px 8px; margin: 2px; border-radius: 12px; display: inline-block;">{keyword}</span> '
            
            st.markdown(keyword_html, unsafe_allow_html=True)
            
            # Keywords as list
            st.subheader("Keywords List")
            for i, keyword in enumerate(results['keywords'], 1):
                st.write(f"{i}. {keyword}")
        
        with tab3:
            if results['chapters']:
                st.subheader("Chapter Structure")
                for i, chapter in enumerate(results['chapters'], 1):
                    with st.expander(f"Chapter {i}: {chapter.get('title', 'Untitled')}"):
                        st.write(chapter.get('content', ''))
            else:
                st.info("Chapter segmentation was not enabled or no clear chapter structure was detected.")
        
        with tab4:
            st.subheader("Document Metadata")
            metadata = results['metadata']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Pages:** {metadata.get('pages', 'Unknown')}")
                st.write(f"**Title:** {metadata.get('title', 'Unknown')}")
                st.write(f"**Author:** {metadata.get('author', 'Unknown')}")
            
            with col2:
                st.write(f"**Subject:** {metadata.get('subject', 'Unknown')}")
                st.write(f"**Creator:** {metadata.get('creator', 'Unknown')}")
                st.write(f"**Creation Date:** {metadata.get('creation_date', 'Unknown')}")
        
        # Export functionality
        st.divider()
        self.render_export_section(results, settings)
    
    def render_export_section(self, results, settings):
        """Render the export section"""
        st.subheader("💾 Export Notes")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            filename = st.text_input(
                "Filename",
                value=f"smart_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Enter filename without extension"
            )
        
        with col2:
            include_keywords = st.checkbox("Include Keywords", value=True)
        
        with col3:
            include_chapters = st.checkbox("Include Chapters", value=bool(results['chapters']))
        
        if st.button("📥 Generate Export File", type="primary"):
            try:
                # Prepare export data
                export_data = {
                    'summary': results['summary'],
                    'keywords': results['keywords'] if include_keywords else [],
                    'chapters': results['chapters'] if include_chapters and results['chapters'] else [],
                    'metadata': results['metadata']
                }
                
                # Generate export file
                with st.spinner("📦 Generating export file..."):
                    file_content = self.exporter.export_notes(
                        export_data,
                        format_type=settings['export_format'],
                        filename=filename
                    )
                
                # Provide download button
                st.download_button(
                    label=f"⬇️ Download {settings['export_format'].upper()} File",
                    data=file_content,
                    file_name=f"{filename}{settings['export_format']}",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document" if settings['export_format'] == '.docx' else "text/plain"
                )
                
                st.success("✅ Export file generated successfully!")
                
            except Exception as e:
                logger.error(f"Export error: {str(e)}")
                st.error(f"❌ Export failed: {str(e)}")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Get configuration from sidebar
        settings = self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📁 Upload PDF Document")
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload an educational PDF document to generate structured notes"
            )
        
        with col2:
            if uploaded_file is not None:
                st.subheader("📋 File Info")
                st.write(f"**Name:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                st.write(f"**Type:** {uploaded_file.type}")
        
        # Process button and results
        if uploaded_file is not None:
            if st.button("🚀 Generate Smart Notes", type="primary", use_container_width=True):
                results = self.process_pdf(uploaded_file, settings)
                if results:
                    st.session_state['results'] = results
                    st.session_state['settings'] = settings
        
        # Display results if available
        if 'results' in st.session_state:
            st.divider()
            st.header("📋 Generated Notes")
            self.render_results(st.session_state['results'], st.session_state['settings'])

def main():
    """Main application entry point"""
    app = SmartNotesApp()
    app.run()

if __name__ == "__main__":
    main()
