"""
Main Streamlit application for the AI Multi-Level Learning Assistant.
Handles PDF ingestion, user queries, and displays dynamic explanations.
"""
import streamlit as st
import tempfile
import os
import sys

# Ensure backend can be imported safely from the top-level
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.document_processor import extract_text_from_pdf, chunk_text
from backend.vector_store import create_vector_store
from backend.rag_pipeline import run_rag_pipeline

st.set_page_config(page_title="Cognitive Assistant", layout="centered")

# Inject Custom Modern 3D/Animated CSS
CUSTOM_CSS = """
<style>
/* Global Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: #1E293B !important;
}

/* Animated Gradient Background for the App */
.stApp {
    background: linear-gradient(-45deg, #F8FAFC, #E2E8F0, #F1F5F9, #E0E7FF);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255,255,255,0.6) !important;
}

/* Advanced 3D Glassmorphism Cards for Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.7) !important;
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.04), 
        inset 0 1px 0 rgba(255, 255, 255, 1) !important;
    padding: 1.8rem !important;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px);
    box-shadow: 
        0 30px 60px rgba(0, 0, 0, 0.08), 
        inset 0 1px 0 rgba(255, 255, 255, 1) !important;
}

/* Typography Enhancements */
.hero-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0F172A 0%, #2563EB 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
    animation: fadeInDown 0.8s ease-out;
}

.hero-subtitle {
    text-align: center;
    color: #475569;
    font-size: 1.15rem;
    font-weight: 400;
    margin-bottom: 3rem;
    animation: fadeInDown 1s ease-out;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-heading {
    color: #0F172A;
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 12px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* 3D Animated Button */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 
        0 4px 6px rgba(37, 99, 235, 0.2), 
        0 1px 3px rgba(0, 0, 0, 0.1), 
        inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
}

.stButton > button::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0) 100%);
    transform: skewX(-20deg);
    animation: shine 4s infinite;
}

@keyframes shine {
    0% { left: -100%; }
    20% { left: 200%; }
    100% { left: 200%; }
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 
        0 12px 20px rgba(37, 99, 235, 0.3), 
        0 4px 6px rgba(0, 0, 0, 0.1), 
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
}

.stButton > button:active {
    transform: translateY(1px);
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.3) !important;
}

/* Inputs and Selects */
.stTextInput > div > div > input, .stSelectbox > div > div > div {
    border-radius: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    background: rgba(255, 255, 255, 0.8) !important;
    color: #1E293B !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.02) !important;
    transition: all 0.2s ease !important;
}

.stTextInput > div > div > input:focus, .stSelectbox > div > div > div:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2), inset 0 2px 4px rgba(0,0,0,0.01) !important;
    background: #FFFFFF !important;
}

/* File Dropzone */
[data-testid="stFileUploadDropzone"] {
    background: rgba(248, 250, 252, 0.5) !important;
    border-radius: 12px !important;
    border: 2px dashed rgba(148, 163, 184, 0.5) !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #3B82F6 !important;
    background: rgba(239, 246, 255, 0.8) !important;
}

/* Spinner Animation Override */
.stSpinner > div > div {
    border-color: #3B82F6 transparent transparent transparent !important;
}

/* Minimal Footer */
.footer-text {
    text-align: center; 
    color: #94A3B8; 
    font-size: 0.85rem; 
    margin-top: 4rem;
    font-weight: 500;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Main App Header
st.markdown("<div class='hero-title'>Cognitive Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Upload your documents and receive context-aware explanations tailored to your expertise level.</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div style='font-size: 1.1rem; font-weight: 800; color: #0F172A; margin-bottom: 2rem; text-align: center; letter-spacing: 0.1em; text-transform: uppercase;'>Configuration</div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='section-heading'>Document Ingestion</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("Process Document", use_container_width=True):
                with st.spinner("Analyzing document structure..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        text = extract_text_from_pdf(tmp_path)
                        chunks = chunk_text(text)
                        create_vector_store(chunks)
                        st.success("Document analyzed successfully.")
                    except Exception as e:
                        st.error(f"Error analyzing document: {e}")
                    finally:
                        os.unlink(tmp_path)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='section-heading'>Complexity Level</div>", unsafe_allow_html=True)
        difficulty = st.selectbox(
            "Explanation Difficulty:",
            ("ELI5", "College", "Interview"),
            label_visibility="collapsed"
        )

# Main Question Input
st.markdown("<div class='section-heading' style='text-align: center; margin-top: 1rem; color: #64748B;'>Query Engine</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    question = st.text_input("Concept to explain:", label_visibility="collapsed", placeholder="Enter the technical concept or question...")
    
    # Add a slight top padding to the button
    st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
    submitted = st.button("Synthesize Explanation", use_container_width=True)

if submitted:
    if not question:
        st.warning("Please provide a concept to explain.")
    elif not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        st.error("Authentication required: Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    else:
        with st.spinner("Synthesizing response..."):
            response, context = run_rag_pipeline(question, difficulty)
            
            if response.startswith("[Error]"):
                st.error(response)
            else:
                with st.container(border=True):
                    st.markdown("<div class='section-heading' style='color: #2563EB; margin-bottom: 1rem;'>AI Analysis</div>", unsafe_allow_html=True)
                    st.markdown(response)
                
                with st.expander("View Retrieved Context"):
                    st.text(context)

st.markdown("---")
st.markdown("<div class='footer-text'>Engineered with FAISS, Streamlit, and Qwen Models</div>", unsafe_allow_html=True)
