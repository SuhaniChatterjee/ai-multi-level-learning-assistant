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

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="centered")

# Inject Custom CSS
CUSTOM_CSS = """
<style>
/* Global Font & Colors */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #1C2B4A !important;
}

/* Main Background */
.stApp {
    background-color: #EEF4FB;
}

/* Sidebar Background */
[data-testid="stSidebar"] {
    background-color: #D6E8F7;
    border-right: none;
}

/* Sidebar gradient banner */
.sidebar-banner {
    background: linear-gradient(135deg, #1A73E8, #4ca1af);
    color: white;
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    text-align: center;
    font-weight: 700;
    font-size: 1.2rem;
    box-shadow: 0 4px 15px rgba(26, 115, 232, 0.2);
}

/* Headings */
h1, h2, h3, h4 {
    color: #1A73E8 !important;
    font-weight: 700 !important;
}

.custom-heading {
    color: #1A73E8;
    font-weight: 700;
    margin-bottom: 15px;
    font-size: 1.15rem;
}

.secondary-text {
    color: #5A7A99;
}

/* Containers (Cards) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border-radius: 16px !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07) !important;
    padding: 1.5rem !important; /* Premium spacing */
}

/* Buttons */
.stButton > button {
    background-color: #1A73E8 !important;
    color: #FFFFFF !important;
    border-radius: 50px !important;
    border: none !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3) !important;
}
.stButton > button:hover {
    background-color: #155cbc !important;
    box-shadow: 0 6px 16px rgba(26, 115, 232, 0.4) !important;
}

/* Input field */
.stTextInput > div > div > input, .stSelectbox > div > div > div {
    border-radius: 12px !important;
    border: 1px solid #c2d6e8 !important;
    background-color: #FFFFFF !important;
    color: #1C2B4A !important;
}
.stTextInput > div > div > input:focus, .stSelectbox > div > div > div:focus {
    border-color: #1A73E8 !important;
}

/* Base text color explicitly forced for markdown */
.stMarkdown {
    color: #1C2B4A;
}

/* File Uploader styling */
[data-testid="stFileUploadDropzone"] {
    background-color: rgba(26, 115, 232, 0.04) !important;
    border-radius: 12px !important;
    border: 2px dashed #b3cde3 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Main App Header
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🌟 AI Learning Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5A7A99; font-size: 1.1rem; margin-bottom: 3rem;'>Upload your study material and get concepts explained seamlessly from ELI5 to Interview Ready.</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-banner'>AI Assistant 🎓</div>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='custom-heading'>1. Upload Material 📚</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        
        if uploaded_file is not None:
            if st.button("Process Document", use_container_width=True):
                with st.spinner("Extracting text and building knowledge base..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        text = extract_text_from_pdf(tmp_path)
                        chunks = chunk_text(text)
                        create_vector_store(chunks)
                        st.success("Document processed successfully! 🎉")
                    except Exception as e:
                        st.error(f"Error processing document: {e}")
                    finally:
                        os.unlink(tmp_path)
    
    st.markdown("<br/>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<div class='custom-heading'>2. Choose Level 🎚️</div>", unsafe_allow_html=True)
        difficulty = st.selectbox(
            "Explanation Difficulty:",
            ("ELI5", "College", "Interview"),
            label_visibility="collapsed",
            help="ELI5 = Simple analogies, College = Academic and structured, Interview = Crisp and technical"
        )

# Main Question Input
st.markdown("<div class='custom-heading' style='font-size: 1.5rem; text-align: center; margin-top: 2rem;'>3. Ask a Question ❓</div>", unsafe_allow_html=True)

question = st.text_input("Concept to explain:", label_visibility="collapsed", placeholder="E.g., What is a CNN?")
submitted = st.button("Get Explanation", use_container_width=True)

if submitted:
    if not question:
        st.warning("Please enter a question.")
    elif not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        st.error("Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    else:
        with st.spinner(f"Generating {difficulty}-level explanation..."):
            response, context = run_rag_pipeline(question, difficulty)
            
            if response.startswith("[Error]"):
                st.error(response)
            else:
                with st.container(border=True):
                    st.markdown("<h3 style='margin-top: 0;'>💡 Explanation</h3>", unsafe_allow_html=True)
                    st.markdown(response)
                
                with st.expander("🔍 Show Source Context Used"):
                    st.text(context)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #5A7A99; font-size: 0.9rem;'>Built with Streamlit, FAISS, and HuggingFace 🚀</div>", unsafe_allow_html=True)
