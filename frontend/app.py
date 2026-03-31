"""
Main Streamlit application for the AI Multi-Level Learning Assistant.
Handles PDF ingestion, user queries, and displays dynamic explanations.
"""
import streamlit as st
import tempfile
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Ensure backend can be imported safely from the top-level
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.document_processor import extract_text_from_pdf, chunk_text
from backend.vector_store import create_vector_store
from backend.rag_pipeline import run_rag_pipeline

st.set_page_config(page_title="Cognitive Assistant", layout="centered")

# Inject Custom Sky & Clouds 3D/Animated CSS
CUSTOM_CSS = """
<style>
/* Global Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1E293B !important;
}

/* Light Blue Sky Background */
.stApp {
    background: linear-gradient(180deg, #90CDF4 0%, #E2E8F0 100%) !important;
    background-attachment: fixed !important;
}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: transparent !important;
}

/* Floating Clouds System */
.clouds-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
}

.cloud {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 200px;
    position: absolute;
    filter: blur(3px); /* Soft fluffiness */
    box-shadow: inset 0 -5px 15px rgba(0,0,0,0.03);
}
.cloud::before, .cloud::after {
    content: '';
    position: absolute;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 50%;
}

.cloud1 {
    width: 320px; height: 110px;
    top: 15%;
    opacity: 0.8;
    animation: float1 45s linear infinite;
    transform: scale(1.1);
}
.cloud1::before { width: 140px; height: 140px; top: -60px; left: 40px; }
.cloud1::after { width: 100px; height: 100px; top: -40px; right: 40px; }

.cloud2 {
    width: 250px; height: 80px;
    top: 40%;
    opacity: 0.5;
    animation: float2 60s linear infinite;
    transform: scale(0.85);
}
.cloud2::before { width: 110px; height: 110px; top: -50px; left: 30px; }
.cloud2::after { width: 80px; height: 80px; top: -30px; right: 30px; }

.cloud3 {
    width: 380px; height: 130px;
    top: 70%;
    opacity: 0.7;
    animation: float1 55s linear infinite;
    transform: scale(1.3);
    animation-delay: -20s;
}
.cloud3::before { width: 180px; height: 180px; top: -80px; left: 60px; }
.cloud3::after { width: 120px; height: 120px; top: -50px; right: 60px; }

.cloud4 {
    width: 200px; height: 70px;
    top: 25%;
    opacity: 0.4;
    animation: float2 35s linear infinite;
    transform: scale(0.6);
    animation-delay: -10s;
}
.cloud4::before { width: 90px; height: 90px; top: -40px; left: 20px; }
.cloud4::after { width: 70px; height: 70px; top: -30px; right: 20px; }

.cloud5 {
    width: 280px; height: 90px;
    top: 85%;
    opacity: 0.6;
    animation: float1 70s linear infinite;
    transform: scale(0.9);
    animation-delay: -35s;
}
.cloud5::before { width: 120px; height: 120px; top: -50px; left: 40px; }
.cloud5::after { width: 80px; height: 80px; top: -30px; right: 30px; }

@keyframes float1 {
    0% { left: -500px; transform: scale(1.1) translateY(0px); }
    50% { transform: scale(1.1) translateY(10px); }
    100% { left: 110vw; transform: scale(1.1) translateY(0px); }
}
@keyframes float2 {
    0% { left: 110vw; transform: scale(0.85) translateY(0px); }
    50% { transform: scale(0.85) translateY(-8px); }
    100% { left: -500px; transform: scale(0.85) translateY(0px); }
}

/* Elevate Main Content to sit above clouds */
.stApp > header { z-index: 50 !important; }
.main { z-index: 10 !important; position: relative; }

/* Sidebar Styling */
[data-testid="stSidebar"] > div:first-child {
    background: rgba(255, 255, 255, 0.45) !important;
    backdrop-filter: blur(30px) !important;
    -webkit-backdrop-filter: blur(30px) !important;
    border-right: 1px solid rgba(255,255,255,0.8) !important;
    z-index: 100 !important;
}
[data-testid="stSidebar"] {
    background: transparent !important;
}

/* Advanced 3D Glassmorphism Cards for Containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.9) !important;
    box-shadow: 
        0 20px 40px rgba(0, 0, 0, 0.05), 
        inset 0 1px 0 rgba(255, 255, 255, 1) !important;
    padding: 1.8rem !important;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
    z-index: 20;
    position: relative;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-5px);
    box-shadow: 
        0 30px 60px rgba(0, 50, 150, 0.1), 
        inset 0 1px 0 rgba(255, 255, 255, 1) !important;
}

/* Typography Enhancements */
.hero-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
    animation: fadeInDown 0.8s ease-out;
    position: relative;
    z-index: 20;
}

.hero-subtitle {
    text-align: center;
    color: #334155;
    font-size: 1.15rem;
    font-weight: 500;
    margin-bottom: 3rem;
    animation: fadeInDown 1s ease-out;
    position: relative;
    z-index: 20;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-heading {
    color: #1E293B;
    font-weight: 800;
    font-size: 1.05rem;
    margin-bottom: 12px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* 3D Animated Button */
.stButton > button {
    background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important;
    color: #FFFFFF !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 12px 24px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    box-shadow: 
        0 6px 12px rgba(37, 99, 235, 0.25), 
        0 1px 3px rgba(0, 0, 0, 0.1), 
        inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
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
    background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.4) 50%, rgba(255,255,255,0) 100%);
    transform: skewX(-20deg);
    animation: shine 4s infinite;
}

@keyframes shine {
    0% { left: -100%; }
    20% { left: 200%; }
    100% { left: 200%; }
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 
        0 15px 25px rgba(37, 99, 235, 0.35), 
        0 5px 10px rgba(0, 0, 0, 0.1), 
        inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(1px);
    box-shadow: 0 2px 4px rgba(37, 99, 235, 0.3) !important;
}

/* Premium File Dropzone */
[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.85) !important;
    border-radius: 16px !important;
    border: 2px dashed rgba(59, 130, 246, 0.5) !important;
    transition: all 0.3s ease;
    padding: 2rem !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #2563EB !important;
    background: rgba(239, 246, 255, 0.95) !important;
}

/* Force Browse Files Button to Match Theme */
[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important;
    color: #FFFFFF !important;
    border-radius: 50px !important;
    border: none !important;
    padding: 8px 24px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploadDropzone"] button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 16px rgba(37, 99, 235, 0.4) !important;
}

/* Inputs and Selects */
.stTextInput > div > div > input, .stSelectbox > div > div > div {
    border-radius: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.4) !important;
    background: rgba(255, 255, 255, 0.9) !important;
    color: #1E293B !important;
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.02) !important;
    transition: all 0.2s ease !important;
}

/* Ensure Placeholder Text is Visible */
.stTextInput > div > div > input::placeholder, 
textarea::placeholder {
    color: #94A3B8 !important;
    opacity: 1 !important;
}

.stTextInput > div > div > input:focus, .stSelectbox > div > div > div:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25), inset 0 2px 4px rgba(0,0,0,0.01) !important;
    background: #FFFFFF !important;
}

/* STRONGLY FORCE TEXT COLORS FOR DARK MODE COMPATIBILITY */
[data-testid="stMarkdownContainer"] *,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stFileUploadDropzone"] *, 
[data-testid="stExpander"] * {
    color: #1E293B !important;
}

/* Fix CSS for code blocks to be readable on dark backgrounds */
[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] pre * {
    color: #F8FAFC !important;
}
[data-testid="stMarkdownContainer"] code {
    color: #E2E8F0 !important;
}

[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploadDropzone"] button * {
    color: #FFFFFF !important;
}

/* Exceptions */
.hero-title, .hero-subtitle {
    -webkit-text-fill-color: initial;
}
.hero-title {
    background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
p.hero-subtitle {
    color: #334155 !important;
}

/* Spinner Animation Override */
.stSpinner > div > div {
    border-color: #3B82F6 transparent transparent transparent !important;
}

/* Fix Bottom Bar Background & App Theming */
[data-testid="stBottomBlockContainer"], 
[data-testid="stBottom"], 
[data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
    padding-bottom: 25px !important;
}
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.6) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05) !important;
}
[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
}
/* Ensure the typing caret is also white */
[data-testid="stChatInput"] textarea::placeholder {
    color: #94A3B8 !important;
}

/* Footer centered relative to Chat Window using pseudo-element */
[data-testid="stBottom"]::after {
    content: "Engineered with FAISS, Streamlit, and Qwen Models";
    display: block;
    text-align: center;
    color: #64748B;
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: -10px;
    padding-bottom: 15px;
    pointer-events: none;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Inject animated clouds HTML
CLOUDS_HTML = """
<div class="clouds-container">
  <div class="cloud cloud1"></div>
  <div class="cloud cloud2"></div>
  <div class="cloud cloud3"></div>
  <div class="cloud cloud4"></div>
  <div class="cloud cloud5"></div>
</div>
"""
st.markdown(CLOUDS_HTML, unsafe_allow_html=True)

# Dark Mode Overrides
is_dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)
if is_dark_mode:
    DARK_CSS = """
    <style>
    .stApp {
        background: linear-gradient(180deg, #020111 0%, #20124d 100%) !important;
    }
    .cloud {
        background: rgba(70, 80, 100, 0.45) !important;
        box-shadow: inset 0 -5px 15px rgba(0,0,0,0.5) !important;
    }
    .cloud::before, .cloud::after {
        background: rgba(70, 80, 100, 0.45) !important;
    }
    
    /* Panel and UI Adjustments for Dark Mode */
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stFileUploadDropzone"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Text Color Adjustments */
    html, body, [class*="css"],
    .hero-subtitle, .section-heading, p, li, span, h1, h2, h3, h4,
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stFileUploadDropzone"] * {
        color: #F8FAFC !important;
    }
    
    .hero-title {
        background: linear-gradient(135deg, #93C5FD 0%, #E0F2FE 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stSelectbox > div > div > div, [data-testid="stChatInput"] {
        background: rgba(15, 23, 42, 0.8) !important;
        color: #F8FAFC !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    </style>
    """
    st.markdown(DARK_CSS, unsafe_allow_html=True)

# Main App Header
st.markdown("<div class='hero-title'>Cognitive Assistant</div>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Upload your documents and receive context-aware explanations tailored to your expertise level.</p>", unsafe_allow_html=True)

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

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Question Input
st.markdown("<div class='section-heading' style='text-align: center; margin-top: 1rem; color: #475569;'>Query Engine</div>", unsafe_allow_html=True)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "context" in message:
            with st.expander("View Retrieved Context"):
                st.text(message["context"])
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter the technical concept or question..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        st.error("Authentication required: Please set the HUGGINGFACEHUB_API_TOKEN environment variable.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing response..."):
                # Pass all messages EXCEPT the last one (which is the current user query) to prevent duplication during LLM history processing
                history_for_backend = st.session_state.messages[:-1]
                response, context = run_rag_pipeline(prompt, difficulty, history_for_backend)
                
                if response.startswith("[Error]"):
                    st.error(response)
                else:
                    with st.expander("View Retrieved Context"):
                        st.text(context)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "context": context
                    })
