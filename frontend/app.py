import streamlit as st
import tempfile
import os
import sys

# Ensure backend can be imported safely from the top-level
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.document_processor import extract_text_from_pdf, chunk_text
from backend.vector_store import create_vector_store
from backend.rag_pipeline import run_rag_pipeline

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓")

st.title("🌟 AI Multi-Level Learning Assistant")
st.markdown("Upload your study material and get concepts explained at your level: from ELI5 to Interview Ready!")

# Sidebar for controls
with st.sidebar:
    st.header("1. Upload Material 📚")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Extracting text and building knowledge base..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # 1. Extract text
                    text = extract_text_from_pdf(tmp_path)
                    
                    # 2. Chunk text
                    chunks = chunk_text(text)
                    
                    # 3. Create Vector Store
                    create_vector_store(chunks)
                    
                    st.success("Document processed successfully! 🎉")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                finally:
                    os.unlink(tmp_path)
    
    st.header("2. Choose Level 🎚️")
    difficulty = st.selectbox(
        "Explanation Difficulty:",
        ("ELI5", "College", "Interview"),
        help="ELI5 = Simple analogies, College = Academic and structured, Interview = Crisp and technical"
    )

st.header("3. Ask a Question ❓")
question = st.text_input("What concept would you like explained?")

if st.button("Get Explanation"):
    if not question:
        st.warning("Please enter a question.")
    elif not os.environ.get("GEMINI_API_KEY"):
        st.error("Please set the GEMINI_API_KEY environment variable. You can export GEMINI_API_KEY=your_key in your terminal.")
    else:
        with st.spinner(f"Generating {difficulty}-level explanation..."):
            response, context = run_rag_pipeline(question, difficulty)
            
            if response.startswith("[Error]"):
                st.error(response)
            else:
                st.markdown("### 💡 Explanation")
                st.markdown(response)
                
                with st.expander("🔍 Show Source Context Used"):
                    st.text(context)
