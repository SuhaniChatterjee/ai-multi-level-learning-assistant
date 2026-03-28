# 🌟 AI Multi-Level Learning Assistant

An intelligent, context-aware AI tutor built to explain PDF study materials at multiple difficulty levels: from "Explain Like I'm 5" to College-level, straight to Interview-ready! Powered by LangChain, FAISS, and Hugging Face Open-Source Models.

## 🎯 Features
- **Local RAG Pipeline:** Upload your own PDFs to ground the AI strictly in your material, avoiding hallucinations.
- **Dynamic Difficulty Levels:** Easily toggle between ELI5, College, or Interview-ready explanations.
- **Open-Source LLMs:** Uses HuggingFace Inference API (Mistral) for capable, open-source generation.
- **FAISS Vector Store:** Lightning-fast context retrieval.

## 🚀 Setup & Installation
1. Install dependencies:
   ```bash
   conda install -c conda-forge faiss-cpu -y
   pip install -r requirements.txt
   pip install langchain langchain-community langchain-huggingface pypdf
   ```
2. Get your Hugging Face API Token and export it:
   ```bash
   export HUGGINGFACEHUB_API_TOKEN="hf_your_token_here"
   ```
3. Run the application:
   ```bash
   streamlit run frontend/app.py
   ```
