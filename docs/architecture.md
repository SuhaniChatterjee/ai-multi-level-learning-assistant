# AI Multi-Level Learning Assistant - Architecture

## System Overview
The system relies on a Retrieval-Augmented Generation (RAG) pipeline to dynamically serve customized learning explanations.

1. **Document Ingestion** (`backend/document_processor.py`): Extracts text from PDFs using `pypdf` and splits it via LangChain `RecursiveCharacterTextSplitter`.
2. **Vector Store** (`backend/vector_store.py`): Embeds text chunks via `sentence-transformers` and indexes them using `faiss-cpu`.
3. **LLM Engine** (`backend/llm_engine.py`): Queries open-source LLMs out of HuggingFace's Inference API.
4. **Prompt Templates** (`backend/prompt_templates.py`): Injects system prompts mimicking different difficulty levels.
5. **Frontend** (`frontend/app.py`): A Streamlit web application orchestrating the flow.
