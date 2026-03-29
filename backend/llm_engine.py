import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage

load_dotenv()

def generate_response(prompt: str) -> str:
    """
    Generate an answer using the HuggingFace Inference API for an open-source model.
    """
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN environment variable. Please export it before running.")
        
    try:
        # Using Qwen2.5-72B-Instruct as it reliably supports serverless conversational endpoints on HF API
        repo_id = "Qwen/Qwen2.5-72B-Instruct"
        
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_new_tokens=1024,
            huggingfacehub_api_token=api_token,
            temperature=0.3
        )
        
        chat_model = ChatHuggingFace(llm=llm)
        messages = [HumanMessage(content=prompt)]
        response = chat_model.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"[Error] LLM generation failed: {e}"
