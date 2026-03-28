import os
from langchain_huggingface import HuggingFaceEndpoint

def generate_response(prompt: str) -> str:
    \"\"\"
    Sends a prompt to the LLM and returns the generated response.
    Uses Hugging Face Inference API for open-source models.
    \"\"\"
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        return "[Error] HUGGINGFACEHUB_API_TOKEN environment variable not set. Please set it to use the LLM."
        
    try:
        # Using a highly capable instruction-tuned open source model
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=512,
            huggingfacehub_api_token=api_token,
            temperature=0.3
        )
        
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"[Error] LLM generation failed: {e}"
