import os
import google.generativeai as genai

def generate_response(prompt: str) -> str:
    \"\"\"
    Sends a prompt to the LLM and returns the generated response.
    \"\"\"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "[Error] GEMINI_API_KEY environment variable not set. Please set it to use the LLM."
        
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash as the lightweight inference model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Error] LLM generation failed: {e}"
