import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

def get_chat_model():
    """Helper function to instantiate the ChatHuggingFace model."""
    api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not api_token:
        raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN environment variable. Please export it before running.")
        
    # Using Qwen2.5-72B-Instruct as it reliably supports serverless conversational endpoints on HF API
    repo_id = "Qwen/Qwen2.5-72B-Instruct"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=1024,
        huggingfacehub_api_token=api_token,
        temperature=0.3
    )
    return ChatHuggingFace(llm=llm)

def rephrase_question(chat_history: list, current_question: str) -> str:
    """
    Given a chat history and the latest user question, formulate a standalone question 
    which can be understood without the chat history.
    """
    if not chat_history:
        return current_question
        
    try:
        chat_model = get_chat_model()
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
        rephrase_prompt = f"Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, without answering it.\n\nChat History:\n{history_text}\n\nFollow Up Input: {current_question}\nStandalone question:"
        
        messages = [HumanMessage(content=rephrase_prompt)]
        response = chat_model.invoke(messages)
        standalone_q = response.content.strip()
        # Basic cleanup if the model repeats the prefix
        if "Standalone question:" in standalone_q:
            standalone_q = standalone_q.split("Standalone question:")[-1].strip()
        return standalone_q
    except Exception as e:
        import logging
        logging.error(f"Error rephrasing question: {e}")
        return current_question

def generate_response(prompt: str, chat_history: list = None) -> str:
    """
    Generate an answer using the HuggingFace Inference API for an open-source model.
    """
    try:
        chat_model = get_chat_model()
        messages = []
        
        if chat_history:
            for msg in chat_history:
                # Assuming chat_history is a list of dicts like: {"role": "user"/"assistant", "content": "..."}
                # For basic chat memory, we only care about text content to provide conversation history to the model
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg.get("content", "")))
                    
        # Add the newest prompt which contains instructions + context + question
        messages.append(HumanMessage(content=prompt))
        
        response = chat_model.invoke(messages)
        return response.content
        
    except ValueError as ve:
        raise ve
    except Exception as e:
        return f"[Error] LLM generation failed: {e}"
