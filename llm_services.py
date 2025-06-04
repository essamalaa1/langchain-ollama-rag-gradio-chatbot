from langchain_ollama import ChatOllama, OllamaEmbeddings
import config

def get_llm():
    """Initializes and returns the Language Model."""
    return ChatOllama(model=config.LLM_MODEL, temperature=0.0)

def get_embeddings():
    """Initializes and returns the embedding model."""
    try:
        return OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        print("Please ensure Ollama is running and the embedding model is available.")
        print(f"Attempted to use: {config.EMBEDDING_MODEL}")
        print("You can pull models using 'ollama pull <model_name>'")
        return None