# --- API Keys ---
SERPAPI_API_KEY = "7ffdbf28402b8d9ea42603fe9c84e6cb7e963a17ba330bcc120abc8f219b6548"

# --- Models ---
LLM_MODEL = "llama3:latest"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Document URLs ---
LANGCHAIN_DOC_URLS = [
    "https://python.langchain.com/docs/get_started/introduction",
    "https://python.langchain.com/docs/how_to/migrate_agent/",
    "https://python.langchain.com/docs/versions/migrating_chains/",
    "https://python.langchain.com/docs/integrations/llms/ollama",
    "https://python.langchain.com/docs/integrations/vectorstores/chroma",
    "https://python.langchain.com/docs/concepts/#llms",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/expression_language/",
    "https://python.langchain.com/docs/integrations/document_loaders/web_base/",
    "https://python.langchain.com/docs/integrations/tools/searx_search/",
]

# --- Directories and Filenames ---
SCRAPED_DOCS_DIR = "scraped_langchain_docs"
PERSIST_DB_DIR = "langchain_chroma_db_persistent"
COLLECTION_NAME = "langchain_docs_scraped_persistent_v1"

# --- RAG Configuration ---
SIMILARITY_THRESHOLD = 0.6

# --- Agent Configuration ---
MAX_ITERATIONS = 7