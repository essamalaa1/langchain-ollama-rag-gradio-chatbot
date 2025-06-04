# LangChain RAG Chatbot with Ollama and Streamlit

This project implements a sophisticated RAG (Retrieval Augmented Generation) chatbot using LangChain, powered by local Ollama models, and featuring a user-friendly Streamlit interface. It can answer questions based on a scraped knowledge base (LangChain documentation), fall back to web search via SerpAPI for general queries, and even execute Python code.

## Features

- **Retrieval Augmented Generation (RAG):** Answers questions using a knowledge base built from scraped LangChain documentation.
- **Local LLM & Embeddings:** Utilizes Ollama to run language models (e.g., `llama3:latest`) and embedding models (e.g., `nomic-embed-text`) locally.
- **Web Search Fallback:** If information isn't found in the local RAG system or the query is general, it uses SerpAPI for web searches.
- **Python Code Execution:** Includes a tool for the agent to execute Python code via `PythonREPLTool`.
- **Conversational Memory:** Remembers previous turns in the conversation for context.
- **Interactive UI:** Built with Streamlit for an easy-to-use chat interface.
- **Persistent Vector Store:** Uses ChromaDB to store document embeddings, persisting them locally for faster startups after the initial build.
- **Configurable:** Easily configure API keys, model names, document URLs, and RAG thresholds.
- **Modular Codebase:** Organized into separate Python files for better maintainability.

## Tech Stack

- **LangChain:** Core framework for building LLM applications.
- **Ollama:** For running local LLMs (e.g., Llama 3) and embedding models.
- **Streamlit:** For creating the web-based user interface.
- **ChromaDB:** Vector store for RAG.
- **SerpAPI:** For the web search tool.
- **BeautifulSoup4 & Requests:** For scraping web content to build the knowledge base.
- **Python 3.9+**

## Prerequisites

Before you begin, ensure you have the following installed and set up:

1. **Python:** Version 3.9 or higher.
2. **Ollama:** Installed and running. You can download it from [ollama.com](https://ollama.com/).
3. **SerpAPI Account:** You'll need an API key from [SerpAPI](https://serpapi.com/) for the web search functionality.

## Setup and Installation

1. **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2. **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` lists all necessary packages: `streamlit`, `langchain`, `langchain-community`, `langchain-experimental`, `langchain-ollama`, `langchain-chroma`, `beautifulsoup4`, `requests`, `chromadb`, `tiktoken`, `pandas`, `google-search-results`)*

4. **Pull Ollama Models:**
    ```bash
    ollama pull llama3:latest
    ollama pull nomic-embed-text
    ```
    Ensure Ollama is running in the background.

5. **Configure API Keys and Settings:**
    - Rename `config.example.py` to `config.py` (if provided).
    - Open `config.py` and set your `SERPAPI_API_KEY`:
        ```python
        SERPAPI_API_KEY = "YOUR_SERPAPI_API_KEY"
        ```
    - Review other settings like `LLM_MODEL`, `EMBEDDING_MODEL`, `LANGCHAIN_DOC_URLS`, etc.

## Running the Application

1. **Ensure Ollama is Running.**

2. **Start the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    This opens the app in your browser (usually at `http://localhost:8501`).

3. **Initial Vector Store Build:**
    On the first run (or if the vector store is empty), the app will scrape content and build the ChromaDB vector store. This may take a few minutes.

## Configuration (`config.py`)

The `config.py` file contains:

- `SERPAPI_API_KEY`: Your SerpAPI key.
- `LLM_MODEL`: Ollama model for the main LLM (e.g., `"llama3:latest"`).
- `EMBEDDING_MODEL`: Ollama embedding model (e.g., `"nomic-embed-text"`).
- `LANGCHAIN_DOC_URLS`: URLs to scrape for RAG.
- `SCRAPED_DOCS_DIR`: Directory for storing scraped text.
- `PERSIST_DB_DIR`: Directory for ChromaDB persistence.
- `COLLECTION_NAME`: Name for the ChromaDB collection.
- `SIMILARITY_THRESHOLD`: Float value (e.g., `0.6`) for deciding fallback.
- `MAX_ITERATIONS`: Max steps for the LangChain agent.

.
├── app.py                          # Main Streamlit app
├── agent_builder.py                # LangChain agent setup
├── config.py                       # Config file
├── llm_services.py                 # LLM and embedding initialization
├── scraper.py                      # Scraping utility
├── tool_definitions.py             # RAG, search, and Python tools
├── vector_store_manager.py         # ChromaDB setup and loading
├── calculate_similarity_threshold.py
├── scraped_langchain_docs/         # Cached scraped docs
├── langchain_chroma_db_persistent/ # Vector store directory
├── requirements.txt
└── README.md
## How It Works

1. **User Input:** Query entered in Streamlit UI.
2. **Agent Processing:** LangChain agent processes query.
3. **Tool Selection:**
    - **LangChainDocsQA:** For RAG queries using ChromaDB.
    - **WebSearch:** Fallback via SerpAPI for general queries or when RAG fails.
    - **PythonInterpreter:** Executes Python code when appropriate.
4. **Tool Execution:** Agent runs the selected tool.
5. **Final Response:** LLM composes and returns an answer based on the tool output.
6. **Memory:** Previous chats are retained to provide contextual answers in future interactions.

---

## Troubleshooting

### Ollama Issues

- Verify Ollama is properly installed and running as a background service.
- Ensure all required models are downloaded using:
  ```bash
  ollama pull llama3:latest
  ollama pull nomic-embed-text
### SerpAPI Errors
- Double-check your `SERPAPI_API_KEY` in `config.py` is valid and active.

### Streamlit Caching Errors
- Streamlit may throw errors related to unhashable types.
- This app uses `id()` to manage cache keying when dealing with complex objects like retrievers.

### LangChain Warnings
- LangChain is evolving quickly. If you see deprecation warnings:
  - Run:
    ```bash
    pip install -U langchain langchain-community langchain-experimental
    ```
  - Adjust your code according to the updated API.

### RAG Not Returning Info
- The answer may not exist in your scraped documents.
- Try adjusting the `SIMILARITY_THRESHOLD` in `config.py`.
- Consider whether your embedding model is suitable for your specific use case.

### Slow Startup
- On first run, documents are scraped and vectorized.
- Future runs use the persisted ChromaDB, significantly improving load times.

---
