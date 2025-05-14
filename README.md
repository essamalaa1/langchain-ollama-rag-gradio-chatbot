# LangChain Conversational RAG Chatbot with Ollama and Gradio

This project implements a Conversational RAG (Retrieval-Augmented Generation) chatbot using LangChain. It's powered by a local LLM (e.g., Llama3) hosted via Ollama and features an interactive Gradio user interface.

The chatbot is designed to answer technical queries by:
1.  Retrieving information from a vector database built from LangChain documentation.
2.  Performing internet searches via SerpAPI as a fallback or for general queries.
3.  Executing Python code for calculations or dynamic reasoning.

## Features

*   **Local LLM Integration**: Uses Ollama to serve local LLMs (e.g., `llama3:latest`).
*   **Local Embeddings**: Utilizes Ollama for embedding models (e.g., `nomic-embed-text`).
*   **Document RAG**:
    *   Scrapes specified LangChain documentation URLs.
    *   Caches scraped content locally.
    *   Splits documents into chunks and embeds them.
    *   Stores and retrieves document chunks from a persistent ChromaDB vector store.
*   **Intelligent RAG Fallback**: Uses similarity scores to determine if RAG output is relevant; falls back to web search if not.
*   **Web Search Capability**: Integrates SerpAPI for general web queries.
*   **Python Code Execution**: Includes a Python REPL tool for dynamic computations.
*   **Conversational Memory**: Maintains conversation history for context.
*   **LangChain Agent**: Employs a `CHAT_CONVERSATIONAL_REACT_DESCRIPTION` agent to intelligently route user queries to the appropriate tool (RAG, Search, Python).
*   **Interactive UI**: Provides a user-friendly chat interface using Gradio.

## Core Technologies

*   **LLM & Embeddings Host**: Ollama
*   **LLM Model**: `llama3:latest` (configurable)
*   **Embedding Model**: `nomic-embed-text` (configurable)
*   **Orchestration Framework**: LangChain
*   **Vector Store**: ChromaDB
*   **Web Scraping**: BeautifulSoup, Requests
*   **Search API**: SerpAPI
*   **Chat Interface**: Gradio
*   **Programming Language**: Python 3.10+

## Prerequisites

*   Python 3.10 or higher.
*   [Ollama](https://ollama.com/) installed and running.
*   A [SerpAPI Account and API Key](https://serpapi.com/).

## Setup Instructions

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/langchain-ollama-rag-gradio-chatbot.git
    cd langchain-ollama-rag-gradio-chatbot
    ```

2.  **Create and Activate a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Create a `requirements.txt` file with the following content:
    ```txt
    langchain
    langchain-ollama
    langchain-community
    langchain-experimental
    beautifulsoup4
    requests
    chromadb
    gradio
    ipywidgets
    pandas
    # python-dotenv # Optional, for .env file management if you modify the script
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Ollama**:
    *   Ensure Ollama is installed and the Ollama application/service is running.
    *   Pull the LLM and embedding models specified in the notebook (`a.ipynb`):
        ```bash
        ollama pull llama3:latest
        ollama pull nomic-embed-text
        ```
        If you change `LLM_MODEL` or `EMBEDDING_MODEL` in the notebook, pull those models instead.

5.  **Configure SerpAPI Key**:
    The notebook `a.ipynb` currently has a placeholder API key. **You MUST replace this.**
    Open `a.ipynb` and find the line:
    ```python
    os.environ["SERPAPI_API_KEY"] = "7ffdbf28402b8d9ea42603fe9c84e6cb7e963a17ba330bcc120abc8f219b6548"
    ```
    Replace the placeholder key with your actual SerpAPI key:
    ```python
    os.environ["SERPAPI_API_KEY"] = "YOUR_ACTUAL_SERPAPI_API_KEY"
    ```
    Alternatively, for better security, remove this line from the notebook and set the environment variable in your terminal session before running Jupyter:
    ```bash
    # On macOS/Linux
    export SERPAPI_API_KEY="YOUR_ACTUAL_SERPAPI_API_KEY"
    # On Windows (Command Prompt)
    set SERPAPI_API_KEY="YOUR_ACTUAL_SERPAPI_API_KEY"
    # On Windows (PowerShell)
    $env:SERPAPI_API_KEY="YOUR_ACTUAL_SERPAPI_API_KEY"
    ```

## Running the Application

1.  Ensure Ollama is running and the required models are available.
2.  Ensure your SerpAPI key is correctly configured (as per Step 5 in Setup).
3.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook a.ipynb
    # or
    jupyter lab a.ipynb
    ```
4.  Run all cells in the `a.ipynb` notebook.
    *   **First Run**: The notebook will scrape documents from `LANGCHAIN_DOC_URLS`, process them, and build the ChromaDB vector store. This may take some time. Scraped content will be cached in `scraped_langchain_docs/` and the vector store in `langchain_chroma_db_persistent/`.
    *   **Subsequent Runs**: The notebook will load the cached documents and the existing vector store, making startup faster.
5.  The Gradio chat interface will launch. The URL (usually `http://127.0.0.1:7860`) will be displayed in the notebook output. Open this URL in your web browser to interact with the chatbot.

## How It Works

*   **Document Ingestion & RAG**:
    *   Content from specified LangChain documentation URLs is scraped using `BeautifulSoup`.
    *   Text is chunked using `RecursiveCharacterTextSplitter`.
    *   Embeddings are generated via `OllamaEmbeddings` (`nomic-embed-text`).
    *   Chunks are stored in a persistent `Chroma` vector store.
    *   The `LangChainDocsQA` tool queries this vector store. It uses a similarity threshold (distance-based, lower is better) to decide if the retrieved documents are relevant. If not, or if RAG fails, it can fall back to web search. The threshold is analyzed in cell 11 of the notebook, with a default of `0.8` in the `rag_tool_func`.
*   **Agent and Tools**:
    *   A `CHAT_CONVERSATIONAL_REACT_DESCRIPTION` agent from LangChain orchestrates interactions.
    *   The agent uses a custom prompt to ensure it outputs structured JSON for tool selection and input.
    *   It has access to three tools:
        1.  `LangChainDocsQA`: For querying the LangChain documentation vector store.
        2.  `WebSearch`: For general internet queries via `SerpAPIWrapper`.
        3.  `PythonInterpreter`: For executing Python code via `PythonREPLTool`.
    *   `ConversationBufferMemory` stores chat history for contextual responses.

## Example Prompts

You can try these prompts in the Gradio interface:

*   "How do I build a custom agent in LangChain?"
*   "What are the key features of llama3?"
*   "What's 2 to the power of 20 in Python?"
*   "Give me an intro to LangChain."
*   "What is the capital of France?"
*   "Current US president?"


## Troubleshooting & Notes

*   **Ollama Not Running**: Ensure the Ollama application/service is active and accessible before running the notebook.
*   **Models Not Found**: Double-check that you have pulled `llama3:latest` and `nomic-embed-text` (or your configured models) in Ollama.
*   **SerpAPI Key Issues**: Verify your `SERPAPI_API_KEY` is correct and has not expired or exceeded limits.
*   **Long First Run**: The initial run involves scraping and vectorization, which can be time-consuming depending on the number of URLs and document sizes.
*   **Agent Parsing Errors**: The agent relies on the LLM to output strictly formatted JSON. If the LLM fails to do so, parsing errors can occur. The notebook uses `handle_parsing_errors="force"`, which attempts to recover, but suboptimal responses or loops might still happen. The quality of the LLM and the prompt are critical here.
*   **TqdmWarning**: You might see a `TqdmWarning: IProgress not found`. This is usually related to `ipywidgets` in Jupyter environments and typically doesn't affect core functionality. Installing `ipywidgets` (included in `requirements.txt`) should help.
*   **Deprecated `agent_executor.run`**: The notebook uses `agent_executor.run()`, which is deprecated. For future development, consider migrating to `agent_executor.invoke()`.
