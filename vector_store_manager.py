import os
import re
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import streamlit as st

import config
from scraper import scrape_and_extract_text

def load_or_build_vector_store(urls, embeddings_function, force_rebuild=False):
    """Loads an existing vector store or builds a new one if not found or empty."""
    if not embeddings_function:
        st.error("Embeddings model not available. Cannot build or load vector store.")
        return None

    os.makedirs(config.SCRAPED_DOCS_DIR, exist_ok=True)
    db_exists = os.path.isdir(config.PERSIST_DB_DIR) and any(os.scandir(config.PERSIST_DB_DIR))

    if db_exists and not force_rebuild:
        st.info(f"Attempting to load vector store from: {config.PERSIST_DB_DIR}")
        try:
            vectorstore = Chroma(
                persist_directory=config.PERSIST_DB_DIR,
                embedding_function=embeddings_function,
                collection_name=config.COLLECTION_NAME
            )
            if vectorstore._collection.count() > 0:
                st.success(f"Successfully loaded vector store with {vectorstore._collection.count()} documents.")
                return vectorstore.as_retriever(search_kwargs={"k": 3})
            else:
                st.warning("Vector store loaded but appears to be empty. Will rebuild.")
        except Exception as e:
            st.warning(f"Error loading vector store: {e}. Will attempt to rebuild.")
    else:
        if force_rebuild:
            st.info(f"Force rebuilding vector store.")
        else:
            st.info(f"Vector store not found at {config.PERSIST_DB_DIR} or is empty. Building new one.")

    with st.spinner("Building new vector store... This may take a while."):
        docs = []
        for i, url in enumerate(urls):
            sanitized_url_part = re.sub(r'[^a-zA-Z0-9_-]+', '_', url.split("//")[-1])
            filename = os.path.join(config.SCRAPED_DOCS_DIR, f"doc_{i}_{sanitized_url_part[:50]}.txt")
            text_content = None

            if os.path.exists(filename) and not force_rebuild:
                with open(filename, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                if text_content:
                    st.write(f"Loaded cached content from {filename}")
                else:
                    st.write(f"Cached file {filename} is empty. Re-scraping.")
                    text_content = None

            if not text_content:
                st.write(f"Scraping: {url}")
                text_content = scrape_and_extract_text(url)
                if text_content:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    st.write(f"Saved scraped content to {filename}")
                else:
                    st.warning(f"Failed to scrape or extract text from {url}. Skipping.")
                    continue

            if text_content:
                docs.append(Document(page_content=text_content, metadata={"source": url}))

        if not docs:
            st.error("No documents were successfully processed to build the vector store.")
            return None

        st.write(f"Collected {len(docs)} documents for vector store.")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        st.write(f"Split documents into {len(chunks)} chunks.")

        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings_function,
                collection_name=config.COLLECTION_NAME,
                persist_directory=config.PERSIST_DB_DIR
            )
            vectorstore.persist()
            st.success(f"Successfully built and persisted vector store with {vectorstore._collection.count()} chunks.")
            return vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            st.error(f"Error building vector store: {e}")
            return None