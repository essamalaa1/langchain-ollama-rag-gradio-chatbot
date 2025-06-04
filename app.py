import streamlit as st
from langchain.memory import ConversationBufferMemory
import config 
from llm_services import get_llm, get_embeddings
from vector_store_manager import load_or_build_vector_store
from tool_definitions import get_tools
from agent_builder import create_agent_executor
import os

# --- Page Configuration ---
st.set_page_config(page_title="LangChain RAG Chatbot", layout="wide")

# --- Environment Variables ---
if config.SERPAPI_API_KEY:
    os.environ["SERPAPI_API_KEY"] = config.SERPAPI_API_KEY
else:
    st.error("SERPAPI_API_KEY not found in config.py or environment. Web search tool will not function.")

# --- Initialization Functions (Cached) ---
@st.cache_resource
def initialize_llm_and_embeddings():
    """Initializes and caches the LLM and embedding models."""
    llm = get_llm()
    embeddings = get_embeddings()
    if not llm:
        st.error("Fatal: Failed to initialize Language Model (LLM). Please check Ollama server and model name in config.py.")
    if not embeddings:
        st.warning("Warning: Failed to initialize Embeddings model. RAG functionality (local document search) will be disabled or limited.")
    return llm, embeddings

@st.cache_resource
def initialize_retriever(_llm, _embeddings, _force_rebuild):
    """
    Initializes and caches the retriever.
    _force_rebuild is the crucial parameter from the UI.
    """
    if not _embeddings:
        st.warning("Embeddings model not available, skipping retriever initialization. Local document search will not work.")
        return None
    return load_or_build_vector_store(config.LANGCHAIN_DOC_URLS, _embeddings, force_rebuild=_force_rebuild)

@st.cache_resource
def initialize_agent_system(_llm, _retriever):
    """
    Initializes and caches the agent executor system.
    Depends on the LLM and the retriever.
    """
    if not _llm:
        st.error("Fatal: LLM not available, cannot initialize agent system.")
        return None

    tools = get_tools(_llm, _retriever)

    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='output'
        )
    
    agent_executor = create_agent_executor(tools, _llm, st.session_state.conversation_memory)
    return agent_executor

# --- Main Application UI and Logic ---
st.title("🦜 LangChain RAG Chatbot")
st.caption("Ask questions about LangChain, search the web, or run Python code.")

# --- Sidebar Definition ---
st.sidebar.header("⚙️ Options")
force_rebuild_vector_store_checkbox = st.sidebar.checkbox("Force Rebuild Vector Store on Next Load/Refresh", False)

if st.sidebar.button("Clear Chat History & Memory"):
    st.session_state.messages = [{"role": "assistant", "content": "Chat history and memory cleared. How can I help you now?"}]
    if "conversation_memory" in st.session_state:
        st.session_state.conversation_memory.clear()
    if "agent_executor" in st.session_state:
        del st.session_state.agent_executor
    if "retriever_identity_for_agent" in st.session_state: # Use the correct key here
        del st.session_state.retriever_identity_for_agent
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("Built with LangChain, Ollama, and Streamlit.")

# --- Initialize Core Components ---
llm, embeddings = initialize_llm_and_embeddings()

if not llm:
    st.error("LLM could not be initialized. The application cannot continue.")
    st.stop()

retriever = None
if embeddings:
    retriever = initialize_retriever(llm, embeddings, force_rebuild_vector_store_checkbox)
else:
    st.warning("Embeddings not initialized. Local document search (RAG) is disabled.")

if retriever:
    st.sidebar.success("✅ RAG system (local docs) is active.")
else:
    st.sidebar.warning("⚠️ RAG system (local docs) is NOT active. Will rely on web search.")

# --- Initialize Agent Executor ---
current_retriever_identity = id(retriever) if retriever is not None else None

if "agent_executor" not in st.session_state or \
   st.session_state.get("retriever_identity_for_agent") != current_retriever_identity:
    
    with st.spinner("Initializing AI Agent..."):
        st.session_state.agent_executor = initialize_agent_system(llm, retriever)
        st.session_state.retriever_identity_for_agent = current_retriever_identity

agent_executor = st.session_state.agent_executor

if not agent_executor:
    st.error("Agent Executor could not be initialized. The application cannot process queries.")
    st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today with LangChain, web searches, or Python tasks?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🤖 Thinking..."):
            try:
                response_payload = agent_executor.invoke({"input": prompt})
                ai_response = response_payload.get('output', "Sorry, I received an unexpected response format from the agent.")
            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")
                ai_response = "I encountered an error while processing your request. Please check the console for details or try again."
                print(f"Detailed agent error: {e}")
            
            full_response = ai_response
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})