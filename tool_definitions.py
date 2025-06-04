from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import SerpAPIWrapper
import config

def get_top_similarity_score(query: str, retriever, k=1) -> float:
    """
    Gets the similarity score of the top k retrieved documents.
    Chroma by default uses L2 distance (lower is better).
    """
    try:
        docs_and_scores = retriever.vectorstore.similarity_search_with_score(query, k=k)
        if docs_and_scores:
            return docs_and_scores[0][1]
    except Exception as e:
        print(f"Error computing similarity score: {e}")
    return float('inf')

def _create_rag_tool_func_internal(query: str, rag_chain, retriever, search_utility, similarity_threshold: float):
    if rag_chain and retriever:
        try:
            top_score = get_top_similarity_score(query, retriever)
            print(f"RAG Tool: Query '{query}', Top similarity score (distance): {top_score}, Threshold: {similarity_threshold}")

            if top_score < similarity_threshold:
                print("RAG Tool: Score is good, proceeding with RAG.")
                result = rag_chain.invoke({"query": query})
                answer = result.get("result", "").strip()
                sources = result.get("source_documents", [])

                if answer: 
                    response_text = f"{answer}"
                    if sources:
                        citations = "\n".join(sorted(list(set(f"- {doc.metadata.get('source')}" for doc in sources))))
                        response_text += f"\n\nSources:\n{citations}"
                    return response_text
                else:
                    print("RAG Tool: RAG returned no answer. Falling back.")
    
            else:
                print(f"RAG Tool: Score {top_score} is not better than threshold {similarity_threshold}. Falling back to web search.")
                

        except Exception as e:
            print(f"RAG Tool: Error during RAG execution: {e}. Falling back to web search.")
        
    else:
        print("RAG Tool: RAG chain or retriever not available. Falling back to web search.")

    try:
        print(f"RAG Tool: Falling back to web search for '{query}'.")
        web_results = search_utility.run(query)
        return (
            "I couldn't find a definitive answer in my local documents, so here's what I found online:\n"
            f"{web_results}"
        )
    except Exception as e:
        print(f"RAG Tool: WebSearch error during fallback: {e}")
        return "Sorry, I'm unable to perform a web search right now due to an error."


def get_tools(llm, retriever):
    """Initializes and returns the list of tools for the agent."""
    if retriever:
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    else:
        rag_chain = None
        print("Warning: Retriever not available. LangChainDocsQA tool will always fall back to web search.")

    search_utility = SerpAPIWrapper(serpapi_api_key=config.SERPAPI_API_KEY)

    def rag_tool_func_for_agent(query: str):
        return _create_rag_tool_func_internal(
            query,
            rag_chain,
            retriever,
            search_utility,
            config.SIMILARITY_THRESHOLD
        )

    langchain_docs_tool = Tool(
        name="LangChainDocsQA",
        func=rag_tool_func_for_agent,
        description=(
            "Use this tool to answer questions about LangChain, its components, "
            "how to use it, example code, integrations, and concepts. "
            "It uses a local knowledge base of LangChain documentation. "
            "If the information is not found locally or the confidence is low, it will automatically search the web."
        )
    )

    def safe_web_search(query: str) -> str:
        try:
            return search_utility.run(query)
        except Exception as e:
            print(f"WebSearch tool error: {e}")
            return "Sorry, I couldn't perform a web search at this moment due to an error."

    web_search_tool = Tool(
        name="WebSearch",
        func=safe_web_search,
        description=(
            "Use this tool for general web searches to find information on topics not covered by LangChainDocsQA, "
            "such as current events, general knowledge, or information outside the LangChain documentation. "
            "Also use if LangChainDocsQA indicates it is falling back to web search."
        )
    )

    python_tool = Tool(
        name="PythonInterpreter",
        func=PythonREPLTool().run,
        description=(
            "Use this tool to execute Python code. You can use it for calculations, "
            "data manipulation, or any other task that can be solved with a short Python script. "
            "Input should be a valid Python script. The output will be the result of the script's execution."
        )
    )
    
    tools = [langchain_docs_tool, web_search_tool, python_tool]
    if not retriever:
        print("Retriever not available, LangChainDocsQA functionality is limited to web fallback.")

    return tools