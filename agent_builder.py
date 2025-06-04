from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import config

def create_agent_executor(tools, llm, memory):
    """Initializes and returns the LangChain agent executor."""
    system_message_content = """
        You are a helpful AI assistant. Solve tasks using Tools.
        Respond to the human as helpfully and accurately as possible.
        You have access to the following tools: LangChainDocsQA, WebSearch, PythonInterpreter.

        The way you use tools is by specifying a json blob.
        Specifically, this json blob must have a `action` key (with the name of the tool to use) and `action_input` key (with the input to the tool).
        The only values that can be in the "action" field are: LangChainDocsQA, WebSearch, PythonInterpreter.

        The $JSON_BLOB must always be enclosed in triple backticks.

        Example:
        Human: What is the capital of France?
        Assistant:
        ```json
        {
        "action": "WebSearch",
        "action_input": "capital of France"
        }
        If you have enough information to answer the human's request, you will respond with "Final Answer:" followed by the answer.
        Do not output the JSON blob if you are providing a Final Answer.
        Begin!
    """
    agent_kwargs = {
    "system_message": system_message_content,
    }

    original_system_prompt = SystemMessagePromptTemplate.from_template("""
    You are an expert assistant that outputs only a JSON object in this exact form:
    {"action":"<ToolName>","action_input":"<string>"}
    action must be one of: LangChainDocsQA, WebSearch, PythonInterpreter, or "Final Answer" if you can answer directly.
    If action is "Final Answer", action_input should be your complete response to the user.
    action_input must be exactly the string to pass to that tool (no extra quotes, no nested dicts if it's a tool input).
    Do not output any other text or formatting besides this JSON object.
    """)
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    prompt_for_agent = ChatPromptTemplate.from_messages([
    original_system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),
    human_prompt,
    MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent_kwargs_custom_prompt = {"prompt": prompt_for_agent}
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs=agent_kwargs_custom_prompt,
        verbose=True,
        handle_parsing_errors="force",
        max_iterations=config.MAX_ITERATIONS,
        early_stopping_method="generate"
    )