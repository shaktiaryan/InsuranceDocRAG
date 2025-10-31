from typing import TypedDict, Any
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from acs_agents.tools import policy_docs_qna_tool, customer_data_tool
from config.settings import Settings
from pydantic import SecretStr

load_dotenv()

# Handle the Azure OpenAI configuration
api_key = Settings.AZURE_OPENAI_API_KEY
endpoint = Settings.AZURE_OPENAI_ENDPOINT
api_version = Settings.AZURE_OPENAI_API_VERSION
model_name = Settings.MODEL_NAME

if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY is not set in environment variables")
if not endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT is not set in environment variables")
if not api_version:
    raise ValueError("AZURE_OPENAI_API_VERSION is not set in environment variables")

llm = AzureChatOpenAI(
    azure_deployment=model_name,
    azure_endpoint=endpoint,
    api_key=SecretStr(api_key),
    api_version=api_version,
    temperature=0.7
)

tools = [policy_docs_qna_tool, customer_data_tool]

system_message = (
    "You are the India First Life Insurance AI Assistant, built to help customers with "
    "policy-related queries, scheme information, and customer service. "
    "Your role is to provide clear, professional, and concise answers (max 50 words).\n\n"
    "Capabilities:\n"
    "- Use the policy documentation tool when customers ask about policies, schemes, terms, or coverage details.\n"
    "- Use the customer data tool when customers provide their customer ID or policy number.\n"
    "- For general insurance/life insurance questions, answer directly using your knowledge.\n\n"
    "Professional Boundaries:\n"
    "- Only access customer data when explicit customer ID is provided\n"
    "- Do not share sensitive customer information\n"
    "- Do not provide financial advice or guarantee claims approval\n"
    "- Direct complex cases to human customer service\n"
    "- Keep responses professional and within insurance domain\n"
    "- If query is unrelated to insurance, politely decline\n"
)

class Input(TypedDict):
    session_id: str
    query: str

class Output(TypedDict):
    response: str

checkpointer = InMemorySaver()

react_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_message,
    checkpointer=checkpointer,
)

def insurance_react_agent(state: Input) -> Output:
    session_id = state["session_id"]
    query = state["query"]
    
    # Cast to Any to avoid type checking issues with config
    config: Any = {"configurable": {"thread_id": session_id}}
    
    ai_msg = react_agent.invoke(
        input={"messages": [{"role": "user", "content": query}]},
        config=config,
    )
    
    return {"response": ai_msg['messages'][-1].content}