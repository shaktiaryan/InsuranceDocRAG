from typing import TypedDict, Any, Dict
from dotenv import load_dotenv
import logging
import time
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from acs_agents.tools import (
    policy_docs_qna_tool, 
    customer_data_tool,
    customer_lookup_tool,
    policy_details_tool,
    claims_status_tool,
    payment_history_tool
)
from config.settings import Settings
from pydantic import SecretStr, BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insurance_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
api_key = Settings.AZURE_OPENAI_API_KEY
endpoint = Settings.AZURE_OPENAI_ENDPOINT
api_version = Settings.AZURE_OPENAI_API_VERSION
model_name = Settings.MODEL_NAME

# Enhanced Input/Output Models
class Metadata(BaseModel):
    timestamp: str
    session_duration: float
    tool_usage: Dict[str, int]
    error_count: int

# Enhanced configuration validation
def validate_azure_config():
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is not set in environment variables")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not set in environment variables")
    if not api_version:
        raise ValueError("AZURE_OPENAI_API_VERSION is not set in environment variables")
    logger.info("Azure OpenAI configuration validated successfully")

try:
    validate_azure_config()
    llm = AzureChatOpenAI(
        azure_deployment=model_name,
        azure_endpoint=endpoint,
        api_key=SecretStr(api_key if api_key else ""),
        api_version=api_version,
        temperature=0.7,
        model_kwargs={"request_timeout": 30},
        max_retries=3       # Add retries
    )
    logger.info(f"Successfully initialized Azure OpenAI model: {model_name}")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI: {str(e)}")
    raise

# Available tools with metadata
tools = [
    {
        "tool": policy_docs_qna_tool,
        "name": "policy_docs_qna_tool",
        "description": "Search insurance policy documentation",
        "required_params": ["query"]
    },
    {
        "tool": customer_data_tool,
        "name": "customer_data_tool",
        "description": "Retrieve customer information (legacy compatibility)",
        "required_params": ["customer_id"]
    },
    {
        "tool": customer_lookup_tool,
        "name": "customer_lookup_tool",
        "description": "Comprehensive customer lookup by ID, name, email, or phone",
        "required_params": ["customer_identifier"]
    },
    {
        "tool": policy_details_tool,
        "name": "policy_details_tool",
        "description": "Retrieve detailed policy information by policy number or ID",
        "required_params": ["policy_identifier"]
    },
    {
        "tool": claims_status_tool,
        "name": "claims_status_tool",
        "description": "Check claim status by claim number or customer ID",
        "required_params": ["claim_identifier", "customer_id"]
    },
    {
        "tool": payment_history_tool,
        "name": "payment_history_tool",
        "description": "Get payment history for customer or policy",
        "required_params": ["customer_id", "policy_id"]
    }
]

# Extract just the tool functions for the agent
tool_functions = [tool["tool"] for tool in tools]

# Enhanced system message with comprehensive tool instructions
system_message = """You are the India First Life Insurance AI Assistant, built to help customers with 
policy-related queries, scheme information, and customer service.

ROLE AND RESPONSIBILITIES:
- Provide clear, professional, and concise answers
- Use available tools appropriately based on query type
- Maintain a helpful and empathetic tone

ENHANCED TOOL USAGE GUIDELINES:

1. Policy Documentation Tool (policy_docs_qna_tool):
   - USE FOR: General policy information, terms, coverage, schemes
   - TRIGGER WORDS: "policy types", "coverage", "premium", "benefits", "insurance plans"
   - EXAMPLE: "Tell me about term life insurance policies"

2. Customer Lookup Tool (customer_lookup_tool):
   - USE FOR: Finding customer by ID, name, email, or phone
   - SUPPORTS: Customer ID (e.g., "1"), full name (e.g., "Rajesh Sharma"), email, phone
   - RETURNS: Complete customer profile with policies, claims, payments
   - EXAMPLE: "Find customer Rajesh Sharma" or "Look up customer ID 1"

3. Policy Details Tool (policy_details_tool):
   - USE FOR: Specific policy information
   - SUPPORTS: Policy number (e.g., "IFL-TERM-001") or policy ID (e.g., "101")
   - RETURNS: Policy details, customer info, claims, payments
   - EXAMPLE: "Show details for policy IFL-TERM-001"

4. Claims Status Tool (claims_status_tool):
   - USE FOR: Claim information and status
   - SUPPORTS: Claim number (e.g., "CLM-2024-001") or customer ID for all claims
   - RETURNS: Claim details, status, amounts
   - EXAMPLE: "Check claim CLM-2024-001" or "Show all claims for customer 1"

5. Payment History Tool (payment_history_tool):
   - USE FOR: Payment records and history
   - SUPPORTS: Customer ID for all payments or policy ID for specific policy
   - RETURNS: Payment history, amounts, methods, status
   - EXAMPLE: "Show payment history for customer 1" or "Payments for policy 101"

6. Legacy Customer Tool (customer_data_tool):
   - USE FOR: Backward compatibility only
   - SUPPORTS: Old customer ID format
   - NOTE: Use customer_lookup_tool for new queries

RESPONSE GUIDELINES:
1. Structure:
   - Start with a clear acknowledgment
   - Provide specific information from tools
   - End with next steps if needed

2. Professional Boundaries:
   - Access customer data only with valid identifiers
   - Never share sensitive information inappropriately
   - No financial advice or claims guarantees
   - Refer complex cases to human support
   - Stay within insurance domain
   - Politely decline non-insurance queries

3. Error Handling:
   - Acknowledge when information is not found
   - Provide alternative suggestions
   - Explain why certain requests can't be fulfilled

4. Safety and Compliance:
   - Verify customer identifiers before lookup
   - Mask sensitive data in responses
   - Follow data protection guidelines

EXAMPLE INTERACTIONS:
User: "What's term life insurance?"
Assistant: Let me search our policy documentation for term life insurance details.
[Use policy_docs_qna_tool]

User: "Find customer Rajesh Sharma"
Assistant: I'll look up the customer information for Rajesh Sharma.
[Use customer_lookup_tool]

User: "Show details for policy IFL-TERM-001"
Assistant: I'll retrieve the details for policy IFL-TERM-001.
[Use policy_details_tool]

User: "Check my claim CLM-2024-001"
Assistant: I'll check the status of claim CLM-2024-001.
[Use claims_status_tool]

User: "Show payment history for customer 1"
Assistant: I'll get the payment history for customer ID 1.
[Use payment_history_tool]

Remember: Always prioritize accuracy and use the most appropriate tool for each query type."""

class Input(TypedDict):
    session_id: str
    query: str

class Output(TypedDict):
    response: str
    metadata: Dict[str, Any]

class SessionState:
    def __init__(self):
        self.start_time = time.time()
        self.tool_usage = {"policy_docs_qna_tool": 0, "customer_data_tool": 0}
        self.error_count = 0

# Session management
sessions: Dict[str, SessionState] = {}

def get_session_state(session_id: str) -> SessionState:
    if session_id not in sessions:
        sessions[session_id] = SessionState()
    return sessions[session_id]

# Initialize checkpointer for conversation memory
checkpointer = InMemorySaver()

try:
    # Create ReAct agent with enhanced configuration
    react_agent = create_react_agent(
        llm,
        tool_functions,
        checkpointer=checkpointer,
        prompt=system_message
    )
    logger.info("Successfully initialized ReAct agent")
except Exception as e:
    logger.error(f"Failed to initialize ReAct agent: {str(e)}")
    raise

def preprocess_query(query: str) -> str:
    """Preprocess and sanitize user query"""
    return query.strip()

def validate_customer_id(query: str) -> bool:
    """Check if query contains valid customer ID format"""
    return "CUST" in query.upper() and any(c.isdigit() for c in query)

def create_session_metadata(session_state: SessionState) -> Dict[str, Any]:
    """Create session metadata for monitoring"""
    return {
        "timestamp": datetime.now().isoformat(),
        "session_duration": time.time() - session_state.start_time,
        "tool_usage": session_state.tool_usage,
        "error_count": session_state.error_count
    }

def insurance_react_agent(state: Input) -> Output:
    """Enhanced insurance ReAct agent with error handling and monitoring"""
    session_id = state["session_id"]
    query = preprocess_query(state["query"])
    
    try:
        # Get or create session state
        session_state = get_session_state(session_id)
        
        # Log incoming request
        logger.info(f"Processing query for session {session_id}: {query}")
        
        # Add customer ID validation
        if validate_customer_id(query):
            logger.info(f"Customer ID detected in query for session {session_id}")
        
        # Configure agent
        # Execute agent
        start_time = time.time()
        ai_msg = react_agent.invoke(
            {"messages": [("user", query)]},
            config={"configurable": {"thread_id": session_id}}
        )
        processing_time = time.time() - start_time
        
        # Update tool usage metrics (simplified since langgraph structure is different)
        # Tool usage tracking would need to be implemented differently
        
        # Create response
        response = ai_msg['messages'][-1].content
        metadata = create_session_metadata(session_state)
        
        # Log success
        logger.info(f"Successfully processed query for session {session_id} in {processing_time:.2f}s")
        
        return {
            "response": response,
            "metadata": metadata
        }
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = str(e)
        logger.error(f"Error processing query for session {session_id}: {error_msg}")
        
        # Update error count in session state
        session_state = get_session_state(session_id)
        session_state.error_count += 1
        
        # Return graceful error response
        return {
            "response": "I apologize, but I encountered an issue processing your request. Please try again or contact our customer service.",
            "metadata": create_session_metadata(session_state)
        }