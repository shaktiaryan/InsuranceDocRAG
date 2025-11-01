from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from config.settings import Settings
import os
from pydantic import SecretStr
from typing import Optional, List, Union
import requests

load_dotenv()

def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{Settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def create_vector_store(pdf_paths: Union[str, List[str]]) -> Chroma:
    """Create persistent vector store using either Ollama or Azure embeddings"""
    # Handle both single path and list of paths
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]
    
    all_documents = []
    
    # Load documents from all PDF files
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            print(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages from {os.path.basename(pdf_path)}")
        else:
            print(f"Warning: PDF file not found at {pdf_path}")

    if not all_documents:
        raise ValueError("No PDF documents found to load")

    print(f"Total pages loaded: {len(all_documents)}")
    
    # Configure text splitter based on settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Settings.CHUNK_SIZE,
        chunk_overlap=Settings.CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} text chunks")

    # Choose embedding model based on configuration
    if Settings.USE_OLLAMA_FOR_RAG and check_ollama_connection():
        print(f"Using Ollama embeddings: {Settings.OLLAMA_EMBEDDING_MODEL}")
        embeddings = OllamaEmbeddings(
            model=Settings.OLLAMA_EMBEDDING_MODEL,
            base_url=Settings.OLLAMA_BASE_URL
        )
    else:
        print("Using Azure OpenAI embeddings (fallback)")
        # Fallback to Azure embeddings
        api_key = Settings.AZURE_OPENAI_API_KEY
        endpoint = Settings.AZURE_OPENAI_ENDPOINT
        api_version = Settings.AZURE_OPENAI_API_VERSION
        
        if not api_key or not endpoint:
            raise ValueError("Neither Ollama nor Azure OpenAI embeddings are available")
        
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            azure_endpoint=endpoint,
            api_key=SecretStr(api_key),
            api_version=api_version
        )

    print("Creating vector store...")
    # Create a persistent Chroma vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vector_store"  # Store data persistently
    )
    print("Vector store created successfully!")
    
    return vector_store

# Initialize vector store with either Ollama or Azure embeddings
vector_store: Optional[Chroma] = None

print("🚀 Initializing RAG pipeline...")

# Check Ollama availability
ollama_available = check_ollama_connection()
if Settings.USE_OLLAMA_FOR_RAG and ollama_available:
    print("✅ Ollama is running - will use local embeddings")
elif Settings.USE_OLLAMA_FOR_RAG and not ollama_available:
    print("⚠️  Ollama not available - falling back to Azure embeddings")
else:
    print("📡 Using Azure OpenAI embeddings")

try:
    vector_store = create_vector_store(Settings.POLICY_DOC_PATH)
except Exception as e:
    print(f"❌ Could not create vector store: {str(e)}")
    print("📋 Will use basic policy information instead")
    vector_store = None

print("🏁 RAG pipeline initialization complete\n")

@tool
def policy_docs_qna_tool(query: str) -> str:
    """Search India First Life Insurance policy documentation for relevant information"""
    try:
        if vector_store is not None:
            # Use RAG with vector search
            print(f"🔍 Searching documents for: {query}")
            docs = vector_store.similarity_search(query, k=3)
            if docs:
                context = "\n\n".join(doc.page_content for doc in docs)
                print(f"📄 Found {len(docs)} relevant document chunks")
                return f"Based on India First Life Insurance documentation:\n\n{context}"
            else:
                print("📄 No relevant documents found")
        
        # Fallback to basic policy information
        print("📋 Using basic policy information")
        policy_info = {
            "term life": "Term Life Insurance provides pure risk cover with high coverage at affordable premiums. Coverage periods: 10-30 years, Entry age: 18-65 years, Max sum assured: Up to Rs. 2 crores",
            "endowment": "Endowment Plan combines insurance and investment with guaranteed returns. Policy terms: 10-25 years, Entry age: 5-60 years, Provides sum assured plus bonuses",
            "ulip": "ULIP (Unit Linked Insurance Plan) is market-linked investment cum insurance. Policy terms: 10+ years, Entry age: 18-60 years, Choice of equity/debt/balanced funds",
            "claim": "Claim Process: Submit claim form, death certificate, policy document within 90 days. Processing time: 15-30 working days",
            "contact": "Customer Service: Toll-free 1800-209-8700, Email: customercare@indiafirstlife.com, Website: www.indiafirstlife.com"
        }
        
        query_lower = query.lower()
        for key, info in policy_info.items():
            if key in query_lower:
                return f"India First Life Insurance - {info}"
        
        return ("I can help with information about Term Life Insurance, Endowment Plans, ULIP, claim processes, and contact details. "
               "For detailed policy documents, please visit www.indiafirstlife.com or call 1800-209-8700.")
        
    except Exception as e:
        print(f"❌ Error in policy search: {str(e)}")
        return "I encountered an issue retrieving policy information. Please contact customer service at 1800-209-8700 for assistance."

def read_insurance_database(file_path: Optional[str] = None) -> dict:
    """Read the comprehensive insurance database with all tables"""
    if file_path is None:
        # Use absolute path relative to the project root
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(script_dir, "data", "insurance_database.json")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Database file {file_path} not found.")
        return {"customers": [], "policies": [], "claims": [], "payments": []}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}.")
        return {"customers": [], "policies": [], "claims": [], "payments": []}

# Load the comprehensive insurance database
# Get the absolute path to the database file
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
insurance_db_path = os.path.join(script_dir, "data", "insurance_database.json")

if os.path.exists(insurance_db_path):
    insurance_db = read_insurance_database(insurance_db_path)
    print(f"✅ Loaded insurance database with {len(insurance_db.get('customers', []))} customers, {len(insurance_db.get('policies', []))} policies")
else:
    print(f"Warning: Insurance database not found at {insurance_db_path}")
    insurance_db = {"customers": [], "policies": [], "claims": [], "payments": []}

def _customer_lookup(customer_identifier: str) -> dict:
    """Internal function for customer lookup"""
    try:
        customers = insurance_db.get("customers", [])
        policies = insurance_db.get("policies", [])
        claims = insurance_db.get("claims", [])
        payments = insurance_db.get("payments", [])
        
        # Find customer by various identifiers
        customer = None
        identifier = customer_identifier.strip().lower()
        
        for cust in customers:
            # Check by customer_id
            if str(cust.get("customer_id")) == identifier:
                customer = cust
                break
            # Check by full name
            full_name = f"{cust.get('first_name', '')} {cust.get('last_name', '')}".lower()
            if identifier in full_name or full_name in identifier:
                customer = cust
                break
            # Check by email
            if cust.get("email", "").lower() == identifier:
                customer = cust
                break
            # Check by phone
            if cust.get("phone", "") == customer_identifier:
                customer = cust
                break
        
        if not customer:
            return {"message": "Customer not found. Please provide a valid customer ID, name, email, or phone number."}
        
        # Get customer's policies
        customer_policies = [p for p in policies if p.get("customer_id") == customer.get("customer_id")]
        
        # Get customer's claims (through policies)
        policy_ids = [p.get("policy_id") for p in customer_policies]
        customer_claims = [c for c in claims if c.get("policy_id") in policy_ids]
        
        # Get customer's payments (through policies)
        customer_payments = [p for p in payments if p.get("policy_id") in policy_ids]
        
        return {
            "customer_info": customer,
            "policies": customer_policies,
            "claims": customer_claims,
            "payments": customer_payments,
            "summary": {
                "total_policies": len(customer_policies),
                "active_policies": len([p for p in customer_policies if p.get("status") == "Active"]),
                "total_claims": len(customer_claims),
                "total_premium_paid": sum([p.get("amount", 0) for p in customer_payments if p.get("status") == "Completed"])
            }
        }
        
    except Exception as e:
        return {"error": f"Error retrieving customer information: {str(e)}"}

@tool
def customer_lookup_tool(customer_identifier: str) -> dict:
    """
    Retrieve comprehensive customer information by customer ID, name, email, or phone.
    Supports: customer_id (e.g., "1"), name (e.g., "Rajesh Sharma"), email, or phone number.
    """
    return _customer_lookup(customer_identifier)

@tool
def policy_details_tool(policy_identifier: str) -> dict:
    """
    Retrieve detailed policy information by policy number or policy ID.
    Supports: policy_number (e.g., "IFL-TERM-001") or policy_id (e.g., "101").
    """
    try:
        policies = insurance_db.get("policies", [])
        customers = insurance_db.get("customers", [])
        claims = insurance_db.get("claims", [])
        payments = insurance_db.get("payments", [])
        
        # Find policy by policy_number or policy_id
        policy = None
        identifier = policy_identifier.strip()
        
        for pol in policies:
            if pol.get("policy_number") == identifier or str(pol.get("policy_id")) == identifier:
                policy = pol
                break
        
        if not policy:
            return {"message": "Policy not found. Please provide a valid policy number or policy ID."}
        
        # Get customer information
        customer = next((c for c in customers if c.get("customer_id") == policy.get("customer_id")), None)
        
        # Get policy claims
        policy_claims = [c for c in claims if c.get("policy_id") == policy.get("policy_id")]
        
        # Get policy payments
        policy_payments = [p for p in payments if p.get("policy_id") == policy.get("policy_id")]
        
        return {
            "policy_info": policy,
            "customer_info": customer,
            "claims": policy_claims,
            "payments": policy_payments,
            "summary": {
                "policy_status": policy.get("status"),
                "premium_due": policy.get("premium_amount"),
                "coverage_amount": policy.get("coverage_amount"),
                "total_claims": len(policy_claims),
                "last_payment": max([p.get("payment_date") for p in policy_payments], default="No payments") if policy_payments else "No payments"
            }
        }
        
    except Exception as e:
        return {"error": f"Error retrieving policy information: {str(e)}"}

@tool
def claims_status_tool(claim_identifier: Optional[str] = None, customer_id: Optional[str] = None) -> dict:
    """
    Retrieve claims information by claim number/ID or all claims for a customer.
    Supports: claim_number (e.g., "CLM-2024-001"), claim_id, or customer_id for all claims.
    """
    try:
        claims = insurance_db.get("claims", [])
        policies = insurance_db.get("policies", [])
        customers = insurance_db.get("customers", [])
        
        if claim_identifier:
            # Find specific claim
            claim = None
            identifier = claim_identifier.strip()
            
            for clm in claims:
                if clm.get("claim_number") == identifier or str(clm.get("claim_id")) == identifier:
                    claim = clm
                    break
            
            if not claim:
                return {"message": "Claim not found. Please provide a valid claim number or claim ID."}
            
            # Get related policy and customer info
            policy = next((p for p in policies if p.get("policy_id") == claim.get("policy_id")), None)
            customer = next((c for c in customers if c.get("customer_id") == policy.get("customer_id")), None) if policy else None
            
            return {
                "claim_info": claim,
                "policy_info": policy,
                "customer_info": customer
            }
        
        elif customer_id:
            # Get all claims for a customer
            customer_policies = [p for p in policies if str(p.get("customer_id")) == str(customer_id)]
            policy_ids = [p.get("policy_id") for p in customer_policies]
            customer_claims = [c for c in claims if c.get("policy_id") in policy_ids]
            
            return {
                "customer_claims": customer_claims,
                "total_claims": len(customer_claims),
                "pending_claims": [c for c in customer_claims if c.get("status") in ["Under Investigation", "Pending Documentation"]],
                "approved_claims": [c for c in customer_claims if c.get("status") == "Approved"]
            }
        
        else:
            return {"message": "Please provide either a claim identifier or customer ID."}
            
    except Exception as e:
        return {"error": f"Error retrieving claims information: {str(e)}"}

@tool
def payment_history_tool(customer_id: Optional[str] = None, policy_id: Optional[str] = None) -> dict:
    """
    Retrieve payment history for a customer or specific policy.
    Supports: customer_id for all payments or policy_id for specific policy payments.
    """
    try:
        payments = insurance_db.get("payments", [])
        policies = insurance_db.get("policies", [])
        
        if policy_id:
            # Get payments for specific policy
            policy_payments = [p for p in payments if str(p.get("policy_id")) == str(policy_id)]
            policy = next((p for p in policies if str(p.get("policy_id")) == str(policy_id)), None)
            
            return {
                "policy_payments": policy_payments,
                "policy_info": policy,
                "total_paid": sum([p.get("amount", 0) for p in policy_payments if p.get("status") == "Completed"]),
                "pending_payments": [p for p in policy_payments if p.get("status") == "Pending"],
                "failed_payments": [p for p in policy_payments if p.get("status") == "Failed"]
            }
        
        elif customer_id:
            # Get all payments for customer
            customer_policies = [p for p in policies if str(p.get("customer_id")) == str(customer_id)]
            policy_ids = [p.get("policy_id") for p in customer_policies]
            customer_payments = [p for p in payments if p.get("policy_id") in policy_ids]
            
            return {
                "customer_payments": customer_payments,
                "total_paid": sum([p.get("amount", 0) for p in customer_payments if p.get("status") == "Completed"]),
                "recent_payments": sorted(customer_payments, key=lambda x: x.get("payment_date", ""), reverse=True)[:5],
                "payment_methods": list(set([p.get("payment_method") for p in customer_payments]))
            }
        
        else:
            return {"message": "Please provide either a customer ID or policy ID."}
            
    except Exception as e:
        return {"error": f"Error retrieving payment information: {str(e)}"}

# Keep the original customer_data_tool for backward compatibility with old JSON format
def read_customer_data(file_path: str = "customers.json") -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}.")
        return {}

# Check if old customer data file exists for backward compatibility
if os.path.exists(Settings.CUSTOMER_DATA_PATH):
    customer_data = read_customer_data(Settings.CUSTOMER_DATA_PATH)
else:
    print(f"Info: Using new database format, old customer data file not found at {Settings.CUSTOMER_DATA_PATH}")
    customer_data = {}

@tool
def customer_data_tool(customer_id: str) -> dict:
    """Legacy tool: Retrieve customer policy details and information for a given customer ID (backward compatibility)"""
    # Try new database first by calling the internal function
    result = _customer_lookup(customer_id)
    if "message" not in result:
        return result
    
    # Fallback to old format
    return customer_data.get(customer_id, {"message": "Customer not found"})