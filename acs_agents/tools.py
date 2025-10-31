from langchain_core.vectorstores import InMemoryVectorStore
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

def create_in_memory_vector_store(pdf_paths: Union[str, List[str]]) -> InMemoryVectorStore:
    """Create vector store using either Ollama or Azure embeddings"""
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
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    print("Vector store created successfully!")
    
    return vector_store

# Initialize vector store with either Ollama or Azure embeddings
vector_store: Optional[InMemoryVectorStore] = None

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
    vector_store = create_in_memory_vector_store(Settings.POLICY_DOC_PATH)
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

# Check if customer data file exists before reading
if os.path.exists(Settings.CUSTOMER_DATA_PATH):
    customer_data = read_customer_data(Settings.CUSTOMER_DATA_PATH)
else:
    print(f"Warning: Customer data file not found at {Settings.CUSTOMER_DATA_PATH}")
    customer_data = {}

@tool
def customer_data_tool(customer_id: str) -> dict:
    """Retrieve customer policy details and information for a given customer ID"""
    return customer_data.get(customer_id, {"message": "Customer not found"})