"""
Configuration settings for Agent-Bot
"""
import os
from dotenv import load_dotenv

# Load environment variables from root .env file
load_dotenv()

class Settings:
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8080
    API_TITLE = "Agent-Bot API"
    CORS_ORIGINS = ["*"]
    
    # Data Paths
    POLICY_DOC_PATH = ["data/indiafirstlifeassuredincomeformilestonesplan-brochure-low1.pdf",
    "data/indiafirst-life-guaranteed-single-premium-plan-143n068v04.pdf", 
    "data/life-insurance-ki-kitaab-english.pdf"]
    CUSTOMER_DATA_PATH = "data/customer_data.json"
    
    # Azure OpenAI Configuration (for main chat)
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    
    # Ollama Configuration (for RAG pipeline)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
    
    # RAG Configuration
    USE_OLLAMA_FOR_RAG = os.getenv("USE_OLLAMA_FOR_RAG", "true").lower() == "true"
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Legacy OpenAI (kept for backward compatibility)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
