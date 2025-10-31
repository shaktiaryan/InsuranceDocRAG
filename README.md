# Insurance Policy ReAct Agent with RAG Pipeline

## Project Overview
This project implements a sophisticated ReAct (Actor-Critic) Agent enhanced with a Retrieval Augmented Generation (RAG) pipeline for handling insurance policy queries. The system combines the reasoning capabilities of a Large Language Model (LLM) with the precision of retrieved context from insurance documentation.

## Key Features
- 🤖 ReAct Architecture (Reasoning and Acting framework)
- 📚 RAG Pipeline with local Ollama embeddings
- 🔍 Vector search for policy documents
- 💾 Customer data management
- 🤝 Azure OpenAI integration

## Architecture
```
User Query ──► ReAct Agent (Azure OpenAI)
                    │
                    ▼
            Reasoning Phase
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
    RAG Pipeline          Customer Data
        │                       │
        ▼                       ▼
   Local Ollama          JSON Database
   Embeddings                   │
        │                       │
        ▼                       ▼
Vector Search ◄─────── Tool Selection
        │                       │
        └─────────►  Response Generation
                            │
                            ▼
                    User Response
```

## Prerequisites
- Python 3.8+
- Ollama installed and running
- Azure OpenAI API access
- PDF policy documents
- Customer data JSON

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shaktiaryan/InsuranceDocRAG.git
cd InsuranceDocRAG
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

5. Start Ollama service (if not running)

6. Run the application:
```bash
python main.py
```

## Configuration
Create a `.env` file with the following configurations:
```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2024-10-01-preview
MODEL_NAME=gpt-4o

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## Project Structure
```
InsuranceDocRAG/
├── acs_agents/
│   ├── __init__.py
│   ├── react_agent.py     # Main conversation handler
│   └── tools.py           # RAG and utility tools
├── config/
│   ├── __init__.py
│   └── settings.py        # Configuration management
├── data/
│   ├── customer_data.json
│   └── policy_docs/*.pdf
├── docs/
│   └── technical_documentation.md
├── main.py               # Application entry point
├── requirements.txt      # Dependencies
└── .env                 # Environment configuration
```

## Documentation
For detailed technical documentation, please refer to [docs/technical_documentation.md](docs/technical_documentation.md).

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- LangChain for the ReAct agent framework
- Ollama for local embeddings
- Azure OpenAI for LLM capabilities