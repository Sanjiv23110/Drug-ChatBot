# Solomind Drug Chatbot

An AI-powered chatbot providing accurate drug information from Health Canada monographs using Azure OpenAI and RAG (Retrieval-Augmented Generation) architecture.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue.svg)](https://www.typescriptlang.org/)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

## 📋 Overview

This application provides pharmacists and healthcare professionals with instant access to comprehensive drug information sourced from Health Canada's official Drug Product Database. Using state-of-the-art AI technology, it delivers accurate, cited answers to drug-related queries in under 2 seconds.

### Key Features

- **🤖 AI-Powered Search**: GPT-4o language model with semantic understanding
- **📚 Official Data**: 30,000+ Health Canada drug monographs
- **⚡ Fast Response**: Sub-2 second query response time
- **🔍 Semantic Search**: ChromaDB vector database for intelligent retrieval
- **📖 Source Citations**: Every answer includes document references
- **⚕️ Medical Disclaimers**: PIPEDA-compliant legal protection
- **📊 Analytics**: Google Analytics 4 integration (optional)
- **🔐 Secure**: Enterprise-grade security with Azure OpenAI

## 🏗️ Architecture

```
┌─────────────────┐
│   React UI      │  Frontend (TypeScript + React)
│  localhost:5173 │  - Chat interface
└────────┬────────┘  - Disclaimer modal
         │           - Real-time responses
         │
         ▼ HTTP/REST
┌─────────────────┐
│  FastAPI        │  Backend (Python)
│  localhost:8000 │  - RAG Service
└────────┬────────┘  - Vector Store
         │           - API Endpoints
         │
         ├─────────────────┬──────────────────┐
         ▼                 ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌─────────────┐
│ Azure OpenAI  │  │  ChromaDB    │  │ Health      │
│               │  │              │  │ Canada PDFs │
│ - GPT-4o      │  │ - Embeddings │  │ - Source    │
│ - Embeddings  │  │ - Metadata   │  │   Data      │
└───────────────┘  └──────────────┘  └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- Azure OpenAI account with deployments:
  - `text-embedding-ada-002` (embeddings)
  - `gpt-4o-agent` or `gpt-4o` (chat)
- Health Canada PDF monographs

### Installation

#### 1. Clone Repository

```bash
git clone https://github.com/Sanjiv23110/Drug-ChatBot.git
cd Drug-ChatBot
```

#### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

**Edit `.env` file:**

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-agent
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Database Configuration
CHROMA_DB_DIR=C:\G\Maclens chatbot w api\chroma_db
DOCUMENTS_DIR=C:\G\chatbot maclens\data

# Optional: Error Monitoring
SENTRY_DSN=
ENVIRONMENT=development
```

#### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Optional: Configure Google Analytics
cp .env.example .env
# Edit .env and add: VITE_GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

#### 4. Data Ingestion

Place PDF files in your `DOCUMENTS_DIR` folder, then run:

```bash
cd ../backend
venv\Scripts\python.exe scripts\ingest.py
```

**Expected Output:**
```
Ingesting from: C:\G\chatbot maclens\data
Found 777 document chunks.
Adding 777 documents in batches of 10...
Processed batch 1/78
...
Processed batch 78/78
Finished adding documents to ChromaDB.
Ingestion complete.
```

#### 5. Run Application

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Access the app:** http://localhost:5173

## 📁 Project Structure

```
Drug-ChatBot/
├── backend/                      # FastAPI backend
│   ├── app/
│   │   ├── api/
│   │   │   └── endpoints.py      # REST API routes
│   │   ├── core/
│   │   │   └── config.py         # Settings & environment vars
│   │   └── services/
│   │       ├── ingestion_service.py  # PDF processing
│   │       ├── rag_service.py        # RAG logic
│   │       └── vector_store.py       # ChromaDB interface
│   ├── scripts/
│   │   └── ingest.py             # Data ingestion script
│   ├── main.py                   # FastAPI app entry point
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment template
│   └── Dockerfile                # Container config
│
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx          # Main chat interface
│   │   │   └── DisclaimerModal.tsx  # Legal disclaimer
│   │   ├── App.tsx               # Root component
│   │   └── main.tsx              # Entry point
│   ├── package.json              # Node dependencies
│   ├── .env.example              # Optional config
│   └── Dockerfile                # Container config
│
├── chroma_db/                    # Vector database (gitignored)
├── docker-compose.yml            # Multi-container orchestration
├── .gitignore                    # Git exclusions
└── README.md                     # This file
```

## 🔧 Core Technologies

### Backend Stack

- **FastAPI** - Modern Python web framework
- **Azure OpenAI** - GPT-4o for chat, text-embedding-ada-002 for embeddings
- **ChromaDB** - Vector database for semantic search
- **Pydantic** - Data validation and settings management
- **PyPDF** - PDF text extraction
- **Uvicorn** - ASGI server

### Frontend Stack

- **React 18** - UI framework
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Google Analytics 4** - Usage tracking (optional)

## 📊 API Documentation

### Endpoints

#### `POST /api/chat`

Send a question and receive an AI-generated answer.

**Request:**
```json
{
  "message": "What are the side effects of Lipitor?"
}
```

**Response:**
```json
{
  "answer": "Common side effects of Lipitor include...",
  "sources": [
    "lipitor_monograph_2024.pdf",
    "atorvastatin_health_canada.pdf"
  ]
}
```

#### `GET /health`

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "model": "gpt-4o-agent"
}
```

#### `GET /api/stats`

Get database statistics.

**Response:**
```json
{
  "document_count": 3070,
  "data_path": "C:\\G\\chatbot maclens\\data"
}
```

## 🔐 Environment Variables

### Backend (.env)

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `AZURE_OPENAI_API_KEY` | ✅ | Azure OpenAI API key | `1234abcd...` |
| `AZURE_OPENAI_ENDPOINT` | ✅ | Azure OpenAI endpoint URL | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | ✅ | Embedding model deployment | `text-embedding-ada-002` |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | ✅ | Chat model deployment | `gpt-4o-agent` |
| `AZURE_OPENAI_API_VERSION` | ✅ | API version | `2024-12-01-preview` |
| `CHROMA_DB_DIR` | ✅ | Vector database directory | `C:\G\Maclens chatbot w api\chroma_db` |
| `DOCUMENTS_DIR` | ✅ | PDF source directory | `C:\G\chatbot maclens\data` |
| `SENTRY_DSN` | ❌ | Error monitoring DSN | `https://...@sentry.io/...` |
| `ENVIRONMENT` | ❌ | Environment name | `development` |

### Frontend (.env - Optional)

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `VITE_GA_MEASUREMENT_ID` | ❌ | Google Analytics ID | `G-XXXXXXXXXX` |

## 🧪 Testing

### Manual Testing

```bash
# Test backend
cd backend
venv\Scripts\activate
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is aspirin used for?"}'

# Test health endpoint
curl http://localhost:8000/health
```

### Testing Queries

Example questions to test:
- "What are the contraindications for Lipitor?"
- "What is the recommended dosage for metformin?"
- "Can pregnant women take acetaminophen?"
- "What are the drug interactions for warfarin?"

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Build images
docker-compose build

# Run containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

**Access:**
- Frontend: http://localhost:3000
- Backend: http://localhost:8000

## 👥 Team Collaboration

### Git Workflow

1. **Clone repository**
   ```bash
   git clone https://github.com/Sanjiv23110/Drug-ChatBot.git
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "Add: descriptive commit message"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Important: Never Commit

- ❌ `.env` files (contain API keys!)
- ❌ `chroma_db/` folder (large database)
- ❌ `node_modules/` (dependencies)
- ❌ `venv/` (Python virtual environment)
- ❌ PDF data files

The `.gitignore` file handles this automatically.

### Sharing Credentials

**Never commit API keys to Git!**

- Share `.env.example` as a template
- Use secure channels for actual credentials (password manager, encrypted messaging)
- Each team member creates their own `.env` file locally

## 📈 Performance

- **Query Response Time:** < 2 seconds
- **Embedding Generation:** ~100ms per document chunk
- **Vector Search:** ~50ms for 3 results from 3,000+ chunks
- **API Latency:** ~1.5s (including Azure OpenAI round-trip)

## 🔍 Troubleshooting

### Backend Won't Start

**Error:** `ModuleNotFoundError: No module named 'openai'`

**Solution:**
```bash
cd backend
venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Build Fails

**Error:** `Cannot find module 'react'`

**Solution:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Database Empty After Ingestion

**Check:**
```bash
cd backend
venv\Scripts\python.exe -c "import chromadb; c = chromadb.PersistentClient(path='C:\\G\\Maclens chatbot w api\\chroma_db'); print(c.list_collections())"
```

**Re-run ingestion if needed:**
```bash
venv\Scripts\python.exe scripts\ingest.py
```

### API Connection Errors

**Check backend is running:**
```bash
curl http://localhost:8000/health
```

**Check CORS configuration in `main.py`**

## 🔗 Resources

- **Health Canada Drug Database:** https://health-products.canada.ca/dpd-bdpp/?lang=eng
- **Azure OpenAI Documentation:** https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **ChromaDB Documentation:** https://docs.trychroma.com/
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **React Documentation:** https://react.dev/

## 📝 License

Proprietary - Solomind Health Technologies. All rights reserved.

## 🤝 Contributing

This is a private repository. For team members:

1. Follow the Git workflow above
2. Write clear commit messages
3. Test locally before pushing
4. Request code reviews for all PRs
5. Keep dependencies up to date

## 💬 Support

For technical support or questions:
- **GitHub Issues:** Use for bug reports and feature requests
- **Team Chat:** [Slack/Discord channel]
- **Documentation:** Refer to inline code comments

---

**Built with ❤️ by the Solomind Team**
