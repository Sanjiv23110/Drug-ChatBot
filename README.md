# Solomind Drug Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot providing accurate drug information from Health Canada monographs.

## 🚀 Features

- **Azure OpenAI Integration**: GPT-4o for chat, text-embedding-ada-002 for embeddings
- **Health Canada Data**: Official drug monographs from Canadian government database
- **Vector Search**: ChromaDB for fast semantic search
- **Medical Disclaimers**: Legal protection with modal + warning banner
- **Modern UI**: React + TypeScript frontend with real-time chat
- **Production Ready**: Error monitoring (Sentry), analytics (Google Analytics), health checks

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- Azure OpenAI account with deployed models
- Health Canada PDF monographs

## 🛠️ Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd "Maclens chatbot w api"
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Azure OpenAI credentials:
# - AZURE_OPENAI_API_KEY
# - AZURE_OPENAI_ENDPOINT
# - AZURE_OPENAI_EMBEDDING_DEPLOYMENT
# - AZURE_OPENAI_CHAT_DEPLOYMENT
```

### 3. Frontend Setup

```bash
cd frontend
npm install

# Configure environment (optional)
cp .env.example .env
# Add Google Analytics ID if desired
```

### 4. Data Ingestion

```bash
cd backend

# Place PDF files in: C:\G\chatbot maclens\data\
# Or update DOCUMENTS_DIR in .env

# Run ingestion
venv\Scripts\python.exe scripts\ingest.py
```

### 5. Run Application

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Access: http://localhost:5173

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React UI  │─────▶│  FastAPI     │─────▶│ Azure       │
│  (Port 5173)│      │  Backend     │      │ OpenAI      │
└─────────────┘      │  (Port 8000) │      └─────────────┘
                     └──────┬───────┘
                            │
                            ▼
                     ┌─────────────┐
                     │  ChromaDB   │
                     │  Vector DB  │
                     └─────────────┘
```

## 📁 Project Structure

```
Maclens chatbot w api/
├── backend/
│   ├── app/
│   │   ├── api/endpoints.py       # API routes
│   │   ├── core/config.py         # Settings
│   │   └── services/
│   │       ├── rag_service.py     # RAG logic
│   │       ├── vector_store.py    # ChromaDB
│   │       └── ingestion_service.py
│   ├── scripts/
│   │   └── ingest.py              # Data ingestion
│   ├── main.py                    # FastAPI app
│   ├── requirements.txt
│   └── .env                       # API keys (GITIGNORED)
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.tsx           # Main chat UI
│   │   │   └── DisclaimerModal.tsx
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── .env                       # Optional config (GITIGNORED)
│
└── chroma_db/                     # Vector DB (GITIGNORED)
```

## 🔑 Environment Variables

### Backend (.env)

```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-agent
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Database
CHROMA_DB_DIR=C:\G\Maclens chatbot w api\chroma_db
DOCUMENTS_DIR=C:\G\chatbot maclens\data

# Optional Monitoring
SENTRY_DSN=
ENVIRONMENT=development
```

### Frontend (.env - Optional)

```bash
VITE_GA_MEASUREMENT_ID=G-XXXXXXXXXX
```

## 👥 Team Collaboration

### Git Workflow

1. **Clone the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes and commit:**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```
4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Important Notes

⚠️ **NEVER commit:**
- `.env` files (contain API keys!)
- `chroma_db/` folder (large database files)
- `data/` folder (PDF files)
- `node_modules/` or `venv/`

✅ **Do commit:**
- Source code (.py, .tsx, .ts)
- Configuration templates (.env.example)
- Documentation
- Requirements files

## 🔐 Security

- All API keys in `.env` (gitignored)
- Medical disclaimers shown on first use
- Persistent warning banner
- Azure OpenAI for HIPAA compliance

## 📊 Monitoring

- **Sentry**: Error tracking (set SENTRY_DSN)
- **Google Analytics**: Usage metrics (set VITE_GA_MEASUREMENT_ID)
- **Health Endpoint**: `/health` for uptime monitoring

## 🚀 Deployment

1. **Database:** Pre-ingest all PDFs, include `chroma_db/` in deployment
2. **Backend:** Deploy FastAPI to cloud (Azure, AWS, DigitalOcean)
3. **Frontend:** Build and deploy to CDN/hosting
4. **Environment:** Set all production environment variables

## 📝 License

Proprietary - Solomind Health Technologies

## 🔗 Resources

- [Health Canada Drug Database](https://health-products.canada.ca/dpd-bdpp/?lang=eng)
- [Azure OpenAI Docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [ChromaDB Docs](https://docs.trychroma.com/)

---

**For support:** Contact development team
