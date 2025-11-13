# 🤖 RAG Chatbot - Microfinance Document Q&A

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with LangChain, FastAPI, and PostgreSQL/PGVector for microfinance business document queries.

## 📋 Features

- **Multi-Provider LLM Support**: Switch between Google Gemini and Fireworks AI
- **Vector Search**: Semantic search using PGVector and HuggingFace embeddings
- **Conversational Memory**: Session-based chat history with LangChain
- **Document Ingestion**: Support for PDF and TXT files
- **Abuse Detection**: Built-in content filtering
- **Scope Control**: Automatic out-of-scope query handling
- **FastAPI Backend**: High-performance async API with CORS support
- **Streamlit Frontend**: Simple, interactive chat interface

## 🏗️ Project Structure

```
CHATBOT/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes_chat.py          # Chat & debug endpoints
│   ├── core/
│   │   └── config.py                # Configuration management
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chains.py                # LangChain RAG pipeline
│   │   └── retriever.py             # PGVector retriever setup
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingest_pgvector.py       # Document ingestion script
│   │   ├── llm_fireworks.py         # Fireworks AI integration
│   │   └── llm_gemini.py            # Google Gemini integration
│   ├── main.py                      # FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── app.py                       # Streamlit UI
│   └── requirements.txt
├── data/
│   ├── Business Indicator MFI 2.pdf
│   └── data.txt
├── .env                             # Environment variables
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL 14+ with pgvector extension
- Google Gemini API key (or Fireworks AI API key)

### 1. Database Setup

```bash
# Install PostgreSQL and create database
createdb ai_docs

# Enable pgvector extension
psql ai_docs -c "CREATE EXTENSION vector;"
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
# Database
PG_DSN=postgresql+psycopg://postgres:yourpassword@localhost:5432/ai_docs

# LLM Provider (Google Gemini)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.2

# Optional: Fireworks AI
FIREWORKS_API_KEY=your_fireworks_key
FIREWORKS_MODEL=accounts/fireworks/models/mixtral-8x7b-instruct
FIREWORKS_TEMPERATURE=0.2

# Vector Store
COLLECTION=business_docs
EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Frontend (optional)
FRONTEND_ORIGIN=http://localhost:8501
```

### 3. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
```

### 4. Ingest Documents

Add your documents to the `data/` folder, then run:

```bash
# Single file
python -m backend.services.ingest_pgvector data/data.txt

# Multiple files
python -m backend.services.ingest_pgvector data/*.pdf data/*.txt

# Specific files
python -m backend.services.ingest_pgvector "data/Business Indicator MFI 2.pdf"
```

### 5. Start the Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

API will be available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### 6. Start the Frontend

```bash
cd frontend
streamlit run app.py
```

Access the chat interface at `http://localhost:8501`

## 🔧 API Endpoints

### `POST /api/chat`
Main chat endpoint with conversational memory.

**Request:**
```json
{
  "message": "What is the meaning of MFI?",
  "session_id": "user-123",
  "provider": "gemini"
}
```

**Response:**
```json
{
  "answer": "MFI stands for Microfinance Institution..."
}
```

### `POST /api/chat_debug`
Debug endpoint to view retrieved document chunks.

**Request:**
```json
{
  "message": "loan policies"
}
```

**Response:**
```json
{
  "chunks": [
    "First 300 chars of chunk 1...",
    "First 300 chars of chunk 2..."
  ]
}
```

### `GET /health`
Health check endpoint.

## ⚙️ Configuration

### Retrieval Settings

In `backend/rag/retriever.py`:
- `SCORE_THRESHOLD`: Minimum similarity score (0.35 default)
- `k`: Number of chunks to retrieve (4 default)

### Chunking Settings

In `backend/services/ingest_pgvector.py`:
- `chunk_size`: 800 characters
- `chunk_overlap`: 120 characters

### Abuse Filter

Customize the abuse word list in `backend/api/routes_chat.py`:
```python
ABUSIVE_WORDS = {
    "idiot", "stupid", "nonsense", ...
}
```

## 🛡️ Features Explained

### Out-of-Scope Handling
The system automatically detects and rejects queries unrelated to microfinance:
- Uses similarity score thresholding
- Returns: _"That topic is outside my scope. I can help with MFI Business Document related information."_

### Conversational Memory
- Session-based chat history using `RunnableWithMessageHistory`
- In-memory storage (can be extended to Redis/database)
- Each user gets isolated conversation context

### Multi-Provider Support
Switch LLM providers on-the-fly:
```python
# Use Gemini (default)
{"message": "...", "provider": "gemini"}

# Use Fireworks AI
{"message": "...", "provider": "fireworks"}
```

## 📊 Performance Tuning

1. **Adjust chunk size**: Balance between context and precision
2. **Tune score threshold**: Higher = stricter relevance filtering
3. **Increase k value**: Retrieve more chunks (may increase noise)
4. **Use faster embeddings**: Switch to lighter models for speed
5. **Enable caching**: Add Redis for repeated queries

## 🐛 Troubleshooting

**Vector store not found:**
```bash
# Ensure documents are ingested
python -m backend.services.ingest_pgvector data/*.pdf
```

**PGVector connection error:**
```bash
# Verify database and extension
psql ai_docs -c "\dx"  # Should show 'vector' extension
```

**LLM API errors:**
- Check API keys in `.env`
- Verify model names are correct
- Ensure sufficient API credits

## 📝 License

MIT License - feel free to use for your projects!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with:** LangChain • FastAPI • PostgreSQL • PGVector • Streamlit • Google Gemini • Fireworks AI
