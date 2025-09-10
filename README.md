# Multimodal Conversational System — RAG + Vision

A smart multimodal assistant that combines image understanding and document Q&A. Queries are dynamically routed between a Vision Agent and a RAG Agent using LangGraph and LangChain, with persistent session memory and automatic S3 sync.

Features:
- Context-aware responses across sessions

- Structured logging for easy monitoring

- Seamless handling of both documents and images

Your all-in-one solution for intelligent, multimodal interactions.

**Demo Video**:     [Watch the demo (.webm)](./demo_vision_rag.webm)  
**Vision example**: [VLM Example](./vision_example.jpg) 
**RAFG example**:   [RAG Example](./rag_example.jpg) 


![Demo Vision RAG](demo_vision_rag.gif)


---

## ⚙️ API Key Configuration
To work properly, a Gemini API key is required.
Create a .env file in the project root with the following content:

[Gemini API Key Setup](https://console.cloud.google.com/apis/api/)

- Click Create credentials → API Key
- Copy the key and add it to your `.env` file:

  ```bash
  # Required: Gemini API Key
  GEMINI_API_KEY="la_tua_api_key"

  # Optional: for automatic S3 synchronization
  AWS_ACCESS_KEY_ID="la_tua_access_key"
  AWS_SECRET_ACCESS_KEY="la_tua_secret_key"
  ```

## 🔍 Main Features

- ✅ **LLM (Gemini API)**
  - Uses **Gemini (via API or local model)** as the main language engine
  - Optimized prompts for Q&A and summarization on documents and images
  - Handles multiple documents with “source” metadata

- 📄 **Document Retrieval (RAG)**
  - Supports formats: PDF, TXT, DOCX, CSV
  - **FAISS** vectorstore updated with every upload
  - Intelligent routing to the Rag agent only when needed

- 🖼️ **Vision & Image Q&A**
  - Supports image formats: PNG, JPG, JPEG, BMP, WEBP
  - Calls **Gemini VLM** for image-based questions
  - Automatic routing to the Vision agent when an image is detected

- 💾 **Persistent Conversational Memory**
  - Each session identified by a `UUID`
  - **RAM + disk persistence** for each user
  - Context-aware retrieval using **retriever with history**
  - Memory updated and saved at each turn

- 🌐 **REST API (FastAPI)**
  - `POST /init_session` → creates a new session UUID
  - `POST /upload` → uploads documents/images, updates vectorstore and graph
  - `POST /chat` → interacts with the multimodal system (RAG + Vision + fallback small-talk)
  - `POST /reset` → resets memory, graph, and session cache
  - JSON responses with `session_id`, `agent`, `response`, `elapsed_time`

- 📈 **Intelligent Agent Routing**
  - Priority: Vision (images) → Technical/RAG (documents) → General fallback
  - Pattern-first for immediate responses without calling the model
  - Configurable timeout for LLM/VLM to prevent blocking

- 💻 **Chainlit Frontend**
  - Simple interface to test the chatbot
  - Quick startup from terminal
  - Connects directly to the FastAPI backend

- ☁️ **Optional S3 Synchronization**
  - Logs and documents automatically synced on startup/shutdown
  - On-demand sync after uploads
  - AWS keys configurable via `.env` (optional)

- 📝 **Advanced Logging and Diagnostics**
  - Session-based logger tracking inputs/outputs
  - Monitors response times, memory state, and chat content
  - Detailed debugging available with `DEBUG` level


---

## 📂 Project Structure
```bash
code/
├── app.py          # FastAPI backend with all REST endpoints
├── app_frontend.py # Chainlit frontend for chatbot testing
├── config.py       # Global configuration and constants
├── utils/          # Utility functions (sessions, hashing, logging)
├── memory/         # Persistent conversational memory
├── rag/            # FAISS vectorstore and RAG chain for document retrieval
├── agents/         # LangGraph construction for agent routing
├── loader/         # LLM and VLM loader (local or Gemini API)
├── models_e_docs/  # AI models and user-uploaded documents
├── app_frontend.py # Chainlit frontend for chatbot testing
└── storage/        # Utilities for optional AWS S3 synchronization
```


## 🐳 Quick Start — Docker Compose (Containerized)

If you prefer containerized setup, Docker Compose allows running backend + frontend in a isolated container.

1️⃣ Clone the repository
```bash
git clone https://github.com/acapuccio-2303/AI_multiagents_rag_vision.git
cd AI_multiagents_rag_vision

```

2️⃣ Make sure Docker & Docker Compose are installed
```bash
docker --version
docker-compose --version
```

2️⃣ Create a .env file

Create a .env file in the root of the project with your Gemini API key and optionally AWS credentials for S3 sync:

[Gemini API Key Setup](https://console.cloud.google.com/apis/api/)

- Click Create credentials → API Key
- Copy the key and add it to your `.env` file:

  ```bash
  # Required: Gemini API Key
  GEMINI_API_KEY="la_tua_api_key"

  # Optional: for automatic S3 synchronization
  AWS_ACCESS_KEY_ID="la_tua_access_key"
  AWS_SECRET_ACCESS_KEY="la_tua_secret_key"
  ```

3️⃣ Build and start container (from project root)
```bash
docker-compose up --build
```

4️⃣ Access services

**Backend FastAPI**: [http://localhost:8000](http://localhost:8000)  
**Chainlit Frontend — start chatting here**: [http://localhost:8501](http://localhost:8501)


5️⃣ Stop container
```bash
docker-compose down
```

---


## ▶️ Local Installation

1️⃣ Clone the repository:
```bash
git clone https://github.com/acapuccio-2303/AI_multiagents_rag_vision.git
cd AI_multiagents_rag_vision
```

2️⃣ Create a virtual environment and install dependencies:

Linux / Mac
```bash
python -m venv venv
source venv/bin/activate
```

Windows
```bash
python -m venv venv
venv\Scripts\activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

### From the project root
3️⃣ Start the FastAPI backend:
```bash
uvicorn code.app:app --reload --host 127.0.0.1 --port 8000 --log-level critical
```

4️⃣ Start the Chainlit frontend:
Per testare la chatbot da interfaccia
```bash
chainlit run code/app_frontend.py -w --host 127.0.0.1 --port 8501
```

---


### ✅ Notes

- Ensure .env is present with Gemini API key before running.
- Docker Compose automatically builds and runs both backend and frontend.
- Persistent memory and uploaded documents remain in models_e_docs/ even when containers are restarted.
- Use LOG_LEVEL=DEBUG in config.py for detailed diagnostic output.

