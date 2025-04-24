## 📄 Advanced Question Answering System (Document-Based QA)

A full-stack NLP system that extracts precise answers from large unstructured documents (PDF, DOCX, TXT) using semantic search and a fine-tuned transformer model.

> **Built with:** FastAPI, Hugging Face Transformers, FAISS, Streamlit, Docker  
> **Deployed on:** fly.io (API)

---

### ✅ Features

- Upload a document (PDF, DOCX, or TXT)
- Ask natural language questions about its content
- Retrieves the most relevant chunk using **semantic search (FAISS + Sentence-BERT)**
- Extracts answers using a **fine-tuned RoBERTa model on SQuAD2** "with less training epochs due to resources constraints"
- Returns:
  - Final Answer
  - Confidence Score
  - Similarity Score
  - Source Chunk

---

### 📦 Tech Stack

| Layer        | Tools / Frameworks |
|--------------|--------------------|
| **Model**    | RoBERTa-base fine-tuned on SQuAD2 (via Hugging Face API) |
| **Retrieval**| Sentence-BERT + FAISS |
| **Backend**  | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Deployment** | Docker + Render |
| **Logging**  | Python logging module |
| **CI/CD**    | GitHub Actions + Render Deploy Hook |

---

### 🚀 Demo

**[Live Demo (Streamlit UI)](https://custom-q-a-system.streamlit.app/)**  

> Upload a document, ask a question, and get a precise answer instantly!

---

### 🧠 How It Works

1. **Document Upload**
   - Preprocessed (cleaned, chunked, embedded)
   - Stored as FAISS vector index

2. **Question Handling**
   - User submits a question
   - Question is embedded and matched with top document chunks

3. **Answer Extraction**
   - Best chunk is sent to the QA model
   - Returns answer + confidence score

---

### 🛠 Installation

```bash
git clone https://github.com/yourusername/end-to-end-qa
cd end-to-end-qa
docker build -t qa-app .
docker run -p 8000:8000 qa-app
```

---

### 🧪 API Endpoints

#### `POST /upload`
Upload a document file.  
**Body:** `multipart/form-data`  
**Returns:** JSON confirmation and index path

#### `POST /ask`
Ask a question about the uploaded document.  
**Body:** `form-data`  
- `question`: your natural language question  
- `filename`: uploaded file name  

**Returns:**  
```json
{
  "answer": "Employees are eligible after 6 months.",
  "confidence": 92.4,
  "score": 0.87,
  "chunk": "According to the policy, employees become eligible..."
}
```

---

### ⚠️ Notes & Limitations

- Currently supports **1 document at a time** (MVP constraint)
- Uses Hugging Face Inference API (rate limits apply on free tier)
- RoBERTa model may take 1–2 seconds per request due to cold starts

---

### 📂 Folder Structure
```
.
├── app/                  # FastAPI logic
├── ui/                   # Streamlit frontend
├── utils/                # File/text helpers
├── logs/                 # Logging system
├── data/                 # Uploaded files and FAISS index
├── Dockerfile            # Deployment config
├── README.md
├── requirements.txt
```

---

### ✍️ Author

**Mohannad Karim**  
NLP & Machine Learning Engineer | MLOps 
[Upwork](https://www.upwork.com/freelancers/~01683e506def8e06a2?mp_source=share)
