from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocessing import process_and_index_file
from app.semantic_search_engine import search_top_chunk
from app.model_response import get_answer
from logs.logger import log_event

app = FastAPI()

RELEVENT_CHUNK_SCORE_THRESHOLD = 0.6
MODEL_RESPONSE_SCORE_THRESHOLD = 0.6

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        log_event("INFO", "Processing and indexing uploaded file has started")
        process_and_index_file(file=file, filename="uploaded_doucement")
        log_event("INFO", "File was sucsesfully processed and indixed")
    except Exception as e:
        log_event("ERROR", f"an error occured while processing and indexing uploaded file: {e}")
    return {"message": "File processed and index created", "file": file.filename}


@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    filename: str = Form(...)
):
    index_path = os.path.join(INDEX_DIR, "uploaded_doucement.index")

    if not os.path.exists(index_path):
        return JSONResponse(status_code=404, content={"error": "Index for file not found."})
    try:
        log_event("INFO", "Retriving relevant chunk has started")
        chunk, score = search_top_chunk(question=question, filename="uploaded_doucement")
        log_event("INFO", "Relevnent chunk was succsesfully retrived")
    except Exception as e:
        log_event("ERROR", f"an error occured while retriving relevant chunk: {e}")    
    
    if score < RELEVENT_CHUNK_SCORE_THRESHOLD:

        return {"answer": "No relevant answer found.", "confidence": 0.0, "score": round(score, 4)}

    result = get_answer(question, chunk)

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "score": round(score, 4),
        "chunk": chunk
    }
