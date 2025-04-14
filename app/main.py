from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocessing import process_and_index_file
from app.semantic_search_engine import search_top_chunk
from app.model_response import get_answer
from logs.logger import log_event

app = FastAPI()

RELEVENT_CHUNK_SCORE_THRESHOLD = 0.7
MODEL_RESPONSE_SCORE_THRESHOLD = 65

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        log_event("INFO", "Processing and indexing uploaded file has started")
        filename_parts = file.filename.split(".")
        ext = ""
        ext = filename_parts[-1]

        await process_and_index_file(file=file, filename=f"uploaded_file.{ext}")

        index_path = os.path.join("data", "index", "uploaded_file.index")

        if not os.path.exists(index_path):
            return JSONResponse(status_code=404, content={"error": "File upload and processing was not sucessful, try again later."})


        log_event("INFO", "File was successfully processed and indexed")
    except Exception as e:
        log_event("ERROR", f"An error occurred while processing and indexing uploaded file: {e}")
    return {"message": "File processed and index created", "file": file.filename}


@app.post("/ask")
async def ask_question(
    question: str = Form(...)
):
    filename = "uploaded_document"
    index_path = os.path.join("data", "index", "uploaded_file.index")

    if not os.path.exists(index_path):
        return JSONResponse(status_code=404, content={"error": "Index for file not found."})

    try:
        log_event("INFO", "Retrieving relevant chunk has started")
        result = search_top_chunk(question=question, filename=filename)
        chunk = result["chunk"]
        score = result["score"]
        log_event("INFO", "Relevant chunk was successfully retrieved")
    except Exception as e:
        log_event("ERROR", f"An error occurred while retrieving relevant chunk: {e}")
        return {"answer": "error retrieving chunk", "confidence": 0.0, "score": 0.0}

    if score < RELEVENT_CHUNK_SCORE_THRESHOLD:
        return {"answer": "no text found", "confidence": 0.0, "score": round(score, 4)}

    try:
        log_event("INFO", "Generating the answer from the model")
        result = get_answer(question, chunk)
        log_event("INFO", "The answer was successfully generated")
    except Exception as e:
        log_event("ERROR", f"An error occurred while generating answer: {e}")
        return {"answer": "error generating answer", "confidence": 0.0, "score": round(score, 4)}

    if result["confidence"] < MODEL_RESPONSE_SCORE_THRESHOLD:
        return {
            "answer": "no answer found",
            "score": round(score, 4),
            "chunk": chunk
        }

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "score": round(score, 4),
        "chunk": chunk
    }
