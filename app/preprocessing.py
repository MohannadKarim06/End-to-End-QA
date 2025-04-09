import os, sys
import numpy as np
import faiss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.file_handler import FILE_HANDLER
from utils.text_handler import TEXT_HANDLER
from logs.logger import log_event


INDEX_DIR = "data//index//"
UPLOAD_DIR = "data//uploaded_files//"

file_handler = FILE_HANDLER()
text_handler = TEXT_HANDLER()


def process_and_index_file(file, filename):
    
    try:
        log_event("INFO", f"saving file has started")
        filepath = file_handler.save_uploaded_file(file, filename)
    except Exception as e:
        log_event("ERROR", f"an error happend while saving the file: {e}")


    try:
        log_event("INFO", f"reading file has started")
        raw_text = file_handler.read_file(filepath)
        log_event("INFO", f"the file was sucsesfully read")
    except Exception as e:
        log_event("ERROR", f"an error happend while reading the file: {e}")


    try:
        log_event("INFO", f"cleaning text has started")
        cleaned = text_handler.clean_text(raw_text)
        log_event("INFO", f"the text was sucsesfully cleaned")
    except Exception as e:
        log_event("ERROR", f"an error happend while cleaning the text: {e}")


    try:
        log_event("INFO", f"chunking text has started")
        chunks = text_handler.chunk_text(cleaned)
        log_event("INFO", f"the text was sucsesfully chunked")
    except Exception as e:
        log_event("ERROR", f"an error happend while chunking the text: {e}")


    try:
        log_event("INFO", f"generating embeddings has started")
        embeddings = text_handler.generate_embeddings(chunks)
        log_event("INFO", f"embeddings are sucssefly generated")
    except Exception as e:
        log_event("ERROR", f"an error happend while generating embeddings: {e}")


    try:
        log_event("INFO", f"creating and saving index has started")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        index_path = os.path.join(INDEX_DIR, f"{filename}.index")
        faiss.write_index(index, index_path)
        chunks_path = os.path.join(INDEX_DIR, f"{filename}_chunks.npy")
        np.save(chunks_path, np.array(chunks))
        log_event("INFO", f"index is created and saved sucssesfully")
    except Exception as e:
        log_event("ERROR", f"an error happend while creating and saving the index: {e}")



    return {
        "message": "File processed and indexed successfully.",
        "index_path": index_path,
        "chunks_path": chunks_path,
        "num_chunks": len(chunks)
    }
