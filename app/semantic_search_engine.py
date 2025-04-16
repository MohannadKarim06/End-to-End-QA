import os, sys
import faiss
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_handler import TEXT_HANDLER


text_handler = TEXT_HANDLER()
INDEX_DIR = r"data\index"


def load_faiss_index():

    index_path = os.path.join(INDEX_DIR, f"uploaded_file.index")
    return faiss.read_index(index_path)

def load_chunks():

    chunks_path = os.path.join(INDEX_DIR, f"uploaded_file_chunks.npy")
    return np.load(chunks_path, allow_pickle=True)

def search_top_chunk(question: str):
    
    index = load_faiss_index()
    chunks = load_chunks()

    question_embedding = text_handler.generate_embeddings([question]).astype("float32")

    top_k = 1
    scores, indices = index.search(question_embedding, top_k)

    top_idx = indices[0][0]
    top_score = float(scores[0][0])
    top_chunk = chunks[top_idx]

    return {
        "chunk": top_chunk,
        "score": top_score,
        "index": top_idx
    }


print(search_top_chunk("when was the first telescope invented ?"))