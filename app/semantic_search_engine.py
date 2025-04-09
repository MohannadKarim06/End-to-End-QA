import os, sys
import faiss
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.text_handler import TEXT_HANDLER


text_handler = TEXT_HANDLER()
INDEX_DIR = "data//index//"


def load_faiss_index(filename):

    index_path = os.path.join(INDEX_DIR, f"{filename}.index")
    return faiss.read_index(index_path)

def load_chunks(filename):

    chunks_path = os.path.join(INDEX_DIR, f"{filename}_chunks.npy")
    return np.load(chunks_path, allow_pickle=True)

def search_top_chunk(question: str, filename: str):
    
    index = load_faiss_index(filename)
    chunks = load_chunks(filename)

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
