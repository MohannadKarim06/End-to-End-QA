import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os, sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


nlp = spacy.load("en_core_web_sm")  
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  


INDEX_FOLDER = "data\index"

class TEXT_HANDLER():
    

    def clean_text(text):
        """Cleans text: removes special characters, multiple spaces, and extra newlines."""
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Remove special characters
        return text.strip()


    def chunk_text(text, max_tokens=512):
        """Splits long text into smaller chunks (512 tokens max) for transformer processing."""
        doc = nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sent in doc.sents:
            tokens = len(sent)
            if current_length + tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sent.text)
            current_length += tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


    def generate_embeddings(text_chunks):
        """Generates vector embeddings for text chunks using SentenceTransformers."""
        embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype="float32")

        return embeddings
