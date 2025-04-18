import re
import spacy
import numpy as np
import os, sys
from transformers import AutoTokenizer, AutoModel
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


INDEX_FOLDER = "/app/data/index"

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


class TEXT_HANDLER():
    

    def clean_text(self, text):
        """Cleans text: removes special characters, multiple spaces, and extra newlines."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  
        return text.strip()


    def chunk_text(self, text, max_tokens=512):
        
        nlp = spacy.load("en_core_web_sm")      
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


    def generate_embeddings(self, text_chunks):
        encoded_input = tokenizer(text_chunks, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1)

        return embeddings.numpy().astype("float32")

