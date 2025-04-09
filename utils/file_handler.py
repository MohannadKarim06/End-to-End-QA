import os
import textract
import fitz  # PyMuPDF for PDFs
from docx import Document
import os, sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



UPLOAD_FOLDER = "data//uploaded_files//"

class FILE_HANDLER():

    def extract_text_from_pdf(filepath):
        """Extracts text from a PDF using PyMuPDF."""
        doc = fitz.open(filepath)
        text = "\n".join([page.get_text("text") for page in doc])
        return text


    def extract_text_from_docx(filepath):
        """Extracts text from a DOCX file."""
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])


    def save_uploaded_file(file, filename):
        """Saves uploaded file to the 'uploaded_files' folder."""
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(file.read())
        return filepath


    def read_file(filepath):
        """Reads text from TXT, PDF, or DOCX files."""
        ext = os.path.splitext(filepath)[-1].lower()
        
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            return FILE_HANDLER().extract_text_from_pdf(filepath)
        elif ext == ".docx":
            return FILE_HANDLER().extract_text_from_docx(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")


