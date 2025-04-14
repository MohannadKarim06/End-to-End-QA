import os
import fitz
from docx import Document


UPLOAD_FOLDER = r"data\uploaded_files"

class FILE_HANDLER:

    @staticmethod
    def extract_text_from_pdf(filepath):
        doc = fitz.open(filepath)
        return "\n".join([page.get_text("text") for page in doc])

    @staticmethod
    def extract_text_from_docx(filepath):
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])

    @staticmethod
    async def save_uploaded_file(file, filename): 
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        file_content = await file.read() 
        
        with open(filepath, "wb") as f:
            f.write(file_content)
        return filepath

    @staticmethod
    def read_file(filepath):
        ext = os.path.splitext(filepath)[-1].lower()
        if ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            return FILE_HANDLER.extract_text_from_pdf(filepath)
        elif ext == ".docx":
            return FILE_HANDLER.extract_text_from_docx(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
