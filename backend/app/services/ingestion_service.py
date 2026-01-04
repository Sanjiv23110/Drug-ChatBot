import os
from typing import List, Dict
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

class IngestionService:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def parse_pdf(self, file_path: str) -> str:
        try:
            reader = pypdf.PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""

    def parse_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""

    def process_directory(self, directory_path: str) -> List[Dict]:
        documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                text = ""
                if file.lower().endswith('.pdf'):
                    text = self.parse_pdf(file_path)
                elif file.lower().endswith('.txt'):
                    text = self.parse_txt(file_path)
                
                if text:
                    chunks = self.text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "id": f"{file}_{i}",
                            "text": chunk,
                            "metadata": {
                                "source": file_path,
                                "filename": file,
                                "chunk_id": i
                            }
                        })
        return documents
