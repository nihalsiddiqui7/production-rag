from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    def __init__(self,pdf_path: str):
        self.pdf_path = pdf_path

    def load(self):
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"File {self.pdf_path} does not exist.")
        

        loader = PyPDFLoader(self.pdf_path)

        documents = loader.load()
        return documents
    
    
