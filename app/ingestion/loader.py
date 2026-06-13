from json import load
from pathlib import Path
from unittest import loader
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import TextLoader




# class PDFLoader:
#     def __init__(self,pdf_path: str):
#         self.pdf_path = pdf_path

#     def load(self):
#         if not Path(self.pdf_path).exists():
#             raise FileNotFoundError(f"File {self.pdf_path} does not exist.")
        

#         loader = PyPDFLoader(self.pdf_path)

#         documents = loader.load()

#         return documents

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def doc_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path) 
    documents = loader.load()
    return documents
    
    
