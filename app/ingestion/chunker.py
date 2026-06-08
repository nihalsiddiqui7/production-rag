from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import(
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

class DocumentChunker:
    def __init__(self,chunk_size = CHUNK_SIZE, CHUNK_OVERLAP = CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=CHUNK_OVERLAP
            )
        
    def chunk_documents(self, documents):
        chunks = self.splitter.split_documents(documents)


        return chunks