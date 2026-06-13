from src.helper import (
    doc_loader,
    clean_docs,
    filter_docs,
    split_docs
)


pdf_path = r"E:\Nihal\RAG_Projects\production_rag\production-rag\data\documents\Hands On Machine Learning with Scikit Learn and TensorFlow.pdf"


docs = doc_loader(pdf_path)
print(f"Loaded {len(docs)} documents.")

docs = clean_docs(docs)
print(f"Cleaned {len(docs)} documents.")

docs = filter_docs(docs)
print(f"Filtered {len(docs)} documents.")

chunks = split_docs(docs)
print(f"Split into {len(chunks)} chunks.")


from dotenv import load_dotenv
load_dotenv()
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

print("Pinecone client initialized successfully!")
index_name = "ml-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
        

)
print(f"Pinecone index '{index_name}' is ready!")

print("Uploading documents to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=index_name
)
print("Documents uploaded successfully!")