from src.helper import doc_loader, clean_docs, filter_docs, parent_child_split_docs
from dotenv import load_dotenv
import json
import os
from pathlib import Path

load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# ── 1. Re-process the PDF ───────────────────────────────────
pdf_path = r"E:\Nihal\RAG_Projects\production_rag\production-rag\data\documents\Hands On Machine Learning with Scikit Learn and TensorFlow.pdf"

docs = doc_loader(pdf_path)
docs = clean_docs(docs)
docs = filter_docs(docs)

pc = parent_child_split_docs(docs)
print(f"{len(pc.children)} children | {len(pc.parents)} parents")

# ── 2. Save parents locally ─────────────────────────────────
PARENT_STORE_PATH = Path("data/parent_store.json")
PARENT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(PARENT_STORE_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {pid: {"page_content": p.page_content, "metadata": p.metadata}
         for pid, p in pc.parents.items()},
        f, ensure_ascii=False, indent=2
    )

# ── 3. Create NEW index (keep old "ml-chatbot" untouched) ───
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

pc_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
NEW_INDEX_NAME = "ml-chatbot-v2"   # ← new index

if not pc_client.has_index(NEW_INDEX_NAME):
    pc_client.create_index(
        name=NEW_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# ── 4. Upload children to NEW index ─────────────────────────
vectorstore = PineconeVectorStore.from_documents(
    documents=pc.children,
    embedding=embedding_model,
    index_name=NEW_INDEX_NAME,
)
print(f"Uploaded to '{NEW_INDEX_NAME}'. Old index 'ml-chatbot' is untouched.")