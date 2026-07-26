# import os

# from dotenv import load_dotenv
# load_dotenv()
# from pinecone import Pinecone

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# INDEX_NAME = "ml-chatbot"

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# vectorstore = PineconeVectorStore(
#     index_name=INDEX_NAME,
#     embedding=embedding_model
# )

# retriever = vectorstore.as_retriever(
#     search_kwargs={"k": 5}
# )


import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
INDEX_NAME = "ml-chatbot-v2"          # ← new parent-child index
PARENT_STORE_PATH = Path("data/parent_store.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Child search settings
K_CHILDREN = 10       # how many child chunks to retrieve
MAX_PARENTS = 3     # how many unique parents to return to LLM


# ═══════════════════════════════════════════════════════════════
#  LOAD PARENT STORE  (once at import time)
# ═══════════════════════════════════════════════════════════════

def _load_parent_store(path: Path = PARENT_STORE_PATH) -> dict:
    """Load parent chunks from the JSON store generated during ingestion."""
    if not path.exists():
        raise FileNotFoundError(
            f"Parent store not found at: {path}\n"
            f"Run store_index.py first to generate it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_parent_store = _load_parent_store()


# ═══════════════════════════════════════════════════════════════
#  CUSTOM RETRIEVER  (drop-in replacement for vectorstore.as_retriever)
# ═══════════════════════════════════════════════════════════════

class ParentChildRetriever:
    """
    Searches small CHILD chunks for high recall,
    but returns large PARENT chunks for rich LLM context.

    ```
    Why not vectorstore.as_retriever()?
    ------------------------------------
    The standard retriever returns the same chunks it searched.
    With parent-child chunking, we search CHILDREN (small, precise)
    but want PARENTS (large, contextual) for the LLM prompt.

    This class wraps the vectorstore and handles the resolution.

    Interface:
        retriever.invoke(query)  →  List[Document]  (parents)
        retriever.get_relevant_documents(query)  →  List[Document]  (parents)

    Both methods are fully compatible with LangChain chains and agents.
    ```
    """

    def __init__(
        self,
        vectorstore: PineconeVectorStore,
        parent_store: dict,
        k_children: int = K_CHILDREN,
        max_parents: int = MAX_PARENTS,
    ):
        self.vectorstore = vectorstore
        self.parent_store = parent_store
        self.k_children = k_children
        self.max_parents = max_parents

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        1. Search child chunks in Pinecone
        2. Deduplicate by parent_id (preserve similarity rank)
        3. Lookup parent texts from local store
        4. Return parent Documents
        """
        # 1️⃣  Search child chunks
        child_docs = self.vectorstore.similarity_search(
            query,
            k=self.k_children,
        )

        if not child_docs:
            return []

        # 2️⃣  Deduplicate parent_ids, keep first-appearance order
        seen = set()
        parent_ids = []
        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                parent_ids.append(pid)

        # 3️⃣  Fetch parent documents
        parent_docs: List[Document] = []
        for pid in parent_ids[:self.max_parents]:
            parent_data = self.parent_store.get(pid)
            if parent_data:
                parent_docs.append(
                    Document(
                        page_content=parent_data["page_content"],
                        metadata=parent_data.get("metadata", {}),
                    )
                )

        return parent_docs

    def invoke(self, query: str) -> List[Document]:
        """Alias for get_relevant_documents — matches LangChain retriever API."""
        return self.get_relevant_documents(query)

    # Optional: async support if your pipeline uses ainvoke()
    async def ainvoke(self, query: str) -> List[Document]:
        """Async wrapper — falls back to sync (Pinecone client is sync anyway)."""
        return self.get_relevant_documents(query)


# ═══════════════════════════════════════════════════════════════
#  INIT
# ═══════════════════════════════════════════════════════════════

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding_model,
)

# ── Drop-in replacement ──────────────────────────────────────
# OLD:  retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# NEW:  retriever = ParentChildRetriever(...)
#
# Every other file in your project stays exactly the same.

retriever = ParentChildRetriever(
    vectorstore=vectorstore,
    parent_store=_parent_store,
    k_children=K_CHILDREN,
    max_parents=MAX_PARENTS,
)