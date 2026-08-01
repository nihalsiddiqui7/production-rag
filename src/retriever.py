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
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
INDEX_NAME = "ml-chatbot-v2"          # parent-child index
PARENT_STORE_PATH = Path("parent_store.json")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Child search settings
K_CHILDREN = 10       # how many child chunks to retrieve
MAX_PARENTS = 3       # how many unique parents to return to LLM

# Hybrid search settings
BM25_TOP_K = 50       # how many BM25 candidates feed the fusion
BM25_WEIGHT = 0.5     # BM25 leg weight in the ensemble (dense gets the rest)

# Retrieval mode: "hybrid" (dense + BM25) or "dense" (semantic only).
# Useful for A/B evaluation of the hybrid upgrade.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").lower()


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


def _build_parent_documents() -> List[Document]:
    """Materialize the parent chunks as Documents (shared by both legs)."""
    docs = []
    for parent_data in _parent_store.values():
        metadata = dict(parent_data.get("metadata", {}))
        metadata.setdefault("parent_id", "")
        docs.append(
            Document(page_content=parent_data["page_content"], metadata=metadata)
        )
    return docs


# ═══════════════════════════════════════════════════════════════
#  DENSE LEG  (semantic search over child chunks → parent chunks)
# ═══════════════════════════════════════════════════════════════

class ParentChildRetriever(BaseRetriever):
    """
    Dense retrieval: search small CHILD chunks (high recall, high precision
    embeddings) and resolve them to their PARENT chunks (rich LLM context).
    """

    vectorstore: PineconeVectorStore
    parent_store: dict
    k_children: int = K_CHILDREN

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Search child chunks, deduplicate by parent_id, return parents."""
        child_docs = self.vectorstore.similarity_search(
            query,
            k=self.k_children,
        )

        seen = set()
        parent_docs: List[Document] = []
        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            parent_data = self.parent_store.get(pid)
            if not parent_data:
                continue
            parent_docs.append(
                Document(
                    page_content=parent_data["page_content"],
                    metadata=dict(parent_data.get("metadata", {})),
                )
            )
        return parent_docs


# ═══════════════════════════════════════════════════════════════
#  HYBRID LEG  (dense + BM25 fused with Reciprocal Rank Fusion)
# ═══════════════════════════════════════════════════════════════

class HybridParentRetriever:
    """
    Merges the dense leg with a BM25 lexical leg using LangChain's
    EnsembleRetriever, which performs weighted Reciprocal Rank Fusion
    (score = weight / (rank + c), with c = 60).

    BM25 catches exact terms and abbreviations that dense embeddings miss
    ("RMSE", "L1", "Lasso"); the fusion ranks parents that both legs agree
    on higher.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        max_parents: int = MAX_PARENTS,
        bm25_weight: float = BM25_WEIGHT,
    ):
        self.max_parents = max_parents
        self.ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[1 - bm25_weight, bm25_weight],
            id_key="parent_id",
        )

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.ensemble.invoke(query)[: self.max_parents]

    def invoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

    async def ainvoke(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


# ═══════════════════════════════════════════════════════════════
#  INIT
# ═══════════════════════════════════════════════════════════════

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedding_model,
)

dense_retriever = ParentChildRetriever(
    vectorstore=vectorstore,
    parent_store=_parent_store,
    k_children=K_CHILDREN,
)

bm25_retriever = BM25Retriever.from_documents(
    _build_parent_documents(),
    k=BM25_TOP_K,
)

hybrid_retriever = HybridParentRetriever(
    dense_retriever=dense_retriever,
    bm25_retriever=bm25_retriever,
)

retriever = hybrid_retriever if RETRIEVAL_MODE == "hybrid" else dense_retriever
