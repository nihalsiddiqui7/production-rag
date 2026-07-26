from config import CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP
from typing import List, Tuple, Dict, Optional
import re
import uuid
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────────────────────────
#  PARENT-CHUNK SIZE CONFIG  (add these to your config.py)
# ──────────────────────────────────────────────────────────────
# PARENT_CHUNK_SIZE = 2000          # large context window
# PARENT_CHUNK_OVERLAP = 400        # overlap between parents
# CHUNK_SIZE = 400                  # small chunks for retrieval
# CHUNK_OVERLAP = 50                # overlap between children
#
#  Rationale:
#  - Parent chunks go to the LLM (rich context, fewer tokens wasted on borders)
#  - Child chunks go to the embedder (high granularity, better recall)
# ──────────────────────────────────────────────────────────────


def doc_loader(pdf_path: str) -> List[Document]:
    """
    Load a PDF and return a list of LangChain Documents.

    ```
    Why?
    ----
    We separate document loading from all other tasks
    so that the ingestion pipeline remains modular.

    Input:
        pdf_path -> path to PDF

    Output:
        List[Document]
    """

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    return docs


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    ```
    Why?
    ----
    PDF files often contain:
        - Null bytes
        - Invalid Unicode characters
        - Corrupted symbols

    These can cause:
        - Pinecone upload failures
        - Embedding generation errors
        - Serialization issues

    Output:
        Clean UTF-8 text
    """

    text = text.replace("\x00", "")

    text = re.sub(
        r"[\ud800-\udfff]",
        "",
        text
    )

    text = (
        text.encode(
            "utf-8",
            errors="ignore"
        )
        .decode(
            "utf-8",
            errors="ignore"
        )
    )

    return text.strip()


def clean_docs(docs: List[Document]) -> List[Document]:
    """
    Clean every document immediately after loading.

    ```
    Why?
    ----
    Cleaning early means the rest of the pipeline
    only works with valid text.

    Flow:
        Raw Documents
            ↓
        Clean Documents
            ↓
        Filtering
            ↓
        Chunking

    Output:
        List[Document]
    """

    cleaned_docs = []

    for doc in docs:

        cleaned_docs.append(
            Document(
                page_content=clean_text(
                    doc.page_content
                ),
                metadata=doc.metadata
            )
        )

    return cleaned_docs


def filter_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only metadata required for retrieval.

    ```
    Why?
    ----
    PDF loaders may generate a lot of metadata.

    Most of it is unnecessary for RAG.

    We keep:
        - source
        - page
        - title

    This reduces storage size in Pinecone.
    """

    filtered_docs = []

    for doc in docs:

        filtered_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get(
                        "source",
                        ""
                    ),
                    "page": doc.metadata.get(
                        "page",
                        0
                    ),
                    "title": doc.metadata.get(
                        "title",
                        ""
                    )
                }
            )
        )

    return filtered_docs


def split_docs(docs: List[Document]) -> List[Document]:
    """
    Split large documents into chunks.

    ```
    Why?
    ----
    Embedding models and LLMs perform better
    when information is divided into smaller,
    semantically meaningful chunks.

    Settings:
        chunk_size = 1000
        chunk_overlap = 200

    Output:
        List[Document]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(
        docs
    )

    return chunks


# ═══════════════════════════════════════════════════════════════
#  PRODUCTION UPGRADE: PARENT-CHILD CHUNKING
# ═══════════════════════════════════════════════════════════════
from config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
@dataclass
class ParentChildChunks:
    """
    Container for parent-child chunking output.

    ```
    Why?
    ----
    Parent-child chunking solves the precision-vs-context trade-off:

        - Child chunks  → embedded, indexed, searched
        - Parent chunks → stored, fetched at retrieval for LLM context

    At query time:
        1. Search child chunks (high recall, small embeddings)
        2. Deduplicate by parent_id
        3. Return parent chunks to the LLM (rich context, fewer border cuts)

    Fields:
        children:  List[Document]
            Small chunks with metadata["parent_id"] pointing to their parent.
            These are what you embed and upsert into your vector DB.

        parents:   Dict[str, Document]
            Mapping parent_id → parent chunk.
            Store these in a cheap key-value store (Redis, S3, or a
            separate Pinecone namespace with no embedding).
            At retrieval, lookup parent_id here and feed the parent text
            to the LLM instead of the child text.
    """

    children: List[Document]
    parents: Dict[str, Document]


def parent_child_split_docs(
    docs: List[Document],
    parent_chunk_size: Optional[int] = None,
    parent_chunk_overlap: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
    child_chunk_overlap: Optional[int] = None,
) -> ParentChildChunks:
    """
    Two-level hierarchical chunking.

    ```
    Why?
    ----
    Standard flat chunking cuts sentences and paragraphs at arbitrary
    boundaries.  A retrieved chunk may miss critical context that sits
    just outside its window.

    Parent-child chunking fixes this:
        - We first split into large PARENT chunks (e.g. 2000 chars).
          These preserve paragraph / section boundaries.
        - Each parent is then split into small CHILD chunks (e.g. 400 chars).
          These give the embedder high-resolution signals.
        - At retrieval we search CHILDREN but send PARENTS to the LLM.

    This is the same strategy used by LlamaIndex and production RAG
    systems at Google / OpenAI.

    Parameters:
        docs:
            Input documents (already cleaned & filtered).
        parent_chunk_size:
            Size of parent chunks.  Default = PARENT_CHUNK_SIZE from config.
        parent_chunk_overlap:
            Overlap between parent chunks.  Default = PARENT_CHUNK_OVERLAP.
        child_chunk_size:
            Size of child chunks.  Default = CHUNK_SIZE from config.
        child_chunk_overlap:
            Overlap between child chunks.  Default = CHUNK_OVERLAP.

    Returns:
        ParentChildChunks(children=..., parents=...)

    Usage in your pipeline:
    -----------------------
        docs = doc_loader("paper.pdf")
        docs = clean_docs(docs)
        docs = filter_docs(docs)

        pc = parent_child_split_docs(docs)

        # 1. Embed & index CHILDREN
        child_embeddings = embedder.encode([c.page_content for c in pc.children])
        vector_db.upsert(
            ids=[c.metadata["chunk_id"] for c in pc.children],
            vectors=child_embeddings,
            metadatas=[c.metadata for c in pc.children]
        )

        # 2. Store PARENTS cheaply (Redis, S3, or separate DB table)
        redis.mset({pid: p.page_content for pid, p in pc.parents.items()})

        # 3. At retrieval time:
        #    - query vector_db with the user question → get top-k child chunks
        #    - collect unique parent_ids from child metadata
        #    - fetch parent texts from Redis
        #    - send parent texts to the LLM
    """

    # ── defaults from config ──────────────────────────────────
    parent_chunk_size = parent_chunk_size or PARENT_CHUNK_SIZE
    parent_chunk_overlap = parent_chunk_overlap or PARENT_CHUNK_OVERLAP
    child_chunk_size = child_chunk_size or CHILD_CHUNK_SIZE
    child_chunk_overlap = child_chunk_overlap or CHILD_CHUNK_OVERLAP

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        # ^ prefer paragraph breaks, then sentence breaks.
        #   This keeps parents semantically coherent.
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n", ". ", " ", ""],
    )

    children: List[Document] = []
    parents: Dict[str, Document] = {}

    for doc in docs:
        # 1️⃣  Split into PARENTS
        parent_chunks = parent_splitter.split_documents([doc])

        for parent in parent_chunks:
            parent_id = str(uuid.uuid4())

            # Store parent with its own ID
            parent.metadata["parent_id"] = parent_id
            parent.metadata["chunk_type"] = "parent"
            parents[parent_id] = parent

            # 2️⃣  Split this parent into CHILDREN
            child_chunks = child_splitter.split_documents([parent])

            for idx, child in enumerate(child_chunks):
                child_id = str(uuid.uuid4())

                child.metadata["parent_id"] = parent_id
                child.metadata["chunk_id"] = child_id
                child.metadata["chunk_type"] = "child"
                child.metadata["child_index"] = idx
                # inherit source metadata from parent / original doc
                child.metadata.setdefault("source", doc.metadata.get("source", ""))
                child.metadata.setdefault("page", doc.metadata.get("page", 0))
                child.metadata.setdefault("title", doc.metadata.get("title", ""))

                children.append(child)

    return ParentChildChunks(children=children, parents=parents)


def get_parent_texts_from_children(
    children: List[Document],
    parents: Dict[str, Document],
) -> List[Document]:
    """
    Deduplicate children by parent_id and return the parent documents.

    ```
    Why?
    ----
    After vector search returns top-k child chunks, you usually have
    multiple children pointing to the same parent.  Sending duplicate
    parent text to the LLM wastes tokens and confuses the model.

    This helper deduplicates while preserving the original parent order
    (based on first appearance of each parent_id in the child list).

    Usage:
        retrieved_children = vector_db.similarity_search(query, k=10)
        parent_docs = get_parent_texts_from_children(retrieved_children, parents)
        # parent_docs → feed these to your LLM prompt
    """

    seen: set = set()
    result: List[Document] = []

    for child in children:
        pid = child.metadata.get("parent_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        parent = parents.get(pid)
        if parent:
            result.append(parent)

    return result