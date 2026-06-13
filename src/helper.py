# from langchain_community.document_loaders import PyPDFLoader
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter





# def doc_loader(pdf_path):
#     loader = PyPDFLoader(pdf_path)
    
#     return loader.load() 


# def filter_docs(docs):
#     filtered_docs = []
    
#     for doc in docs:
#         src = doc.metadata.get("source")
#         filtered_docs.append(Document(page_content=doc.page_content, metadata={"source": src,"page": doc.metadata.get("page"),"title": doc.metadata.get("title")}))
#     return filtered_docs





# def split_docs(filtered_docs):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_documents(filtered_docs)
#     return chunks



# from sentence_transformers import SentenceTransformer
# import re

# def clean_text(text):
#     text = text.replace("\x00", "")
#     text = re.sub(r'[\ud800-\udfff]', '', text)

#     text = (
#         text.encode("utf-8", errors="ignore")
#             .decode("utf-8", errors="ignore")
#     )

#     return text.strip()



# def embed_docs(chunks):
#     """
#     Purpose:
#     1. Clean PDF text
#     2. Generate embeddings
#     3. Preserve metadata
#     """

#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     cleaned_texts = []
#     metadata = []

#     for chunk in chunks:
#         text = clean_text(chunk.page_content)

#         cleaned_texts.append(text)
#         metadata.append(chunk.metadata)

#     embeddings = model.encode(
#         cleaned_texts,
#         convert_to_numpy=True,
#         show_progress_bar=True
#     )

#     return embeddings, cleaned_texts, metadata



from typing import List
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        docs
    )

    return chunks
    

