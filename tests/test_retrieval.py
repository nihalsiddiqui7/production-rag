from app.ingestion.loader import PDFLoader
from app.ingestion.chunker import DocumentChunker
from app.ingestion.embedder import Embedder
from app.database.vectordb import ChromaVectorDB


PDF_PATH = r"E:\Nihal\RAG_Projects\production_rag\production-rag\data\documents\mlnotes.txt"


def main():

    # Load PDF
    loader = PDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Pages Loaded: {len(documents)}")

    # Chunk Documents
    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(
        documents
    )

    print(f"Chunks Created: {len(chunks)}")

    # Extract text
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # Extract metadata
    metadatas = [
        chunk.metadata
        for chunk in chunks
    ]

    # Generate IDs
    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    # Clean text
    clean_texts = []

    for text in texts:

        if not isinstance(text, str):
            continue

        text = text.replace("\x00", "")

        text = text.strip()

        if len(text) == 0:
            continue

        clean_texts.append(text)

    print(f"Valid Chunks: {len(clean_texts)}")

    # Generate embeddings
    embedder = Embedder()

    embeddings = embedder.generate_embeddings(
        clean_texts
    )

    print("Embeddings Generated")

    # Store in Chroma
    vectordb = ChromaVectorDB()

    vectordb.add_documents(
        ids=ids,
        texts=clean_texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print("Documents added to vector database successfully!")

    # Query
    query = input("\nEnter your query: ")

    query_embedding = embedder.generate_embeddings(
        [query]
    )[0]

    results = vectordb.search(
        query_embedding
    )

    print("\nTop Results:\n")

    for doc in results["documents"][0]:

        print("=" * 80)

        print(doc[:500])

        print("\n")


if __name__ == "__main__":
    main()