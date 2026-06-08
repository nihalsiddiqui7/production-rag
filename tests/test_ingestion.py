from app.ingestion.loader import PDFLoader
from app.ingestion.chunker import DocumentChunker

PDF_PATH = "data/documents/sample.pdf"


def main():

    loader = PDFLoader(PDF_PATH)

    documents = loader.load()

    print(f"\nPages Loaded: {len(documents)}")

    chunker = DocumentChunker()

    chunks = chunker.chunk_documents(
        documents
    )

    print(f"\nChunks Created: {len(chunks)}")

    print("\nFirst Chunk:\n")

    print(chunks[0].page_content[:1000])


if __name__ == "__main__":
    main()